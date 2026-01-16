# classifier_train/train_heads.py

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Dataset: (embedding, label)
class EmbDataset(Dataset):
    def __init__(self, emb: np.ndarray, y: np.ndarray):
        self.emb = torch.from_numpy(emb).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.emb.shape[0]

    def __getitem__(self, i):
        return self.emb[i], self.y[i]


# Utility: top-1 accuracy check
@torch.no_grad()
def accuracy(head, loader, device):
    head.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = head(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


# Optimizer builder
def build_optimizer(params, name, lr, wd):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError("optim must be sgd|adamw")


# Train a linear classifier head
def train_one(name, emb, y, out_dir: Path, device, epochs, batch_size, lr, wd, val_ratio, amp, optim):
    N = emb.shape[0]
    num_classes = int(y.max()) + 1
    feat_dim = emb.shape[1]

    rng = np.random.default_rng(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    n_val = int(N * val_ratio)

    val_idx = idx[:n_val] if n_val > 0 else np.array([], dtype=np.int64)
    tr_idx = idx[n_val:] if n_val > 0 else idx

    tr_loader = DataLoader(
        EmbDataset(emb[tr_idx], y[tr_idx]),
        batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True
    )

    # Train accuracy evaluation loader
    tr_eval_loader = DataLoader(
        EmbDataset(emb[tr_idx], y[tr_idx]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    va_loader = DataLoader(
        EmbDataset(emb[val_idx], y[val_idx]),
        batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    ) if n_val > 0 else None

    # Linear head: 512-d embedding -> class logits
    head = nn.Linear(feat_dim, num_classes, bias=True).to(device)
    opt = build_optimizer(head.parameters(), optim, lr, wd)
    crit = nn.CrossEntropyLoss()

    # Mixed precision (optional)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    best = -1.0

    for ep in range(1, epochs + 1):
        head.train()
        total_loss = 0.0

        for x, yy in tr_loader:
            x = x.to(device, non_blocking=True)
            yy = yy.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                logits = head(x)
                loss = crit(logits, yy)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * x.size(0)

        tr_loss = total_loss / max(1, len(tr_loader.dataset))
        tr_acc = accuracy(head, tr_eval_loader, device)

        if va_loader is not None:
            va_acc = accuracy(head, va_loader, device)
            print(f"[{name}] ep={ep:03d} loss={tr_loss:.4f} train_acc={tr_acc:.6f} val_acc={va_acc:.6f}")
            if va_acc > best:
                best = va_acc
                torch.save(
                    {"name": name, "feat_dim": feat_dim, "num_classes": num_classes, "state_dict": head.state_dict()},
                    out_dir / f"{name}_head_best.pt"
                )
        else:
            print(f"[{name}] ep={ep:03d} loss={tr_loss:.4f} train_acc={tr_acc:.6f}")

    # Save final checkpoint
    torch.save(
        {"name": name, "feat_dim": feat_dim, "num_classes": num_classes, "state_dict": head.state_dict()},
        out_dir / f"{name}_head_last.pt"
    )
    print(f"[{name}] done | best={best:.4f}" if va_loader else f"[{name}] done")


# CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arcface_npy", type=str, required=True)
    ap.add_argument("--facenet_npy", type=str, required=True)
    ap.add_argument("--labels_npy", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--optim", type=str, default="sgd", choices=["sgd", "adamw"])
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    arc = np.load(Path(args.arcface_npy).expanduser().resolve())
    face = np.load(Path(args.facenet_npy).expanduser().resolve())
    y = np.load(Path(args.labels_npy).expanduser().resolve())

    # input checks
    assert arc.shape[0] == face.shape[0] == y.shape[0], \
        f"LEN mismatch: arc={arc.shape[0]} face={face.shape[0]} y={y.shape[0]}"
    assert arc.shape[1] == 512 and face.shape[1] == 512, "Embedding dim must be 512"
    assert y.min() == 0 and y.max() == (y.shape[0] - 1), \
        "labels must be unique per gallery image (0..N-1), assuming one image per identity."

    print(f"[INFO] device={device} | N={y.shape[0]} | classes={int(y.max()) + 1} | out={out_dir}")
    print(f"[INFO] val_ratio={args.val_ratio} (0 disables validation split)")

    # Train separate heads for ArcFace and FaceNet embeddings
    train_one("arcface", arc, y, out_dir, device, args.epochs, args.batch_size, args.lr, args.wd,
              args.val_ratio, args.amp, args.optim)
    train_one("facenet", face, y, out_dir, device, args.epochs, args.batch_size, args.lr, args.wd,
              args.val_ratio, args.amp, args.optim)


if __name__ == "__main__":
    main()
