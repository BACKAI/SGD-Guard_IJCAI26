import os
import clip
import h5py
import argparse
import io
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from diff_jpeg import diff_jpeg_coding

from ops_transforms import (
    input_brightness, input_color, input_contrast, input_crop, input_gamma,
    input_hue, input_rotate, input_saturation, input_scale,
    input_sharpness, input_translateX, input_translateY, OPS_DEFAULT_ARGS
)

from utils import *
from attack_tools import gen_pgd_confs
from peft import PeftModel
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler
from diffusers import logging as diff_logging
diff_logging.set_verbosity_error()

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image


parser = argparse.ArgumentParser(description='Fast Smart Attack with DA-EOT & Diff-JPEG')

parser.add_argument('--t', default=4, type=int, help='LCM Inference Steps')
parser.add_argument('--alpha', default=2, type=float, help='Step size')
parser.add_argument('--iter', default=5, type=int, help='Iterations')
parser.add_argument('--epsilon', default=3, type=int, help='Perturbation budget')
parser.add_argument('--id_lambda', default=5.0, type=float, help='Weight for ID Disruption')
parser.add_argument('--gpu_ids', default='0', type=str)

parser.add_argument('--arcface_path', type=str, default="../pretrained/arcface_checkpoint.tar")
parser.add_argument('--farl_path', type=str, default="../pretrained/FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
parser.add_argument('--gallery_file', type=str, default="../pretrained/gallery_20k.h5")
parser.add_argument('--lora_path', type=str, default="../ckpt/adapter_model.safetensors")
parser.add_argument('--data_path', type=str, default='../my_dataset')
parser.add_argument('--output_dir', type=str, default='../output')

parser.add_argument('--lambda_da', type=float, default=1.0, help='Weight for DA-EOT loss')
parser.add_argument('--lambda_jpeg', type=float, default=1.0, help='Weight for JPEG loss')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def normalize(t): return t / t.norm(dim=-1, keepdim=True)



def find_similar_images(gallery, text, descending=True):
    gallery = gallery.float()
    text = text.float()
    sim = (gallery @ text.T).squeeze()
    return sim.argsort(descending=descending), sim



def extract_id_diff(model, img): 
    img = F.interpolate(img, (112, 112), mode='bilinear', align_corners=False)
    return normalize(model((img.float() - 0.5) / 0.5))



def extract_clip_diff(model, img, dev): 
    img = F.interpolate(img, (224, 224), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.481, 0.457, 0.408], device=dev).view(1,3,1,1)
    std = torch.tensor([0.268, 0.261, 0.275], device=dev).view(1,3,1,1)
    norm_img = (img.to(model.dtype) - mean) / std
    return normalize(model.encode_image(norm_img))



def get_clip_loss(feat, pos, neg):
    sim_pos = (feat * pos).sum()
    sim_neg = (feat * neg).sum()
    if sim_pos < sim_neg: return (1 - sim_pos) + sim_neg, "Pos"
    else: return (1 - sim_neg) + sim_pos, "Neg"



def load_transform_mean_vectors(npz_path, device):
    data = np.load(npz_path)
    names = data["transform_names"]
    vecs = torch.from_numpy(data["mean_vectors"]).float().to(device)
    vecs = F.normalize(vecs, dim=1)
    names = [str(n) for n in names.tolist()]
    return names, vecs



def apply_single_transform_unit01(x, tname, OPS=OPS_DEFAULT_ARGS):
    x_range = x * 2.0 - 1.0
    B, C, H, W = x_range.shape
    params = {
        "batch_size": B, "image_height": int(H), "image_width": int(W), "image_resize": int(max(H, W)),
    }

    if tname == "brightness": x_t = input_brightness(x_range, factor_delta=OPS['Brightness'][0])
    elif tname == "color": x_t = input_color(x_range, factor_delta=OPS['Color'][0])
    elif tname == "contrast": x_t = input_contrast(x_range, factor_delta=OPS['Contrast'][0])
    elif tname == "crop": x_t = input_crop(x_range, params=params)
    elif tname == "gamma": x_t = input_gamma(x_range, delta=OPS['Gamma'][0])
    elif tname == "hue": x_t = input_hue(x_range, delta=OPS['Hue'][0])
    elif tname == "rotate": x_t = input_rotate(x_range, theta=OPS['Rotate'][0])
    elif tname == "saturation": x_t = input_saturation(x_range, delta=OPS['Saturation'][0])
    elif tname == "scale": x_t = input_scale(x_range)
    elif tname == "sharpness": x_t = input_sharpness(x_range, factor_delta=OPS['Sharpness'][0])
    elif tname == "translateX": x_t = input_translateX(x_range, delta=OPS['TranslateX'][0], params=params)
    elif tname == "translateY": x_t = input_translateY(x_range, delta=OPS['TranslateY'][0], params=params)
    else: x_t = x_range

    x_t = (x_t + 1.0) * 0.5
    x_t = torch.clamp(x_t, 0.0, 1.0)
    return x_t



def jpeg_forward_only(x_01: torch.Tensor, q: int) -> torch.Tensor:
    device = x_01.device
    x_cpu = x_01.detach().clamp(0.0, 1.0).cpu()
    B = x_cpu.size(0)
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    out_list = []
    for b in range(B):
        img = to_pil(x_cpu[b])
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(q))
        buf.seek(0)
        img_jpeg = Image.open(buf).convert("RGB")
        out_list.append(to_tensor(img_jpeg))
    out = torch.stack(out_list, dim=0).to(device)
    return out



class Diffusion_Purifier(torch.nn.Module):
    def __init__(self, model_id, lora_path, device, steps=4):
        super().__init__()
        self.device = device
        self.steps = steps
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, safety_checker=None, requires_safety_checker=False, torch_dtype=torch.float32 
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)
        
        if os.path.isfile(lora_path):
            lora_dir = os.path.dirname(lora_path)
        else:
            lora_dir = lora_path
            
        self.pipe.unet = PeftModel.from_pretrained(self.pipe.unet, lora_dir)
        self.pipe.unet.merge_and_unload()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.vae.to(dtype=torch.float32)


    def purify(self, x):
        x_input = x.detach().cpu().clamp(0, 1)
        pil_img = to_pil_image(x_input[0])
        pil_img_512 = pil_img.resize((512, 512), Image.BICUBIC)
        
        with torch.no_grad():
            latents = self.pipe(
                prompt="", image=pil_img_512, num_inference_steps=self.steps,
                strength=0.4, guidance_scale=1.0, output_type="latent", return_dict=False
            )[0]
            images = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
            images = (images / 2 + 0.5).clamp(0, 1)

        images_resized = F.interpolate(images, size=(x.shape[2], x.shape[3]), mode='bicubic', align_corners=False, antialias=True)
        return images_resized.to(self.device)



def load_diffusion_model(lora_path, device, steps):
    purifier = Diffusion_Purifier(
        model_id="runwayml/stable-diffusion-v1-5", lora_path=lora_path, device=device, steps=steps
    )
    return purifier



def generate_fast_adv(x, clip_centers, feat_id_orig, purifier, arcface, clip_model, pgd_conf, 
                      transform_names, mean_vecs):
    
    delta = torch.zeros_like(x).to(device)
    eps = pgd_conf['eps']
    alpha = pgd_conf['alpha']
    targets = ["eyebrows", "eyes", "nose", "lips"]
    quality_list = [70.0, 50.0, 30.0] 
    
    K_da = len(transform_names)

    for i in range(pgd_conf['iter']):
        x_adv = x + delta
        
        x_purified = purifier.purify(x_adv)
        x_purified = x_purified.detach()
        x_purified.requires_grad_()
        
        feat_c = extract_clip_diff(clip_model, x_purified, device)
        feat_i = extract_id_diff(arcface, x_purified)
        
        c_losses = []
        for attr in targets:
            l, _ = get_clip_loss(feat_c, clip_centers[attr]['pos'], clip_centers[attr]['neg'])
            c_losses.append(l)
        loss_tensor = torch.stack(c_losses)
        weights = F.softmax(loss_tensor, dim=0)
        clip_loss = (weights * loss_tensor).sum()
        
        id_loss_purified = (feat_i * feat_id_orig).sum()
        
        main_loss = clip_loss + (opt.id_lambda * id_loss_purified)
        loss_da = torch.zeros(1, device=device)
        
        if opt.lambda_da > 0:
            with torch.no_grad():
                feat_curr = extract_id_diff(arcface, x_adv)
                g_feat = F.normalize(feat_curr - feat_id_orig, dim=1) 
                cos = torch.matmul(g_feat, mean_vecs.t()).squeeze(0)  
                w = torch.softmax(cos, dim=0) 
            imgs_t = []
            for k in range(K_da):
                x_t = apply_single_transform_unit01(x_adv, transform_names[k], OPS_DEFAULT_ARGS)
                imgs_t.append(x_t)
            
            batch_imgs_da = torch.cat(imgs_t, dim=0)
            feat_batch_da = extract_id_diff(arcface, batch_imgs_da) 
            sim_batch_da = (feat_batch_da * feat_id_orig).sum(dim=1) 
            loss_da = (w * sim_batch_da).sum()

        loss_jpeg = torch.zeros(1, device=device)
        
        if opt.lambda_jpeg > 0:
            loss_list = []
            for q in quality_list:
                q_tensor = torch.tensor([q], device=device, dtype=torch.float32)
                with torch.no_grad():
                    x_jpeg_fwd = jpeg_forward_only(x_adv, int(q))
                x_adv_255 = x_adv * 255.0
                x_jpeg_diff_255 = diff_jpeg_coding(image_rgb=x_adv_255, jpeg_quality=q_tensor, ste=True)
                x_jpeg_diff = torch.clamp(x_jpeg_diff_255 / 255.0, 0.0, 1.0)
                x_jpeg_ste = x_jpeg_fwd + (x_jpeg_diff - x_jpeg_diff.detach())
                feat_jpeg = extract_id_diff(arcface, x_jpeg_ste)
                loss_list.append((feat_jpeg * feat_id_orig).sum())
            loss_jpeg = torch.stack(loss_list).mean()

        total_loss = main_loss + (opt.lambda_da * loss_da) + (opt.lambda_jpeg * loss_jpeg)
        
        total_loss.backward()
        grad = x_purified.grad.data
        
        if x_adv.grad is not None:
            grad += x_adv.grad.data
            x_adv.grad.zero_()

        delta -= grad.sign() * alpha
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(x + delta, 0, 1) - x
        
    return torch.clamp(x + delta, 0, 1).detach()



def run():
    if not os.path.exists(opt.output_dir): os.makedirs(opt.output_dir)
    save_path = opt.output_dir
    if not save_path.endswith('/') and not save_path.endswith('\\'):
        save_path += '/'

    print("Loading Models...")
    arcface = torch.load(opt.arcface_path, map_location=device).eval().to(device)
    purifier = load_diffusion_model(opt.lora_path, device, opt.t)

    clip_model, _ = clip.load("ViT-B/16", device=device)
    try: clip_model.load_state_dict(torch.load(opt.farl_path, map_location=device)['state_dict'], strict=False)
    except: pass
    
    print("Loading DA-EOT Vectors...")
    transform_names, mean_vecs = load_transform_mean_vectors(opt.transform_npz, device)

    print("Preparing Centers...")
    with h5py.File(opt.gallery_file, 'r') as f:
        g_clip = torch.from_numpy(f['gallery_clip'][:]).to(device).float()
    
    clip_centers = {}
    targets = ["eyebrows", "eyes", "nose", "lips"]
    with torch.no_grad():
        for attr in targets:
            tf = normalize(clip_model.encode_text(clip.tokenize(attr).to(device))).mean(0, True)
            idx, _ = find_similar_images(g_clip, tf)
            clip_centers[attr] = {
                'pos': normalize(g_clip[idx[:10]].mean(0, True)),
                'neg': normalize(g_clip[idx[-10:]].mean(0, True))
            }

    print("Loading Data...")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    ds = torchvision.datasets.ImageFolder(opt.data_path, transform=tf)
    pgd = gen_pgd_confs(eps=opt.epsilon, alpha=opt.alpha, iter=opt.iter, input_range=(0,1))
    
    print(f"Start Fast Smart Attack (with DA-EOT={opt.lambda_da}, JPEG={opt.lambda_jpeg})...")
    for i in tqdm(range(len(ds))):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(device)
        
        with torch.no_grad():
            id_orig = extract_id_diff(arcface, x.float())
            
        x_adv = generate_fast_adv(x, clip_centers, id_orig, purifier, arcface, clip_model, pgd,
                                  transform_names, mean_vecs)
        
        with torch.no_grad():
            pred_x0 = purifier.purify(x_adv)
        
        si(x_adv, save_path + f'{i}_adv.png')
        pkg = {'x': x.cpu(), 'x_adv': x_adv.cpu(), 'pred_x0': pred_x0.cpu()}
        torch.save(pkg, save_path + f'{i}.bin')
        si(torch.cat([x, x_adv, pred_x0], -1), save_path + f'{i}_all.png')
        si(pred_x0, save_path + f'{i}_final.png')

    print(f"Saved images to {save_path}")

if __name__ == "__main__":
    run()