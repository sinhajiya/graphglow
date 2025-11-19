import torch
import torchvision.models as models
import cv2
from torch.nn.functional import l1_loss, interpolate
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16]
vgg16 = vgg16.to(device).eval()

def rgb_tensor_to_lab_tensor(img):
    arr = (img[0].detach().cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    return torch.tensor(lab.transpose(2,0,1), device=img.device).unsqueeze(0)

def gram_matrix(x):
    _,C,H,W = x.shape
    Fm = x.view(C, H*W)
    return (Fm @ Fm.t()) / (C*H*W+1e-8)

def compute_makeup_losses(pred, ref, orig, masks, vgg_feat=vgg16):
    pix = 0
    for r in ["lips","left_eye","right_eye","cheeks"]:
        if r not in masks: continue
        m = masks[r]
        pix += ((pred-ref).abs() * m).mean()

    face = masks.get("face", torch.zeros_like(list(masks.values())[0]))
    mk  = sum([masks.get(r, torch.zeros_like(face)) for r in ["lips","left_eye","right_eye","cheeks"]])
    id_mask = (face-mk).clamp(0,1)
    identity = ((pred-orig).abs()*id_mask).mean()

    pred_lab = rgb_tensor_to_lab_tensor(pred)
    ref_lab  = rgb_tensor_to_lab_tensor(ref)
    color = 0
    for r in ["lips","left_eye","right_eye","cheeks"]:
        m = masks.get(r, None)
        if m is None: continue
        m3 = m.repeat(1,3,1,1)
        pred_mean = (pred_lab*m3).view(3,-1).mean(1)
        ref_mean  = (ref_lab*m3).view(3,-1).mean(1)
        color += (pred_mean-ref_mean).abs().mean()

    with torch.no_grad():
        f_ref = vgg_feat(ref)
    f_pred = vgg_feat(pred)
    style = 0
    for r in ["lips","left_eye","right_eye","cheeks"]:
        m = masks.get(r, None)
        if m is None: continue
        m_feat = interpolate(m, f_pred.shape[2:], mode="bilinear", align_corners=False)
        m3 = m_feat.repeat(1,f_pred.shape[1],1,1)
        style += l1_loss(gram_matrix(f_pred*m3), gram_matrix(f_ref*m3))

    bg = ((pred-orig).abs() * (1-face)).mean()
    mean_shift = (pred-orig).mean(dim=[2,3])
    color_pen = (mean_shift.pow(2).sum())*0.2

    total = pix + identity + color + 0.5*style + bg + color_pen
    return dict(total=total, pix=pix, id=identity, color=color,
                style=style, bg=bg, color_pen=color_pen)