import pandas as pd
import os, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader as PyGLoader
import torchvision.transforms as T
from models import *
from data import MakeupDataset
from grapg import build_edge_attr
from viz import viz
from rasterizer import *
from loss import compute_makeup_losses



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def init_models_from_sample(sample, proj_dim=64):
    D = sample.x.shape[1]
    encoder = GATv2Styler(D).to(device)
    projector = nn.Linear(D, proj_dim).to(device)
    unet = SimpleUNet(in_ch=proj_dim+3).to(device)
    return encoder, projector, unet, proj_dim


def train(MAKEUP_DIR, NON_MAKEUP_DIR,out_dir,
                 limit=None, epochs=100, batch_size=4, lr=1e-4,
                 proj_dim=64, use_amp=True, num_workers=0):

    os.makedirs(out_dir, exist_ok=True)
    ds = MakeupDataset(MAKEUP_DIR, NON_MAKEUP_DIR, limit=limit)
    ld = PyGLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # find first valid sample
    sample = None
    for i in range(len(ds)):
        s = ds[i]
        if hasattr(s, 'x') and s.x.numel() > 0:
            sample = s; break
    if sample is None:
        raise RuntimeError('No valid samples in dataset; check CSVs and images')

    encoder, projector, unet, _ = init_models_from_sample(sample, proj_dim=proj_dim)

    params = list(encoder.parameters()) + list(projector.parameters()) + list(unet.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-6)
    scaler = GradScaler(enabled=use_amp)

    visualized_graph_once = False

    for epoch in range(epochs):
        encoder.train(); projector.train(); unet.train()
        epoch_loss = 0.0
        t0 = time.time()
        last_vis = None

        for batch in ld:
            batch = batch.to(device)
            edge_attr = build_edge_attr(batch.edge_index, batch.is_makeup)

            with autocast(enabled=use_amp):
                out_nodes = encoder(batch.x, batch.edge_index, edge_attr)

            batch_loss = torch.tensor(0.0, device=device)
            ptr = batch.ptr.cpu().numpy() if hasattr(batch, 'ptr') else np.array([0, out_nodes.shape[0]])

            for gi in range(len(ptr)-1):
                s, e = int(ptr[gi]), int(ptr[gi+1])
                nodes_local = out_nodes[s:e]
                is_make = batch.is_makeup[s:e]
                n_non = int((is_make == 0).sum().item())
                if n_non == 0: continue

                coords_all = batch.coords_all[s:e].cpu().numpy()
                coordsN = coords_all[:n_non]; coordsM = coords_all[n_non:]
                roi_all = batch.roi_labels[s:e].cpu().numpy()
                roiN = roi_all[:n_non]; roiM = roi_all[n_non:]

                edge_mask = (batch.edge_index[0] >= s) & (batch.edge_index[0] < e)
                local_edges = batch.edge_index[:, edge_mask].cpu().numpy()

                if not visualized_graph_once:
                    img_non = batch.img_non[gi] if isinstance(batch.img_non, list) else batch.img_non
                    img_make = batch.img_make[gi] if isinstance(batch.img_make, list) else batch.img_make
                    vis_path = os.path.join(out_dir, f'graph_epoch{epoch+1}_sample{gi}.png')
                    viz(img_non, img_make, coordsN, coordsM, roiN, roiM, local_edges, is_make.cpu().numpy(), vis_path)
                    visualized_graph_once = True

                mu = nodes_local.mean(0, keepdim=True)
                sigma = nodes_local.std(0, keepdim=True) + 1e-6
                nodes_norm = (nodes_local - mu) / sigma
                proj_feats = projector(nodes_norm)

                img_non = batch.img_non[gi] if isinstance(batch.img_non, list) else batch.img_non
                img_make = batch.img_make[gi] if isinstance(batch.img_make, list) else batch.img_make

                pm = batch.parsing_masks[gi] if isinstance(batch.parsing_masks, list) else batch.parsing_masks
                if isinstance(pm, list) and len(pm) > 0 and isinstance(pm[0], dict): pm = pm[0]
                parsing_masks = pm

                H, W = img_non.shape[:2]

                image_feat = rasterize_nodes_to_image_fast(proj_feats, coords_all, H, W, proj_feats.shape[1], sigma_px=max(4, min(H,W)//32), device=device)
                if image_feat.dim() == 2: image_feat = image_feat.unsqueeze(0)
                if image_feat.shape[1:] != (H, W):
                    image_feat = F.interpolate(image_feat.unsqueeze(0), size=(H,W), mode='bilinear', align_corners=False).squeeze(0)

                orig_t = torch.tensor(cv2.cvtColor(img_non, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)/255.0, device=device).unsqueeze(0)
                ref_t  = torch.tensor(cv2.cvtColor(img_make, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2,0,1)/255.0, device=device).unsqueeze(0)

                inp = torch.cat([image_feat.unsqueeze(0), orig_t], dim=1)
                inp_pad, orig_hw = pad_to_8(inp)

                with autocast(enabled=use_amp):
                    delta_pad = unet(inp_pad)
                delta = crop_back(delta_pad, orig_hw)

                if delta.shape[2:] != orig_t.shape[2:]:
                    delta = F.interpolate(delta, size=orig_t.shape[2:], mode='bilinear')
                if delta.shape[1] == 1: delta = delta.repeat(1,3,1,1)
                elif delta.shape[1] > 3: delta = delta[:, :3]

                roi_masks_t = {}
                for k in ["lips","left_eye","right_eye","cheeks","face"]:
                    arr = parsing_masks.get(k, np.zeros((H,W)))
                    roi_masks_t[k] = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

                face_mask = roi_masks_t["face"].repeat(1,3,1,1)
                pred_rgb = torch.clamp(orig_t + delta * face_mask, 0, 1)

                losses = compute_makeup_losses(pred_rgb, ref_t, orig_t, roi_masks_t)
              
                batch_loss += losses['total']

                if gi == 0:
                    last_vis = (orig_t[0].permute(1,2,0).detach().cpu().numpy(), ref_t[0].permute(1,2,0).detach().cpu().numpy(), pred_rgb[0].permute(1,2,0).detach().cpu().numpy())

            opt.zero_grad()
            if use_amp:
                scaler.scale(batch_loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                batch_loss.backward()
                opt.step()

            epoch_loss += batch_loss.item()

        # end batches
        if last_vis is not None:
            o, r, p = last_vis
            o_img = (o*255).astype(np.uint8); r_img = (r*255).astype(np.uint8); p_img = (p*255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f'epoch{epoch+1:03d}_orig.png'), cv2.cvtColor(o_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f'epoch{epoch+1:03d}_ref.png'),  cv2.cvtColor(r_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(out_dir, f'epoch{epoch+1:03d}_pred.png'), cv2.cvtColor(p_img, cv2.COLOR_RGB2BGR))


        # save checkpoint every 5 epochs (overwrite)
        if (epoch+1) % 15 == 0:
            ckpt = {
        'encoder': encoder.state_dict(),
        'projector': projector.state_dict(),
        'unet': unet.state_dict(),
        'epoch': epoch+1
        }
            torch.save(ckpt, os.path.join(out_dir, "model_latest.pt"))


        print(f"[Epoch {epoch+1}/{epochs}] Loss={epoch_loss:.4f} Time={time.time()-t0:.1f}s -> saved to {out_dir}")
              
    return encoder, projector, unet




if __name__ == "__main__":
    epochs = 50
    MAKEUP_DIR = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/make_split.csv"
    NON_MAKEUP_DIR = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/non_split.csv"
    OUTPUT_DIR = f"./results_epochs_{epochs}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train(MAKEUP_DIR, NON_MAKEUP_DIR, OUTPUT_DIR, epochs=epochs)