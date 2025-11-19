# Updated full script using TWO SPLIT FILES (makeup CSV + non-makeup CSV)
# Both CSVs list: <absolute_path>,<split>
# Only rows with split == 'train' are used to build dataset pairs.

import os, random, time, csv
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_geometric.loader import DataLoader as PyGLoader
import torchvision.transforms as T
import torchvision.models as models
import mediapipe as mp
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# ------------------------------
# INPUT CSVs
# ------------------------------
MAKEUP_CSV = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/make_split.csv"
NONMAKEUP_CSV = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/non_split.csv"
OUTPUT_DIR = "./training_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Face ROIs
# ============================================================
FACE_REGIONS = {
    "lips" :[0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39,37],
    "left_eye": [33, 246, 173, 133, 155, 154, 153, 145,35, 7,30,29,27,28,56,161,112,247,110,25,163,144,471,153,154,173,190,226 ,23,24,22,26],
    "right_eye": [263,390,373,374,380,381,382,362,475,386,387,388,359,441,398,255,390,373,374,398,442,257,359]
}
ROI_LABELS = {name: i+1 for i,name in enumerate(FACE_REGIONS.keys())}

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True, refine_landmarks=True, max_num_faces=1
)

# ============================================================
# load CSV helper
# ============================================================
def load_split_csv(path):
    out = []
    with open(path, 'r') as f:
        rdr = csv.reader(f)
        for row in rdr:
            if len(row) < 2: continue
            p = row[0].strip()
            sp = row[1].strip().lower()
            if sp == "train":
                out.append(p)
    return out

# ============================================================
# Landmark extraction
# ============================================================
def extract_landmark_info(image_rgb):
    H, W = image_rgb.shape[:2]
    res = mp_face_mesh.process(image_rgb)
    if not res.multi_face_landmarks:
        return None, None
    lm = res.multi_face_landmarks[0].landmark
    coords = np.array([[int(l.x * W), int(l.y * H)] for l in lm], dtype=np.int32)
    labels = np.zeros(len(coords), dtype=np.int64)
    for r, ids in FACE_REGIONS.items():
        lab = ROI_LABELS[r]
        for i in ids:
            if i < len(labels): labels[i] = lab
    return coords, labels

# ============================================================
# Fallback ROI masks
# ============================================================
def face_parsing_mask_fallback(coords, H, W, blur_ks=31, blur_sigma=9.0):
    masks, face_mask = {}, np.zeros((H, W), np.float32)
    for region, ids in FACE_REGIONS.items():
        pts = [coords[i] for i in ids if i < len(coords)]
        if len(pts) < 3:
            masks[region] = np.zeros((H, W), np.float32)
            continue
        hull = cv2.convexHull(np.array(pts))
        m = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(m, hull, 1)
        m = cv2.GaussianBlur(m.astype(np.float32), (blur_ks, blur_ks), blur_sigma)
        masks[region] = np.clip(m, 0, 1)
        face_mask = np.clip(face_mask + m, 0, 1)
    masks["face"] = face_mask
    return masks

# ============================================================
# ResNet feature extractor
# ============================================================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(
            m.conv1, m.bn1, m.relu, m.maxpool,
            m.layer1, m.layer2, m.layer3
        )
        self.out_dim = 256
    def forward(self, x):
        return self.features(x)

resnet_face = ResNetFeatureExtractor().to(device).eval()
resnet_pre = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def sample_resnet_feats(img_bgr, coords):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_bgr.shape[:2]
    x = resnet_pre(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad(): fmap = resnet_face(x)
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
    gx = (coords_t[:,0] / (W-1)) * 2 - 1
    gy = (coords_t[:,1] / (H-1)) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    feats = F.grid_sample(fmap, grid, mode='bilinear', align_corners=True)
    return feats.squeeze(-1).squeeze(0).transpose(0,1)

# ============================================================
# Graph utils
# ============================================================
def knn_edges_torch(x, k):
    N = x.shape[0]
    if N == 0: return torch.empty((2,0), dtype=torch.long)
    dist = torch.cdist(x, x)
    dist.fill_diagonal_(1e9)
    k = min(k, max(1, N-1))
    idx = dist.topk(k, largest=False).indices
    src = torch.arange(N).unsqueeze(1).repeat(1,k).reshape(-1)
    tgt = idx.reshape(-1)
    return torch.stack([src, tgt], dim=0)

def build_inter_edges_roi(N_non, region_non, region_make):
    src, tgt = [], []
    for r in region_non:
        A = region_non[r]
        B = region_make[r]
        L = min(len(A), len(B))
        for i in range(L):
            tgt.append(A[i])
            src.append(B[i] + N_non)
    if len(src)==0:
        return torch.empty((2,0), dtype=torch.long)
    return torch.tensor([src, tgt], dtype=torch.long)

def build_graph(non_xy, make_xy, non_f, make_f, rmN, rmM, k=6):
    N_non = non_xy.shape[0]
    x = torch.cat([non_f, make_f], dim=0)
    eN = knn_edges_torch(torch.tensor(non_xy, dtype=torch.float32), k)
    eM = knn_edges_torch(torch.tensor(make_xy, dtype=torch.float32), k) + N_non
    eI = build_inter_edges_roi(N_non, rmN, rmM)
    edges = []
    if eN.numel(): edges.append(eN)
    if eM.numel(): edges.append(eM)
    if eI.numel(): edges.append(eI)
    edge_index = torch.cat(edges, 1) if edges else torch.empty((2,0),dtype=torch.long)
    edge_attr = torch.zeros((edge_index.shape[1],1))
    if eI.numel(): edge_attr[-eI.shape[1]:,0]=1
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.N_non = N_non
    data.N_make = make_xy.shape[0]
    return data

# ============================================================
# Dataset using TWO CSVs
# ============================================================
class MakeupDataset(Dataset):
    def __init__(self, makeup_csv, nonmakeup_csv, limit=None):
        self.make_list = load_split_csv(makeup_csv)
        self.non_list = load_split_csv(nonmakeup_csv)
        if limit:
            self.make_list = self.make_list[:limit]
            self.non_list = self.non_list[:limit]
        if len(self.make_list)==0 or len(self.non_list)==0:
            raise RuntimeError("CSV split has no train images.")
        self.pairs = [{"makeup":m, "non":random.choice(self.non_list)} for m in self.make_list]
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]
        imM = cv2.imread(p["makeup"])
        imN = cv2.imread(p["non"])
        if imM is None or imN is None:
            return Data(x=torch.empty((0,1)))
        rgbM = cv2.cvtColor(imM, cv2.COLOR_BGR2RGB)
        rgbN = cv2.cvtColor(imN, cv2.COLOR_BGR2RGB)
        cM,lM = extract_landmark_info(rgbM)
        cN,lN = extract_landmark_info(rgbN)
        if cM is None or cN is None:
            return Data(x=torch.empty((0,1)))
        rmN = {r:[i for i in ids if i<len(cN)] for r,ids in FACE_REGIONS.items()}
        rmM = {r:[i for i in ids if i<len(cM)] for r,ids in FACE_REGIONS.items()}
        fN = sample_resnet_feats(imN,cN)
        fM = sample_resnet_feats(imM,cM)
        data = build_graph(cN,cM,fN,fM,rmN,rmM)
        data.coords_all = torch.tensor(np.vstack([cN,cM]),dtype=torch.float32)
        data.roi_labels = torch.tensor(np.concatenate([lN,lM]),dtype=torch.long)
        data.img_non = imN
        data.img_make = imM
        mask = torch.zeros(data.N_non+data.N_make,dtype=torch.long)
        mask[data.N_non:] = 1
        data.is_makeup = mask
        masks = face_parsing_mask_fallback(cN, imN.shape[0], imN.shape[1])
        data.parsing_masks = [masks]
        return data

# ============================================================
# GNN + UNet
# ============================================================
class GATv2Styler(nn.Module):
    def __init__(self, D, H=256):
        super().__init__()
        self.lin_in = nn.Linear(D, H)
        self.c1 = GATv2Conv(H, H//4, heads=4, concat=True, dropout=0.1, edge_dim=1)
        self.c2 = GATv2Conv(H, H//4, heads=4, concat=True, dropout=0.1, edge_dim=1)
        self.n1 = nn.LayerNorm(H)
        self.n2 = nn.LayerNorm(H)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.lin_out = nn.Linear(H, D)
    def forward(self,x,ei,ea):
        h = self.drop(self.act(self.lin_in(x)))
        h = self.drop(self.n1(self.act(self.c1(h,ei,ea))))
        h = self.drop(self.n2(self.act(self.c2(h,ei,ea))))
        return self.lin_out(h)+x

class SimpleUNet(nn.Module):
    def __init__(self,in_ch=64,base=64):
        super().__init__()
        def blk(ci,co): return nn.Sequential(nn.Conv2d(ci,co,3,padding=1),nn.ReLU(),nn.Conv2d(co,co,3,padding=1),nn.ReLU())
        self.e1 = blk(in_ch,base)
        self.p = nn.MaxPool2d(2)
        self.e2 = blk(base,base*2)
        self.e3 = blk(base*2,base*4)
        self.u2 = nn.ConvTranspose2d(base*4,base*2,2,stride=2)
        self.d2 = blk(base*4,base*2)
        self.u1 = nn.ConvTranspose2d(base*2,base,2,stride=2)
        self.d1 = blk(base*2,base)
        self.out = nn.Conv2d(base,3,1)
    def forward(self,x):
        e1=self.e1(x); p1=self.p(e1)
        e2=self.e2(p1); p2=self.p(e2)
        e3=self.e3(p2)
        u2=self.u2(e3)
        d2=self.d2(torch.cat([u2,e2],1))
        u1=self.u1(d2)
        d1=self.d1(torch.cat([u1,e1],1))
        return self.out(d1)

# rasterize, losses, build_edge_attr (redefined below)

def pad_to_8(t):
    B,C,H,W = t.shape
    H2 = ((H+7)//8)*8
    W2 = ((W+7)//8)*8
    return F.pad(t, (0,W2-W,0,H2-H)), (H,W)

def crop_back(t, orig):
    H,W = orig
    return t[:,:, :H, :W]

def rasterize_nodes_to_image_fast(node_feats, coords, H, W, D,
                                  sigma_px=8, device="cuda"):
    if node_feats.numel() == 0:
        return torch.zeros((D,H,W), device=device)

    coords = torch.tensor(coords, dtype=torch.float32, device=device)
    N,_ = node_feats.shape

    x = (coords[:,0] / (W-1)) * 2 - 1
    y = (coords[:,1] / (H-1)) * 2 - 1
    centers = torch.stack([x,y], dim=-1).view(N,1,1,2)

    gsize = int(max(3, sigma_px*4))
    if gsize % 2 == 0:
        gsize += 1

    ax = torch.linspace(-1,1,gsize, device=device)
    yy,xx = torch.meshgrid(ax, ax, indexing="ij")

    base = torch.exp(-(xx**2 + yy**2)/(2*(0.5**2)))
    base = (base / base.sum()).view(1,1,gsize,gsize).expand(N,1,gsize,gsize)

    gx = xx * (sigma_px/(W/2))
    gy = yy * (sigma_px/(H/2))
    local_grid = torch.stack([gx,gy], dim=-1).unsqueeze(0).expand(N,-1,-1,-1)

    grid = torch.clamp(local_grid + centers, -1, 1)

    weight = F.grid_sample(base, grid, mode='bilinear', align_corners=True)
    feats  = node_feats.unsqueeze(-1).unsqueeze(-1)

    feat_maps = feats * weight
    canvas_w  = weight.sum(0).clamp(min=1e-6)
    canvas_f  = feat_maps.sum(0)

    return canvas_f / canvas_w

# Loss utilities
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

def compute_makeup_losses(pred, ref, orig, masks, vgg_feat):
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
        m_feat = F.interpolate(m, f_pred.shape[2:], mode="bilinear", align_corners=False)
        m3 = m_feat.repeat(1,f_pred.shape[1],1,1)
        style += F.l1_loss(gram_matrix(f_pred*m3), gram_matrix(f_ref*m3))

    bg = ((pred-orig).abs() * (1-face)).mean()
    mean_shift = (pred-orig).mean(dim=[2,3])
    color_pen = (mean_shift.pow(2).sum())*0.2

    total = pix + identity + color + 0.5*style + bg + color_pen
    return dict(total=total, pix=pix, id=identity, color=color,
                style=style, bg=bg, color_pen=color_pen)

def build_edge_attr(edge_index, is_makeup_mask):
    src, tgt = edge_index
    inter = ((is_makeup_mask[src]==1) & (is_makeup_mask[tgt]==0)).float()
    return inter.unsqueeze(1).to(device)

# Visualization save helper

def visualize_graph_hybrid_roi_only_save(
        img_non, img_make,
        coordsN, coordsM,
        roi_labelsN, roi_labelsM,
        edge_index, is_make,
        out_path
    ):

    coordsN = np.asarray(coordsN, dtype=np.float32)
    coordsM = np.asarray(coordsM, dtype=np.float32)
    roi_labels = np.concatenate([roi_labelsN, roi_labelsM])
    is_make = np.asarray(is_make)

    H, W = img_non.shape[:2]
    canvas = np.zeros((H, W*2, 3), dtype=np.uint8)
    canvas[:, :W]  = img_non
    canvas[:, W:]  = img_make

    coordsM_shift = coordsM.copy()
    coordsM_shift[:,0] += W
    coords_all = np.vstack([coordsN, coordsM_shift])

    roi_mask = roi_labels != 0
    e = edge_index
    keep = roi_mask[e[0]] & roi_mask[e[1]]
    e = e[:, keep]

    for i in range(e.shape[1]):
        u = int(e[0, i]); v = int(e[1, i])
        x1, y1 = coords_all[u]; x2, y2 = coords_all[v]
        pt1 = (int(x1), int(y1)); pt2 = (int(x2), int(y2))
        if is_make[u] == 0 and is_make[v] == 0:
            color = (255, 255, 255)
        elif is_make[u] == 1 and is_make[v] == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.line(canvas, pt1, pt2, color, 1)

    for i, (x, y) in enumerate(coords_all):
        if not roi_mask[i]: continue
        pt = (int(x), int(y))
        if is_make[i] == 0:
            cv2.circle(canvas, pt, 2, (255, 255, 0), -1)
        else:
            cv2.circle(canvas, pt, 2, (0, 255, 255), -1)

    cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

# ============================================================
# Training: full forward, loss, backprop, saving outputs per epoch
# ============================================================
def init_models_from_sample(sample, proj_dim=64):
    D = sample.x.shape[1]
    encoder = GATv2Styler(D).to(device)
    projector = nn.Linear(D, proj_dim).to(device)
    unet = SimpleUNet(in_ch=proj_dim+3).to(device)
    return encoder, projector, unet, proj_dim


def train_hybrid(makeup_csv=MAKEUP_CSV, nonmakeup_csv=NONMAKEUP_CSV,
                 limit=None, epochs=100, batch_size=4, lr=1e-4,
                 proj_dim=64, out_dir=OUTPUT_DIR, use_amp=True, num_workers=0):

    os.makedirs(out_dir, exist_ok=True)
    ds = MakeupDataset(makeup_csv, nonmakeup_csv, limit)
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
                    visualize_graph_hybrid_roi_only_save(img_non, img_make, coordsN, coordsM, roiN, roiM, local_edges, is_make.cpu().numpy(), vis_path)
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

                losses = compute_makeup_losses(pred_rgb, ref_t, orig_t, roi_masks_t, vgg16)
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
        if (epoch+1) % 5 == 0:
            ckpt = {
        'encoder': encoder.state_dict(),
        'projector': projector.state_dict(),
        'unet': unet.state_dict(),
        'epoch': epoch+1
        }
            torch.save(ckpt, os.path.join(out_dir, "model_latest.pt"))


        print(f"[Epoch {epoch+1}/{epochs}] Loss={epoch_loss:.4f} Time={time.time()-t0:.1f}s -> saved to {out_dir}")
              
    return encoder, projector, unet

# ======================================
# Example run (adjust paths as needed)
# ======================================
if __name__ == '__main__':
    makeup_folder = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/images/makeup"
    nonmakeup_folder = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/images/non-makeup"
    

    train_hybrid()
    

