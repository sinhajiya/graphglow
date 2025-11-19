import torch
from torch.nn.functional import grid_sample, pad

def pad_to_8(t):
    B,C,H,W = t.shape
    H2 = ((H+7)//8)*8
    W2 = ((W+7)//8)*8
    return pad(t, (0,W2-W,0,H2-H)), (H,W)

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

    weight = grid_sample(base, grid, mode='bilinear', align_corners=True)
    feats  = node_feats.unsqueeze(-1).unsqueeze(-1)

    feat_maps = feats * weight
    canvas_w  = weight.sum(0).clamp(min=1e-6)
    canvas_f  = feat_maps.sum(0)

    return canvas_f / canvas_w