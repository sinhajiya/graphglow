import torch.nn as nn
import torchvision.models as models
import torch
import cv2
import torchvision.transforms as T
from torch.nn.functional import grid_sample

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    feats = grid_sample(fmap, grid, mode='bilinear', align_corners=True)
    return feats.squeeze(-1).squeeze(0).transpose(0,1)