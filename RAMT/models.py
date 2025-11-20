from torch_geometric.nn import GCNConv 
from torch_geometric.data import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs_creation import extract_landmarks, build_graph
from features import extract_region_mask, build_features
import cv2
from skimage import color, exposure
import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage import color, exposure
from scipy.interpolate import griddata
from landmarks import ROI_LABELS


device = "cuda" if torch.cuda.is_available() else "cpu"

# (1) Histogram-matching → produces coarse makeup
# (2) GCN → refines color locally using graph structure

# take in image -> predict lab color output 
class MakeupGCN(nn.Module):

    def __init__(self, device=device, in_channels=9, hidden_channels=128, num_layers=4):
        super(MakeupGCN, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), # 9 -> 128
            nn.BatchNorm1d(hidden_channels), # batch norm
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.gcn_layers = nn.ModuleList()   # gcn layers init
        self.batch_norms = nn.ModuleList()  

# 4 layers of gcn + batch norm
        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))

# linear layers for color pred
        self.output = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels // 2, 3) 
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.input_proj(x) # 9 -> 128

        for gcn, bn in zip(self.gcn_layers, self.batch_norms): # gcn + bn with res
            x_new = gcn(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new

        lab_colors = self.output(x)  # lab color output.
        return lab_colors

# makeup transfer network
def post_processor(model, img_src, img_ref, device="cuda"):
    """
    graph-based post-processor
    """

    model.eval()
    coords_src, labels_src = extract_landmarks(img_src, device)
    coords_ref, labels_ref = extract_landmarks(img_ref, device)

    if coords_src is None or coords_ref is None:
        return None

#prepare the res-> copy the nonmakeup image
    result = img_src.copy()
    h, w = img_src.shape[:2]

# histogram matching for each roi
    for _, region_label in ROI_LABELS.items():
        # Get masks for all labels
        src_mask, _ = extract_region_mask(img_src, coords_src, labels_src, region_label)
        ref_mask, _ = extract_region_mask(img_ref, coords_ref, labels_ref, region_label)

        if src_mask is None or ref_mask is None:
            continue

        # Extract regions pixels; both shouldhave same res
        src_region = cv2.bitwise_and(img_src, img_src, mask=src_mask)
        ref_region = cv2.bitwise_and(img_ref, img_ref, mask=cv2.resize(ref_mask, (w, h)) if ref_mask.shape != src_mask.shape else ref_mask)

        # Get non-zero pixels
        src_pixels = src_region[src_mask > 0]
        ref_pixels = ref_region[src_mask > 0]

        if len(src_pixels) == 0 or len(ref_pixels) == 0:
            continue
            
# match histogram
# exposure.match_histogram: 
        matched_pixels = np.zeros_like(src_pixels)
        for ch in range(3):
            matched_pixels[:, ch] = exposure.match_histograms(
                src_pixels[:, ch],
                ref_pixels[:, ch]
            )

# expand -> smooth -> 
# the regions around faces have sharp sa boundary toh expanding then blurring helps in a sense that bade me color transfer hojayega and then we come back and smoothing will help in smooth boundaries around lips nd eyes. 
        kernel = np.ones((9, 9), np.uint8) 
        expanded_mask = cv2.dilate(src_mask, kernel, iterations=2)
        smooth_mask = cv2.GaussianBlur(expanded_mask.astype(np.float32), (25, 25), 10)
        smooth_mask = smooth_mask[:, :, np.newaxis]

        temp_result = result.copy().astype(np.float32)
        temp_result[src_mask > 0] = matched_pixels
# blending
        result = (smooth_mask * 0.9 * temp_result + (1 - smooth_mask * 0.9) * result).astype(np.uint8)

# makeup refinement
    edge_index = build_graph(coords_src, labels_src, coords_ref, labels_ref, k=6)

    x_src = build_features(result, coords_src)  # features of blendedd result at source coordinates
    x_ref = build_features(img_ref, coords_ref) # for ref image
    x_all = torch.cat([x_src, x_ref], dim=0)

    data = Data(x=x_all.to(device), edge_index=edge_index.to(device))

    with torch.no_grad():
        gcn_refined_lab = model(data). # apply the makeup gcn to apply the makeup -> not sending the whole face in the gcn-> only the regions

    result_lab = color.rgb2lab(result)
    coords_src_px = coords_src.cpu().numpy()
    coords_src_px[:, 0] *= w
    coords_src_px[:, 1] *= h

    n_src = len(coords_src)
    gcn_refined_np = gcn_refined_lab[:n_src].cpu().numpy()
    labels_src_np = labels_src.cpu().numpy()

    for region_name, region_label in ROI_LABELS.items():
        mask_idx = labels_src_np == region_label

        if not mask_idx.any():
            continue

        region_coords = coords_src_px[mask_idx]
        region_colors = gcn_refined_np[mask_idx] #   

# gcn can only predict the lab colors at landmarks not the whole region but makeup needs to be applied across all pixels-> use griddata
        try:
            # region k around hull nikalo and then transfer the color and blur it.
            hull = ConvexHull(region_coords)
            hull_points = region_coords[hull.vertices].astype(np.int32)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull_points, 1)
            mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 5) 

            mask_coords = np.where(mask > 0.1)
            if len(mask_coords[0]) == 0:
                continue

            points = np.column_stack([mask_coords[1], mask_coords[0]])


            for ch in range(3):
                interp = griddata(region_coords, region_colors[:, ch], points,
                                method='cubic', fill_value=region_colors[:, ch].mean())

                alpha = mask[mask_coords] * 0.3  
                result_lab[mask_coords[0], mask_coords[1], ch] = (
                    alpha * interp + (1 - alpha) * result_lab[mask_coords[0], mask_coords[1], ch]
                )
        except:
            continue

    final_rgb = color.lab2rgb(result_lab)
    final_rgb = (np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)

    return final_rgb
