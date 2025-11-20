import numpy as np
from scipy.spatial import ConvexHull
from skimage import color, exposure
import cv2
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def histogram_match_region(source_region, reference_region):
    """
    Match histogram of source region to reference region.
    """
    matched = np.zeros_like(source_region)

    # Match each channel separately
    for ch in range(3):
        matched[:, :, ch] = exposure.match_histograms(
            source_region[:, :, ch],
            reference_region[:, :, ch]
        )

    return matched.astype(np.uint8)


def extract_region_mask(image, coords, labels, region_label):
    """Extract mask and pixels for a specific region."""
    h, w = image.shape[:2]

    mask_idx = labels.cpu().numpy() == region_label # mask index
    region_coords = coords.cpu().numpy()[mask_idx]  # roi coords; pixels mein nhi hain

    region_coords_px = region_coords.copy() # converting in pixeld
    region_coords_px[:, 0] *= w
    region_coords_px[:, 1] *= h

    try:
        hull = ConvexHull(region_coords_px)  # outer boundary around the selected region
        hull_points = region_coords_px[hull.vertices].astype(np.int32) # vertices of polygon in prev step

        mask = np.zeros((h, w), dtype=np.uint8) # mask krdia
        cv2.fillConvexPoly(mask, hull_points, 1) #  andar ka region is 1; 0 is bg

        return mask, region_coords_px  # mask -> ek label ke liye binary image; rergion_coords_px -> polygon ke vertices
    
    except:
        return None, None

# node features
def extract_color_features(image, landmark, device=device,patch=24):
    """Extract color features from image at landmark."""
    x, y = int(landmark[0] * image.shape[1]), int(landmark[1] * image.shape[0])  # landmark -> pixel
    h, w = image.shape[:2]
 
 # a square patch around every landmark
    xl = max(0, x - patch//2)   
    xr = min(w, x + patch//2)
    yl = max(0, y - patch//2)
    yr = min(h, y + patch//2)

    crop = image[yl:yr, xl:xr]
    if crop.size == 0:
        return torch.zeros(9, device=device)

# LAB color
    crop_lab = color.rgb2lab(crop / 255.0) # cropped patch -> norm -> lab
    # mean and std for each channels
    mean_lab = torch.tensor(crop_lab.mean(axis=(0, 1)), dtype=torch.float32)  # mean
    std_lab = torch.tensor(crop_lab.std(axis=(0, 1)), dtype=torch.float32) # std dev

# RGB color
    mean_rgb = torch.tensor(crop.mean(axis=(0, 1)) / 255.0, dtype=torch.float32)  

    return torch.cat([mean_lab, std_lab, mean_rgb], dim=0).to(device) # 9 dim feature vectors


def build_features(image, coords):
    # har ek node ka feature (9 dim) and stack. 
    feats = []
    for lm in coords.cpu().numpy():
        feat = extract_color_features(image, lm, patch=24)
        feats.append(feat)
    return torch.stack(feats, dim=0)