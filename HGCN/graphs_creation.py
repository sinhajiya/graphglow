import mediapipe as mp
import numpy as np
from landmarks import FACE_REGIONS, ROI_LABELS
import torch


mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    refine_landmarks=True,
    max_num_faces=1
)

def extract_landmarks(image_rgb, device="cpu"):
    h, w, _ = image_rgb.shape
    results = mp_face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None, None

    all_landmarks = np.array([[lm.x, lm.y] for lm in results.multi_face_landmarks[0].landmark])

    relevant_indices = []
    labels = []

    for region_name, idxs in FACE_REGIONS.items():
        label = ROI_LABELS[region_name]
        for idx in idxs:
            if idx < len(all_landmarks):
                relevant_indices.append(idx)
                labels.append(label)

    landmarks = all_landmarks[relevant_indices]

    coords = torch.tensor(landmarks, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    return coords, labels_tensor


def knn_edges(x, k=6):  
    device = x.device
    N = x.shape[0]
    dist = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float('inf'))
    knn_idx = dist.topk(min(k, N-1), largest=False).indices
    src = torch.arange(N, device=device).unsqueeze(1).repeat(1, min(k, N-1)).reshape(-1)
    tgt = knn_idx.reshape(-1)
    return torch.stack([src, tgt], dim=0)


def build_graph(coords_src, labels_src, coords_ref, labels_ref, k=6):
    device = coords_src.device
    N_src = coords_src.shape[0]

    intra_src = knn_edges(coords_src, k=k)
    intra_ref = knn_edges(coords_ref, k=k) + N_src

    edges_cross = []
    for roi_id in ROI_LABELS.values():
        idx_src = torch.where(labels_src == roi_id)[0]
        idx_ref = torch.where(labels_ref == roi_id)[0]

        if len(idx_src) == 0 or len(idx_ref) == 0:
            continue

        num_links = min(len(idx_src), len(idx_ref))
        for i in range(num_links):
            edges_cross.append([idx_src[i].item(), idx_ref[i].item() + N_src])

    inter_cross = (
        torch.tensor(edges_cross, dtype=torch.long, device=device).t()
        if edges_cross else torch.empty((2, 0), dtype=torch.long, device=device)
    )

    return torch.cat([intra_src, intra_ref, inter_cross], dim=1)