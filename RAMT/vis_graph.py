import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from data import MakeupDataset
from graphs_creation import extract_landmarks, build_graph


def draw_inter_face_graph(img_src, img_ref,
                          coords_src, coords_ref,
                          edge_index,
                          node_size=1,
                          edge_thickness=1):

    # combine images side-by-side
    h1, w1, _ = img_src.shape
    h2, w2, _ = img_ref.shape
    H = max(h1, h2)
    W = w1 + w2

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # place images
    canvas[:h1, :w1] = img_src
    canvas[:h2, w1:w1+w2] = img_ref

    # convert normalized coords → pixel coords
    def to_pixel(coords, w, h):
        pts = coords.clone().cpu().numpy()
        pts[:, 0] = pts[:, 0] * w
        pts[:, 1] = pts[:, 1] * h
        return pts.astype(np.int32)

    pts_src = to_pixel(coords_src, w1, h1)
    pts_ref = to_pixel(coords_ref, w2, h2)

    # shift ref points right by w1 pixels
    pts_ref_shifted = pts_ref.copy()
    pts_ref_shifted[:, 0] += w1

    N_src = len(pts_src)

    # draw edges
    edges = edge_index.t().cpu().numpy()
    for s, t in edges:
        p1 = pts_src[s] if s < N_src else pts_ref_shifted[s - N_src]
        p2 = pts_src[t] if t < N_src else pts_ref_shifted[t - N_src]

        cv2.line(canvas, tuple(p1), tuple(p2), (0, 255, 0), edge_thickness)

    # draw nodes
    for p in pts_src:
        cv2.circle(canvas, tuple(p), node_size, (0, 0, 255), -1)  # red
    for p in pts_ref_shifted:
        cv2.circle(canvas, tuple(p), node_size, (255, 0, 0), -1)  # blue

    plt.figure(figsize=(10, 8))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return canvas


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    MAKEUP_DIR = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/make_split.csv"
    NON_MAKEUP_DIR = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/non_split.csv"

    dataset = MakeupDataset(
       MAKEUP_DIR,NON_MAKEUP_DIR,
        device=device,
        split="test",
        max_samples=5
    )

    idx = 0
    sample = dataset[idx]
    if sample is None:
        print("Invalid sample")
        exit()

    img_src = sample.img_src
    img_ref = sample.img_ref

    coords_src = sample.coords_src
    labels_src = sample.labels_src

    coords_ref, labels_ref = extract_landmarks(img_ref, device)

    edge_index = build_graph(coords_src, labels_src, coords_ref, labels_ref, k=6)

    result = draw_inter_face_graph(
        img_src, img_ref,
        coords_src, coords_ref,
        edge_index
    )

    cv2.imwrite("inter_face_graph.png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     MAKEUP_DIR = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/make_split.csv"
#     NON_MAKEUP_DIR = "/home/DSE411/Documents/mlg/project/makeup_dataset/all/cache/non_split.csv"

#     dataset = MakeupDataset(
#        MAKEUP_DIR,NON_MAKEUP_DIR,
#         device=device,
#         split="test",
#         max_samples=None
#     )

#     idx = 0   
#     sample = dataset[idx]
#     if sample is None:
#         exit()

#     img_src = sample.img_src
#     img_ref = sample.img_ref

#     coords_src = sample.coords_src
#     labels_src = sample.labels_src

#     coords_ref, labels_ref = extract_landmarks(img_ref, device)

#     edge_index = build_graph(
#         coords_src,
#         labels_src,
#         coords_ref,
#         labels_ref,
#         k=6
#     )

#     output_img = draw_face_graph(
#         img_src,
#         coords_src,
#         coords_ref,
#         edge_index,
#         node_size=4,
#         edge_thickness=1
#     )

#     cv2.imwrite("graph_vis.png", cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
#     print("Saved graph_vis.png")
