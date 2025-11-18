from pathlib import Path
import cv2
from graphs_creation import extract_landmarks, build_graph
from features import build_features, extract_color_features
import torch
from torch_geometric.data import Data
import pandas as pd
import cv2
import torch
from torch_geometric.data import Data
from pathlib import Path

from graphs_creation import extract_landmarks, build_graph
from features import build_features, extract_color_features


class MakeupDataset:
    def __init__(self,
                 makeup_csv,
                 non_csv,
                 device="cpu",
                 split="train",
                 max_samples=None):

        self.device = device
        self.split = split.lower().strip()
        assert self.split in ["train", "test"], "split must be 'train' or 'test'"

        # Load CSVs
        # self.makeup_df = pd.read_csv(makeup_csv)
        # self.non_df    = pd.read_csv(non_csv)
        self.makeup_df = pd.read_csv(makeup_csv, header=None, sep=",")
        self.non_df    = pd.read_csv(non_csv, header=None, sep=",")

        # Filter by split
        self.makeup_df = self.makeup_df[self.makeup_df[1] == self.split]
        self.non_df    = self.non_df[self.non_df[1] == self.split]

        # Extract file paths as Path objects
        self.makeup_paths = [Path(p) for p in self.makeup_df[0].tolist()]
        self.non_paths    = [Path(p) for p in self.non_df[0].tolist()]

        if max_samples:
            self.makeup_paths = self.makeup_paths[:max_samples]
            self.non_paths    = self.non_paths[:max_samples]

        print(f"[{self.split.upper()}] Makeup: {len(self.makeup_paths)}, Non-makeup: {len(self.non_paths)}")

    def __len__(self):
        return min(len(self.makeup_paths), len(self.non_paths))

    def __getitem__(self, idx):
        img_ref = cv2.imread(str(self.makeup_paths[idx]))
        img_src = cv2.imread(str(self.non_paths[idx]))

        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
        img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)

        coords_ref, labels_ref = extract_landmarks(img_ref, self.device)
        coords_src, labels_src = extract_landmarks(img_src, self.device)

        if coords_ref is None or coords_src is None:
            return None

        edge_index = build_graph(coords_src, labels_src, coords_ref, labels_ref, k=6)

        x_src = build_features(img_src, coords_src)
        x_ref = build_features(img_ref, coords_ref)
        x_all = torch.cat([x_src, x_ref], dim=0)

        # Build target colors from ref landmarks
        target_colors = []
        for lm in coords_ref.cpu().numpy():
            feat = extract_color_features(img_ref, lm)[:3]
            target_colors.append(feat)
        target_colors = torch.stack(target_colors, dim=0)

        data = Data(
            x=x_all.to(self.device),
            edge_index=edge_index.to(self.device),
            y=target_colors.to(self.device),
            coords_src=coords_src,
            labels_src=labels_src,
            img_src=img_src,
            img_ref=img_ref
        )

        return data
