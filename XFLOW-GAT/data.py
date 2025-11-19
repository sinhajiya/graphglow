import pandas as pd
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
import csv
from landmarks import *
from feature_ex import sample_resnet_feats
from grapg import build_graph

import sys
sys.path.append("/home/DSE411/Documents/mlg/graphglow/HGCN")
from landmarks import FACE_REGIONS, ROI_LABELS



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

class MakeupDataset(Dataset):
    def __init__(self,
                 makeup_csv,
                 nonmakeup_csv,
                 split="train",
                 limit=None):

        self.split = split.lower().strip()
        assert self.split in ["train", "test"], "split must be 'train' or 'test'"

        # CSV format: path, split
        make_df = pd.read_csv(makeup_csv, header=None, sep=",")
        non_df  = pd.read_csv(nonmakeup_csv, header=None, sep=",")

        # select only rows matching split
        make_df = make_df[make_df[1] == self.split]
        non_df  = non_df[non_df[1] == self.split]

        # paths list
        self.make_list = make_df[0].tolist()
        self.non_list  = non_df[0].tolist()

        if limit:
            self.make_list = self.make_list[:limit]
            self.non_list  = self.non_list[:limit]

        if len(self.make_list) == 0 or len(self.non_list) == 0:
            raise RuntimeError(f"No images found for split: {self.split}")

        # random pair: makeup → random non-makeup
        self.pairs = [{"makeup": m, "non": random.choice(self.non_list)}
                      for m in self.make_list]

        print(f"[{self.split.upper()}] Makeup={len(self.make_list)}  NonMakeup={len(self.non_list)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]

        imM = cv2.imread(str(p["makeup"]))
        imN = cv2.imread(str(p["non"]))

        if imM is None or imN is None:
            return Data(x=torch.empty((0, 1)))

        rgbM = cv2.cvtColor(imM, cv2.COLOR_BGR2RGB)
        rgbN = cv2.cvtColor(imN, cv2.COLOR_BGR2RGB)

        # your existing function
        cM, lM = extract_landmark_info(rgbM)
        cN, lN = extract_landmark_info(rgbN)

        if cM is None or cN is None:
            return Data(x=torch.empty((0, 1)))

        rmN = {r: [i for i in ids if i < len(cN)] for r, ids in FACE_REGIONS.items()}
        rmM = {r: [i for i in ids if i < len(cM)] for r, ids in FACE_REGIONS.items()}

        fN = sample_resnet_feats(imN, cN)
        fM = sample_resnet_feats(imM, cM)

        data = build_graph(cN, cM, fN, fM, rmN, rmM)

        data.coords_all = torch.tensor(np.vstack([cN, cM]), dtype=torch.float32)
        data.roi_labels = torch.tensor(np.concatenate([lN, lM]), dtype=torch.long)

        data.img_non = imN
        data.img_make = imM

        mask = torch.zeros(data.N_non + data.N_make, dtype=torch.long)
        mask[data.N_non:] = 1
        data.is_makeup = mask

        masks = face_parsing_mask_fallback(cN, imN.shape[0], imN.shape[1])
        data.parsing_masks = [masks]

        return data
