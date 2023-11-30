import sys

import torch
import os
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms


class Kits19(Dataset):
    def __init__(self, kind, location='data_npy', tuning=False, tune_size=0.1, transform=None, normalize=False):
        """
        kind must be either 'train' or 'valid'
        """
        self.kind = kind
        self.root = os.path.join(location, kind)
        self.img_path = os.path.join(self.root, 'image')
        self.seg_path = os.path.join(self.root, 'segmentation')
        self.tuning = tuning
        self.total = 37250 if self.kind == 'train' else 7922
        self.tune_list = []
        self.tune_cnt = 0
        self.transform = transform
        self.normalization = normalize
        if self.normalization:
            self.normalize = transforms.Normalize((0.5,), (0.5,))

        if self.tuning:
            self.tune_cnt = int(self.total * tune_size)
            self.tune_list = np.random.choice(self.total, self.tune_cnt, replace=False)

    def __len__(self):
        num = self.tune_cnt if self.tuning else self.total
        return num

    def _get_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        if torch.max(mask) == 0:#label on it
            return torch.tensor([0,0,0,0])
        _, y_indices, x_indices = torch.where(mask > 0)
        x_min, y_min = (x_indices.min(), y_indices.min())
        x_max, y_max = (x_indices.max(), y_indices.max())

        # add perturbation to bounding box coordinates
        H, W = mask.shape[1:]
        # add perfurbation to the bbox
        assert H == W, f"{W} and {H} are not equal size!!"
        x_min = max(0, x_min - np.random.randint(0, 10))
        x_max = min(W, x_max + np.random.randint(0, 10))
        y_min = max(0, y_min - np.random.randint(0, 10))
        y_max = min(H, y_max + np.random.randint(0, 10))

        return torch.tensor([x_min, y_min, x_max, y_max])

    def __getitem__(self, idx):
        idx = self.tune_list[idx] if self.tuning else idx

        img_id = "{:05d}.npy".format(idx)
        final_img_path = os.path.join(self.img_path, img_id.format(idx))
        final_seg_path = os.path.join(self.seg_path, img_id.format(idx))
        img = torch.tensor(np.load(final_img_path), dtype=torch.float32)
        seg = torch.tensor(np.load(final_seg_path), dtype=torch.uint8)[1:] #seg[0]background, seg[1]forground
        bbox = self._get_bbox(seg)
        if self.normalization:
            img = self.normalize(img)

        if self.transform is not None:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            img = self.transform(img)

            random.seed(seed)
            torch.manual_seed(seed)
            seg = self.transform(seg)

        return img, seg, bbox
