import os
import sys

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import functional as tf


class Prostate(Dataset):
    def __init__(self, src_domain, domain_idx=None, base_dir=None, split='train'):
        self.base_dir = base_dir
        self.src_domain = src_domain
        self.domain_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4', 'Domain5', 'Domain6']
        self.domain_idx = domain_idx
        self.split = split
        self.id_path = [x for x in os.listdir(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image'))
                        if not x.endswith('.png')]

        print("total {} samples".format(len(self.id_path)))

    def __len__(self):
        return len(self.id_path)
    
    def _get_bbox_gt(self, mask: torch.Tensor) -> torch.Tensor:
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
        # return torch.tensor([0., 0., 512., 512.])

    def _get_bbox(self, mask: torch.Tensor, id) -> torch.Tensor:
        x_min, y_min, x_max, y_max = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], f'src_domain_idx_{self.src_domain}_best', id))
        x_min *= 2
        y_min *= 2
        x_max *= 2
        y_max *= 2

        # add perturbation to bounding box coordinates
        H, W = mask.shape[1:]
        # add perfurbation to the bbox
        assert H == W, f"{W} and {H} are not equal size!!"
        x_min = max(0, x_min - np.random.randint(0, 10))
        x_max = min(W, x_max + np.random.randint(0, 10))
        y_min = max(0, y_min - np.random.randint(0, 10))
        y_max = min(H, y_max + np.random.randint(0, 10))

        return torch.tensor([x_min, y_min, x_max, y_max])
        # return torch.tensor([0., 0., 512., 512.])

    def __getitem__(self, index):
        id = self.id_path[index]
        img = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'image', id))

        mask = np.load(os.path.join(self.base_dir, self.domain_name[self.domain_idx], 'mask', id))
        sample = {'img': img, 'mask': mask}

        img = sample['img']
        mask = sample['mask']
        # mask = mask.transpose(1,2,0)
        img = cv2.resize(img, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024))
        img = img.transpose(2, 0, 1)
        # mask = mask.transpose(2,0,1)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).unsqueeze(0)  # .long()
        min_val = torch.min(img)
        max_val = torch.max(img)
        img = (img - min_val) / (max_val - min_val)
        img = torch.round(img*255).to(torch.uint8)
        img = tf.equalize(img)
        img = img / 255
        bbox = self._get_bbox(mask, id)
        bbox_gt = self._get_bbox_gt(mask)
        return img, mask.long(), bbox, bbox_gt

if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoader

    base_dir = '/root/autodl-tmp/prostate'
    trainset = Prostate(base_dir=base_dir, src_domain=0, split='train', domain_idx=1)
    trainloader = DataLoader(trainset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    for i, (img, mask, bbox, bbox_gt) in enumerate(trainloader):
        print(bbox)
        sys.exit()
