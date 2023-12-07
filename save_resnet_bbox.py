import argparse
import sys
import monai
import numpy as np
from torch import optim
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.prostate import Prostate
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
import pdb

from utils.utils import dice_score, compute_iou, filter_mask
from models.models import CorruptionEncoder, ImageDecoder
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test(args, validSets, validLoaders, prompt_model):
    alldice = []
    for i in range(5):
        test_dice = []
        curset = validSets[i]
        save_dir = os.path.join(curset.base_dir, curset.domain_name[curset.domain_idx], f'src_domain_idx_{args.domain}_75')
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        tqdmtest = tqdm(validLoaders[i])
        ious = []
        reious = []
        prompt_model.eval()
        for iteration, (image, mask, bbox, id) in enumerate(tqdmtest):
            # box = torch.zeros_like(bbox)
            with torch.no_grad():
                _, prompt_masks = prompt_model(image.cuda())
                prompt_masks = torch.sigmoid(prompt_masks)
                dice = dice_score((prompt_masks > args.thresh).float(), mask.cuda())
                test_dice.append(dice.item())
                
                # for i in range(bbox.shape[0]):
                #     _, y_indices, x_indices = torch.where(prompt_masks[i] > args.thresh)
                #     x_min = y_min = 0
                #     x_max = y_max = 512
                #     if len(x_indices) != 0:
                #         x_min, y_min = (x_indices.min(), y_indices.min())
                #         x_max, y_max = (x_indices.max(), y_indices.max())
                #         x_min = max(0, x_min - np.random.randint(0, 10))
                #         x_max = min(512, x_max + np.random.randint(0, 10))
                #         y_min = max(0, y_min - np.random.randint(0, 10))
                #         y_max = min(512, y_max + np.random.randint(0, 10))
                #     box[i] = torch.tensor([x_min, y_min, x_max, y_max])

            prompt_masks = prompt_masks.cpu()
            prompt_masks = transforms.Resize(128)(prompt_masks)
            bbox = bbox.cpu()
            for i in range(bbox.shape[0]):
                # a = compute_iou(box[i], bbox[i])
                # ious.append(a)
                rebox = filter_mask(prompt_masks[i] > args.thresh) * 4
                b = compute_iou(rebox, bbox[i])
                reious.append(b)
                # print(a)
                # print(b)
                np.save(os.path.join(save_dir, id[i]), rebox)
            
            tqdmtest.set_postfix(eval_dice=np.mean(test_dice))
            
        # print(np.mean(ious))
        print(np.mean(reious))
        alldice.append(np.mean(test_dice))
    print('Average Dice:', np.mean(alldice))
    print(np.mean(alldice), alldice)
    return np.mean(alldice), alldice

def main():
    parser = argparse.ArgumentParser(description='SAM4Med')
    parser.add_argument('--input_size', type=int, default=512, help='the image size')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--data_path', type=str, default='data/prostate', help='path to dataset')
    parser.add_argument('--domain', type=int, default=0, help='domain id')
    parser.add_argument('--thresh', type=float, default=0.75)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # device = torch.device('cuda:' + args.gpu)

    trainset = Prostate(base_dir=args.data_path, split='train', domain_idx=args.domain)
    trainLoader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True,drop_last=False)

    testsets = []
    testLoaders = []
    for i in range(6):
        if i!= args.domain:
            testsets.append(Prostate(base_dir=args.data_path, split='train', domain_idx=i))
            testLoaders.append(DataLoader(testsets[-1], batch_size=1, shuffle=False, num_workers=4, pin_memory=True,drop_last = False))

    prompt_model = torch.load(f'prompt_models/best_{args.domain}.pth').cuda()

    test(args, testsets, testLoaders, prompt_model)
    
    tqdmbar = tqdm(trainLoader)
    ious = []
    reious = []
    save_dir = os.path.join(trainset.base_dir, trainset.domain_name[trainset.domain_idx], f'src_domain_idx_{args.domain}_75')
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    prompt_model.eval()
    for iteration, (image, mask, bbox, id) in enumerate(tqdmbar):
        # box = torch.zeros_like(bbox)
        prompt_masks = None

        with torch.no_grad():
            _, prompt_masks = prompt_model(image.cuda())
            prompt_masks = torch.sigmoid(prompt_masks)
            # for i in range(bbox.shape[0]):
            #     _, y_indices, x_indices = torch.where(prompt_masks[i] > args.thresh)
            #     x_min = y_min = 0
            #     x_max = y_max = 512
            #     if len(x_indices) != 0:
            #         x_min, y_min = (x_indices.min(), y_indices.min())
            #         x_max, y_max = (x_indices.max(), y_indices.max())
            #         x_min = max(0, x_min - np.random.randint(0, 10))
            #         x_max = min(512, x_max + np.random.randint(0, 10))
            #         y_min = max(0, y_min - np.random.randint(0, 10))
            #         y_max = min(512, y_max + np.random.randint(0, 10))
            #     box[i] = torch.tensor([x_min, y_min, x_max, y_max])
            prompt_masks = prompt_masks.cpu()
            prompt_masks = transforms.Resize(128)(prompt_masks)
            bbox = bbox.cpu()
            for i in range(bbox.shape[0]):
                # ious.append(compute_iou(box[i], bbox[i]))
                rebox = filter_mask(prompt_masks[i] > args.thresh) * 4
                reious.append(compute_iou(rebox, bbox[i]))
                np.save(os.path.join(save_dir, id[i]), rebox)

    # print(np.mean(ious))
    print(np.mean(reious))
    
        
if __name__ == '__main__':
    main()