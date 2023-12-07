import argparse
import sys
import monai
import numpy as np
from torch import optim
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.prostate_resnet_bbox import Prostate
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

from utils.utils import dice_score, compute_iou
from models.models import CorruptionEncoder, ImageDecoder
import random

def test(args, validLoader, sam_model, prompt_model):
    sam_trans = ResizeLongestSide(args.input_size)
    alldice = []
    dataid = 0
    for i in range(5):
        if dataid == args.domain:
            dataid+=1
        test_dice = []
        # tqdmtest = tqdm(validLoader[i])
        tqdmtest = tqdm(validLoader[-1])
        ious = []
        new_ious = []
        prompt_model.eval()
        for iteration, (image, mask, bbox, bbox_gt, id) in enumerate(tqdmtest):
            box = torch.zeros_like(bbox)
            with torch.no_grad():
                _, prompt_masks = prompt_model(image.cuda())
                prompt_masks = torch.sigmoid(prompt_masks)
                dice = dice_score((prompt_masks > args.thresh).float(), mask.cuda())
                test_dice.append(dice.detach().item())
                pdb.set_trace()
                # save_image((prompt_masks > args.thresh).float(), f'imgs/resnet_domain_{dataid}_iter_{iteration}.png')
                # save_image(mask.float(), f'imgs/gt_domain_{dataid}_iter_{iteration}.png')
                for i in range(bbox.shape[0]):
                    _, y_indices, x_indices = torch.where(prompt_masks[i] > args.thresh)
                    x_min = y_min = 0
                    x_max = y_max = 512
                    if len(x_indices) != 0:
                        x_min, y_min = (x_indices.min(), y_indices.min())
                        x_max, y_max = (x_indices.max(), y_indices.max())
                        x_min = max(0, x_min - np.random.randint(0, 10))
                        x_max = min(512, x_max + np.random.randint(0, 10))
                        y_min = max(0, y_min - np.random.randint(0, 10))
                        y_max = min(512, y_max + np.random.randint(0, 10))
                    box[i] = torch.tensor([x_min, y_min, x_max, y_max])

            for i in range(box.shape[0]):
                ious.append(compute_iou(box[i], bbox_gt[i]))
                new_ious.append(compute_iou(bbox[i], bbox_gt[i]))

            tqdmtest.set_description(f"Testing on domain{dataid}")
            tqdmtest.set_postfix(eval_dice=np.mean(test_dice))
            
        dataid+=1
        print(np.mean(ious))
        print(np.mean(new_ious))
        alldice.append(np.mean(test_dice))
    print('Average Dice:', np.mean(alldice))
    print(np.mean(alldice), alldice)
    return np.mean(alldice), alldice

def main():
    parser = argparse.ArgumentParser(description='SAM4Med')
    parser.add_argument('--input_size', type=int, default=512, help='the image size')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='vit model of sam')
    parser.add_argument('--sam_ckpt', type=str, default='SAM/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/prostate', help='path to dataset')
    parser.add_argument('--domain', type=int, default=4, help='domain id')
    parser.add_argument('--thresh', type=float, default=0.75, help='thresh hold')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # device = torch.device('cuda:' + args.gpu)

    trainset = Prostate(base_dir=args.data_path, src_domain=0, split='train', domain_idx=args.domain)
    trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)

    testsets = []
    testLoaders = []
    for i in range(6):
        if i != args.domain:
            testsets.append(Prostate(base_dir=args.data_path, src_domain=0, split='train', domain_idx=i))
            testLoaders.append(DataLoader(testsets[-1], batch_size=4, shuffle=False, num_workers=4, pin_memory=True,drop_last = True))

    sam_model = sam_model_registry[args.vit_name](checkpoint=args.sam_ckpt).cuda()
    prompt_model = torch.load(f'/root/projects/SAMMed/prompt_model/best_{args.domain}.pth').cuda()
    prompt_model.eval()

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    sam_trans = ResizeLongestSide(args.input_size)

    optimizer = optim.Adam(sam_model.parameters(), lr=args.lr, weight_decay=0.001)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.01, verbose=True)
    writer = SummaryWriter()
    sam_model.train()
    test(args, testLoaders, sam_model, prompt_model)
    maxdice, maxalldice = 0 , None
    max_epoch = 0
    for epoch in range(args.epoch):
        epoch_loss = []
        epoch_dice = []
        tqdmbar = tqdm(trainLoader)
        ious = []
        new_ious = []
        for iteration, (image, mask, bbox, bbox_gt, id) in enumerate(tqdmbar):
            box = torch.zeros_like(bbox)
            prompt_masks = None
            
            with torch.no_grad():
                _, prompt_masks = prompt_model(image.cuda())
                prompt_masks = torch.sigmoid(prompt_masks)
                # save_image((prompt_masks > args.thresh).float(), f'imgs/resnet_train_iter_{iteration}.png')
                # save_image(mask.float(), f'imgs/gt_train_iter_{iteration}.png')
                for i in range(bbox.shape[0]):
                    _, y_indices, x_indices = torch.where(prompt_masks[i] > args.thresh)
                    x_min = y_min = 0
                    x_max = y_max = 512
                    if len(x_indices) != 0:
                        x_min, y_min = (x_indices.min(), y_indices.min())
                        x_max, y_max = (x_indices.max(), y_indices.max())
                        x_min = max(0, x_min - np.random.randint(0, 10))
                        x_max = min(512, x_max + np.random.randint(0, 10))
                        y_min = max(0, y_min - np.random.randint(0, 10))
                        y_max = min(512, y_max + np.random.randint(0, 10))
                    box[i] = torch.tensor([x_min, y_min, x_max, y_max])

            for i in range(box.shape[0]):
                ious.append(compute_iou(box[i], bbox_gt[i]))
                new_ious.append(compute_iou(bbox[i], bbox_gt[i]))

        print(np.mean(ious))
        print(np.mean(new_ious))
        exit()

if __name__ == '__main__':
    main()