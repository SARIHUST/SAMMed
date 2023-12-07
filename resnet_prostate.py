import argparse
import monai
import numpy as np
import torchvision
from torch import optim
from network import deeplabv3_resnet50

from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.prostate import Prostate
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import pickle
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
import sys

from utils.utils import dice_score


def compute_project_term(mask_scores, gt_bitmasks):
    def dice_coefficient(x, target):
        eps = 1e-5
        n_inst = x.size(0)
        x = x.reshape(n_inst, -1)
        target = target.reshape(n_inst, -1)
        intersection = (x * target).sum(dim=1)
        union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
        loss = 1. - (2 * intersection / union)
        return loss

    ms = torch.empty(0).cuda()
    gt = torch.empty(0).cuda()
    for i in range(mask_scores.shape[1]):
        ms = torch.cat(([ms, mask_scores[:, i, :, :]]))
        gt = torch.cat(([gt, gt_bitmasks[:, i, :, :]]))
    ms = ms.unsqueeze(1)
    gt = gt.unsqueeze(1)
    mask_losses_y = dice_coefficient(
        ms.max(dim=2, keepdim=True)[0],
        gt.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        ms.max(dim=3, keepdim=True)[0],
        gt.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def test(args, validLoader, unet):
    sam_trans = ResizeLongestSide(args.input_size)
    alldice = []
    dataid = 0
    for i in range(5):
        if dataid == args.domain:
            dataid += 1
        test_dice = []
        tqdmtest = tqdm(validLoader[i], ncols=150)
        for iteration, (image, mask, bbox, id) in enumerate(tqdmtest):
            image, mask = image.cuda(), mask.cuda()
            _, mask_predictions = unet(image)
            mask_predictions = torch.sigmoid(mask_predictions)
            mask_predictions = (mask_predictions > 0.75).float()
            dice = dice_score(mask_predictions, mask)
            test_dice.append(dice.detach().item())

            # Update the progress bar
            tqdmtest.set_description(f"Testing on domain{dataid}")
            tqdmtest.set_postfix(eval_dice=np.mean(test_dice))
            tqdmtest.update()
        dataid += 1
        alldice.append(np.mean(test_dice))
    print('Average Dice:', np.mean(alldice))
    return np.mean(alldice), alldice


def main():
    parser = argparse.ArgumentParser(description='SAM4Med')
    parser.add_argument('--input_size', type=int,
                        default=512, help='the image size')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='batch_size per gpu')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--epoch', type=int, default=100,
                        help='training epoch')
    parser.add_argument('--data_path', type=str,
                        default='data/prostate', help='path to dataset')
    parser.add_argument('--domain', type=int, default=0, help='domain id')

    args = parser.parse_args()
    # net = smp.Unet(
    #     encoder_name="resnet50",  # Use resnet18 as the encoder
    #     encoder_weights="imagenet",  # Use pre-trained weights on ImageNet
    #     in_channels=3,  # Input channels depend on the dataset, 3 for RGB images
    #     classes=1  # Set the number of output channels according to the number of classes in your dataset
    # )
    net = deeplabv3_resnet50(1)

    # net = torch.load('./prompt_model/best_0.pth')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda:' + args.gpu)
    net = net.to(device)

    trainset = Prostate(base_dir=args.data_path, split='train',
                        domain_idx=args.domain, transforms=None)
    trainLoader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    testset = []
    testLoader = []
    for i in range(6):
        if i != args.domain:
            testset.append(Prostate(base_dir=args.data_path,
                           split='train', domain_idx=i, transforms=None))
            testLoader.append(DataLoader(
                testset[-1], batch_size=8, shuffle=True, num_workers=4, pin_memory=True, drop_last=True))

    # sam_model = sam_model_registry[args.vit_name](checkpoint=args.sam_ckpt).cuda()

    seg_loss = monai.losses.DiceCELoss(
        sigmoid=False, squared_pred=True, reduction="mean")
    sam_trans = ResizeLongestSide(args.input_size)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=[0.9, 0.999])
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.01, verbose=True)
    writer = SummaryWriter()
    net.train()
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = torch.nn.BCELoss()
    # val_dice, alldice = test(args,testLoader,net)
    # sys.exit(0)
    maxdice, maxalldice = 0, None
    for epoch in range(args.epoch):
        epoch_loss = []
        epoch_dice = []
        tqdmbar = tqdm(trainLoader, ncols=150)
        for iteration, (image, mask, bbox, id) in enumerate(tqdmbar):
            image, mask = image.cuda(), mask.cuda()

            # save_image(mask.float(),'1.png')
            # print(bbox)
            # sys.exit(0)
            # TODO
            _, mask_predictions = net(image)

            # loss = ce_loss(mask.float(), mask_predictions)
            mask_predictions = torch.sigmoid(mask_predictions)
            # loss = compute_project_term(mask_predictions, mask)
            loss = seg_loss(mask_predictions, mask) + \
                bce_loss(mask_predictions, mask.float())
            mask_predictions = (mask_predictions > 0.9).float()
            dice = dice_score(mask_predictions, mask)

            epoch_loss.append(loss.detach().item())
            epoch_dice.append(dice.detach().item())

            # empty gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdmbar.set_description(f"Epoch:{epoch + 1}/{args.epoch}")
            tqdmbar.set_postfix(loss=np.mean(epoch_loss),
                                dice=np.mean(epoch_dice))
            tqdmbar.update()
        if (epoch+1) % 5 == 0:
            val_dice, all_dice = test(args, testLoader, net)
            if val_dice > maxdice:
                maxdice = val_dice
                maxalldice = all_dice
                torch.save(net, f'./prompt_models/best_{args.domain}.pth')
    #     writer.add_scalars("loss",{"train": round(np.mean(epoch_loss), 4),},epoch,)
    #     writer.add_scalars("dice",{"train": round(np.mean(epoch_dice), 4),"val": round(val_dice, 4),},epoch,)
    print('Final result. Best Dice:', maxdice, maxalldice)
    torch.save(net, f'./prompt_model/final_{args.domain}.pth')
    with open('result2.txt', 'a') as f:
        f.write('Training on domain {} \n'.format(args.domain))
        f.write(str(maxdice)+'\n')
        f.write(','.join([str(x) for x in maxalldice])+'\n')


if __name__ == '__main__':
    main()
