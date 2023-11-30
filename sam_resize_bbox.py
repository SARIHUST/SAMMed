import argparse
import sys
import monai
import numpy as np
from torch import optim
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.prostate_resize import Prostate
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

from utils.utils import dice_score
from models.models import CorruptionEncoder, ImageDecoder
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def test(args, validLoader, sam_model):
    sam_trans = ResizeLongestSide(args.input_size)
    alldice = []
    dataid = 0
    for i in range(5):
        if dataid == args.domain:
            dataid+=1
        test_dice = []
        tqdmtest = tqdm(validLoader[i])
        for iteration, (image, mask, bbox, bbox_gt) in enumerate(tqdmtest):
            box = sam_trans.apply_boxes(bbox, (1024, 1024))
            box_tensor = torch.as_tensor(box, dtype=torch.float).cuda()
            image = image.cuda()
            mask = mask.cuda()
            
            with torch.no_grad():
                image_embeddings = sam_model.image_encoder(image)  # (B,256,64,64)

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_tensor,
                    masks=None,
                )
                
            low_res_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,  # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            mask_predictions= sam_model.postprocess_masks(
                low_res_predictions,
                input_size=[1024,1024],
                original_size=[1024,1024],
            )
            
            predicted_masks = mask_predictions
            predicted_masks = torch.sigmoid(predicted_masks) > 0.5
            dice = dice_score(predicted_masks, mask)
            test_dice.append(dice.detach().item())

            # Update the progress bar
            tqdmtest.set_description(f"Testing on domain{dataid}")
            tqdmtest.set_postfix(eval_dice=np.mean(test_dice))
            tqdmtest.update()
        dataid+=1
        alldice.append(np.mean(test_dice))
    print('Average Dice:', np.mean(alldice))
    print(np.mean(alldice), alldice)
    return np.mean(alldice), alldice

def main():
    parser = argparse.ArgumentParser(description='SAM4Med')
    parser.add_argument('--input_size', type=int, default=512, help='the image size')
    parser.add_argument('--vit_name', type=str, default='vit_b', help='vit model of sam')
    parser.add_argument('--sam_ckpt', type=str, default='../SAMUS-main/SAM/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint of SAM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/prostate', help='path to dataset')
    parser.add_argument('--domain', type=int, default=0, help='domain id')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # device = torch.device('cuda:' + args.gpu)

    trainset = Prostate(base_dir=args.data_path, src_domain=args.domain, split='train', domain_idx=args.domain)
    trainLoader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,drop_last=True)

    testsets = []
    testLoaders = []
    for i in range(6):
        if i!= args.domain:
            testsets.append(Prostate(base_dir=args.data_path, src_domain=args.domain, split='train', domain_idx=i))
            testLoaders.append(DataLoader(testsets[-1], batch_size=4, shuffle=False, num_workers=4, pin_memory=True,drop_last = True))

    sam_model = sam_model_registry[args.vit_name](checkpoint=args.sam_ckpt).cuda()

    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    sam_trans = ResizeLongestSide(args.input_size)

    optimizer = optim.Adam(sam_model.parameters(), lr=args.lr, weight_decay=0.001)

    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.01, verbose=True)
    writer = SummaryWriter()
    sam_model.train()
    # test(args, testLoaders, sam_model)
    maxdice, maxalldice = 0 , None
    max_epoch = 0
    for epoch in range(args.epoch):
        epoch_loss = []
        epoch_dice = []
        tqdmbar = tqdm(trainLoader)
        for iteration, (image, mask, bbox, bbox_gt) in enumerate(tqdmbar):
            box = sam_trans.apply_boxes(bbox, (1024, 1024))
            box_tensor = torch.as_tensor(box, dtype=torch.float).cuda()
            image = image.cuda()
            mask = mask.cuda()
            
            # pdb.set_trace()
            with torch.no_grad():
                image_embeddings = sam_model.image_encoder(image)  # (B,256,64,64)

                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_tensor,
                    masks=None,
                )
                
            low_res_predictions, _ = sam_model.mask_decoder(
                image_embeddings=image_embeddings,  # (B, 256, 64, 64)
                image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
                multimask_output=False,
            )
            mask_predictions= sam_model.postprocess_masks(
                low_res_predictions,
                input_size=[1024,1024],
                original_size=[1024,1024],
            )
            
            predicted_masks = mask_predictions
            loss_seg = seg_loss(predicted_masks, mask)
            predicted_masks = torch.sigmoid(predicted_masks) > 0.5
            dice = dice_score(predicted_masks, mask)
            
            loss = loss_seg

            epoch_loss.append(loss.detach().item())
            epoch_dice.append(dice.detach().item())


            # start optimizing the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tqdmbar.set_description(f"Epoch:{epoch + 1}/{args.epoch}")
            tqdmbar.set_postfix(loss=np.mean(epoch_loss), dice=np.mean(epoch_dice), seg=loss_seg.detach().item())
            tqdmbar.update()

        if (epoch + 1) % 5 == 0:
            val_dice, alldice = test(args, testLoaders, sam_model)
            if val_dice > maxdice:
                maxdice = val_dice
                maxalldice = alldice
                max_epoch = epoch + 1
            writer.add_scalars("loss",{"train": round(np.mean(epoch_loss), 4),},epoch,)
            writer.add_scalars("dice",{"train": round(np.mean(epoch_dice), 4),"val": round(val_dice, 4),},epoch,)

    print('Final result. Best Dice:', maxdice, maxalldice)
    with open(f'resize_results.txt','a') as f:
        f.write(f'trained on domain {args.domain}\n')
        f.write(f'max dice {str(maxdice)}, epoch {max_epoch}\n')
        f.write(str(maxalldice) + '\n\n')

if __name__ == '__main__':
    main()