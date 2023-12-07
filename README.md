# Leveraging SAM for Single-Source Domain Generalization in Medical Image Segmentation

## Abstract

Domain Generalization (DG) aims to reduce domain shifts between domains to achieve promising performance on the unseen target domain, which has been widely practiced in medical image segmentation. Single-source domain generalization (SDG) is the most challenging setting that trains on only one source domain. Although existing methods have made considerable progress on SDG of medical image segmentation, the performances are still far from the applicable standards when faced with a relatively large domain shift. To address this problem, in this paper, we leverage the Segment Anything Model (SAM) to SDG to greatly improve the ability of generalization. Specifically, we introduce a parallel framework, the source images are sent into the SAM module and normal segmentation module respectively. To reduce the calculation resources, we apply a merging strategy before sending images to the SAM module. We extract the bounding boxes from the segmentation module and send the refined version as prompts to the SAM module. We evaluate our model on a classic DG dataset and achieve competitive results compared to other state-of-the-art DG methods. Furthermore, We conducted a series of ablation experiments to prove the effectiveness of the proposed method.

## Dataset

Download the pre-processed [**Prostate**](https://drive.google.com/file/d/1-SCjNklFEAq7MlBwcw2ZNR179JqlOubL/view) dataset provided by [SAML](https://github.com/liuquande/SAML) and put it in the `data` folder for your data.

## Pretrained Model

We use the checkpoint of SAM in [vit_b](https://github.com/facebookresearch/segment-anything) version. Don't forget to follow the installation part of SAM.

## Training process

#### step 1

> Generate the coarse prediction masks from the segmentation network (Resnet).

```bash
python resnet_prostate.py --domain domain_id --batch_size batchsize --gpu gpuid --epoch epochs
```

#### step 2

>Formulated the refined bounding boxes.
```bash
python save_resnet_bbox.py --domain domain_id --thresh threshold
```

#### step 3

>Generate the final prediction masks results by fine-tuning SAM with the refined bounding boxes.

```bash
python sam_4_preprocessed_bbox.py --domain domain_id --batch_size batchsize --gpu gpuid --epoch epochs
```

