import os
import json
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
import wandb

from dataset import SceneTextDatasetV2, SceneTextDatasetV3
from east_dataset import EASTDataset
from detect import get_bboxes
from deteval import calc_deteval_metrics


_Optimizer = torch.optim.Optimizer


def seed_everything(seed: int) -> None:
    """
    시드 고정 method

    :param seed: 시드
    :type seed: int
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer: _Optimizer) -> float:
    """
    optimizer 통해 lr 얻는 method

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :return: learning_rate
    :rtype: float
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def set_wandb(exp_name: str, configs) -> None:
    wandb.init(
        name=exp_name,
        project="ocr",
        config={
                'seed': configs['seed'],
                'split': configs['split'],
                'image_size': configs['image_size'],
                'input_size': configs['input_size'],
                'batch_size': configs['batch_size'],
                'learning_rate': configs['learning_rate'],
                'epoch': configs['max_epoch']
            }
    )


def set_data(
    data_dir, image_size, input_size, ignore_tags,
    batch_size, num_workers, split
):
    train_dataset = SceneTextDatasetV2(
        data_dir,
        split=f'train_{str(split)}',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        val_aug=False,
        color_jitter=True,
        normalize=True,
        blur=False,
        noise=False
    )
    val_dataset = SceneTextDatasetV2(
        data_dir,
        split=f'val_{str(split)}',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        val_aug=False,
        color_jitter=True,
        normalize=True,
        blur=False,
        noise=False
    )
    eval_dataset = SceneTextDatasetV2(
        data_dir,
        split=f'val_{str(split)}',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        val_aug=True,
        color_jitter=True,
        normalize=True,
        blur=False,
        noise=False
    )
    train_dataset = EASTDataset(train_dataset)
    val_dataset = EASTDataset(val_dataset)
    eval_dataset = EASTDataset(eval_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=num_workers
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size // 2,
        shuffle=False,
        num_workers=num_workers
    )
    return train_loader, val_loader, eval_loader


def get_gt_bboxes(images):
    gt_box, transcription = {}, {}
    for path in images:
        gt_box[path], transcription[path] = [], []
        for idx in images[path]['words'].keys():
            gt_box[path].append(images[path]['words'][idx]['points'])
            transcription[path].append(
                ["1"] * len(images[path]['words'][idx]['points'])
            )
    return gt_box, transcription


def convert_map_bbox(
    score_maps, geo_maps, orig_sizes, map_scale=0.5, input_size=1024
):
    by_sample_bboxes = []
    for score_map, geo_map, orig_size in zip(score_maps, geo_maps, orig_sizes):
        map_margin = int(abs(orig_size[0] - orig_size[1]) * map_scale * input_size / max(orig_size))
        if orig_size[0] == orig_size[1]:
            score_map, geo_map = score_map, geo_map
        elif orig_size[0] > orig_size[1]:
            score_map, geo_map = score_map[:, :, :-map_margin], geo_map[:, :, :-map_margin]
        else:
            score_map, geo_map = score_map[:, :-map_margin, :], geo_map[:, :-map_margin, :]

        bboxes = get_bboxes(score_map, geo_map)
        if bboxes is None:
            bboxes = np.zeros((0, 4, 2), dtype=np.float32)
        else:
            bboxes = bboxes[:, :8].reshape(-1, 4, 2)
            bboxes *= max(orig_size) / input_size
        by_sample_bboxes.append(bboxes)

    return by_sample_bboxes


def evaluate(data_dir, split, predict_box):
    json_path = os.path.join(data_dir, f'ufo/val_{str(split)}.json')
    with open(json_path) as f:
        file = json.load(f)
    image_paths = sorted(list(file['images'].keys()))
    pred_bboxes = dict()
    for idx in range(len(image_paths)):
        image_fname = image_paths[idx]
        sample_bboxes = predict_box[idx]
        pred_bboxes[image_fname] = sample_bboxes
    gt_box, transcription = get_gt_bboxes(file['images'])

    metric = calc_deteval_metrics(
        pred_bboxes, gt_box, transcription
    )
    return metric


def set_train_data(
    data_dir, image_size, input_size, ignore_tags,
    batch_size, num_workers
):
    train_dataset = SceneTextDatasetV3(
        data_dir,
        split='train',
        image_size=image_size,
        crop_size=input_size,
        ignore_tags=ignore_tags,
        color_jitter=True,
        normalize=True,
        blur=False,
        noise=True
    )
    train_dataset = EASTDataset(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader
