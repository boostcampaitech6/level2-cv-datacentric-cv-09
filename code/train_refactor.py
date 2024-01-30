import os
import time
import math
from tqdm import tqdm
from datetime import timedelta
from argparse import ArgumentParser
from omegaconf import OmegaConf
import wandb

import torch
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler

from model import EAST
from utils import (
    seed_everything, get_lr, set_wandb, set_train_data
)


def train(
    train_loader, lr, max_epoch, save_interval,
    model_dir, batch_size, experiment_name,
):
    scaler = GradScaler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[max_epoch // 2], gamma=0.1
    )

    pbar = tqdm(range(max_epoch), desc='Epoch', position=0, leave=False)
    for epoch in pbar:
        model.train()
        train_loss, train_start = 0, time.time()
        train_cls_loss, train_angle_loss, train_IoU_loss = 0, 0, 0
        train_pbar = tqdm(
            train_loader, desc='training', position=1, leave=False
        )
        for img, gt_score_map, gt_geo_map, roi_mask in train_pbar:
            loss, extra_info = model.train_step(
                img, gt_score_map, gt_geo_map, roi_mask
            )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.item()
            train_loss += loss_value
            train_cls_loss += extra_info['cls_loss']
            train_angle_loss += extra_info['angle_loss']
            train_IoU_loss += extra_info['iou_loss']
            train_pbar.set_postfix({
                'train_loss': loss_value,
                'train_cls_loss': extra_info['cls_loss'],
                'train_angle_loss': extra_info['angle_loss'],
                'train_IoU_loss': extra_info['iou_loss']
            })

        train_loss = train_loss / \
            math.ceil(len(train_loader.dataset) / batch_size)
        train_cls_loss = train_cls_loss / \
            math.ceil(len(train_loader.dataset) / batch_size)
        train_angle_loss = train_angle_loss / \
            math.ceil(len(train_loader.dataset) / batch_size)
        train_IoU_loss = train_IoU_loss / \
            math.ceil(len(train_loader.dataset) / batch_size)
        current_lr = get_lr(optimizer)

        train_end = time.time()
        print(
            f"Train Epoch[{epoch+1}/{max_epoch}] "
            f"| lr {current_lr} | train loss {train_loss:4.4}"
            f" | train_cls_loss {train_cls_loss:4.4} "
            f"| train_angle_loss {train_angle_loss:4.4} "
            f"| train_IoU_loss {train_IoU_loss:4.4} | train time: "
            f"{str(timedelta(seconds=train_end - train_start)).split('.')[0]}"
        )

        wandb.log({
                "train_loss": train_loss,
                "train_cls_loss": train_cls_loss,
                "train_angle_loss": train_angle_loss,
                "train_IoU_loss": train_IoU_loss,
                'train_rgb': wandb.Image(img[0])
            }, step=epoch+1)

        scheduler.step()

        save_dir = os.path.join(model_dir, experiment_name)
        if (epoch+1) % save_interval == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            ckpt_fpath = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(cfg):
    seed_everything(cfg.seed)
    set_wandb(exp_name=cfg.experiment_name, configs=cfg)
    train_loader = set_train_data(
        data_dir=cfg.data_dir, image_size=cfg.image_size,
        input_size=cfg.input_size, ignore_tags=cfg.ignore_tags,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train(
        train_loader, lr=cfg.learning_rate,
        max_epoch=cfg.max_epoch, save_interval=cfg.save_interval,
        model_dir=cfg.model_dir, batch_size=cfg.batch_size,
        experiment_name=cfg.experiment_name
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/train.yaml"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)
    main(cfg)
