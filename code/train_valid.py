import os
import time
import math
import json
from tqdm import tqdm
from datetime import timedelta
from argparse import ArgumentParser
from omegaconf import OmegaConf
import wandb
from PIL import Image

import torch
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler

from model import EAST
from utils import (
    seed_everything, get_lr, set_wandb, convert_map_bbox, evaluate, set_data
)


def train(
    train_loader, val_loader, eval_loader, lr, max_epoch, save_interval,
    model_dir, batch_size, split, data_dir, experiment_name,
    root_folder, image_size
):
    epoch_train_loss, epoch_valid_loss, f1 = 0, 0, 0
    best_loss = 10

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
        epoch_train_loss = train_loss
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

        if (epoch+1) % save_interval == 0:
            model.eval()
            val_loss, val_start = 0, time.time()
            val_cls_loss, val_angle_loss, val_IoU_loss = 0, 0, 0
            image_sizes, predict_box = [], []

            # Validate
            val_pbar = tqdm(
                val_loader, desc='validating', position=1, leave=False
            )
            for img, gt_score_map, gt_geo_map, roi_mask in val_pbar:
                with torch.no_grad():
                    loss, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )
                    loss_value = loss.item()
                    val_loss += loss_value
                    val_cls_loss += extra_info['cls_loss']
                    val_angle_loss += extra_info['angle_loss']
                    val_IoU_loss += extra_info['iou_loss']
                    val_pbar.set_postfix({
                        'val_loss': loss_value,
                        'val_cls_loss': extra_info['cls_loss'],
                        'val_angle_loss': extra_info['angle_loss'],
                        'val_IoU_loss': extra_info['iou_loss']
                    })

            val_loss = val_loss / \
                math.ceil(len(val_loader.dataset) / (batch_size // 2))
            val_cls_loss = val_cls_loss / \
                math.ceil(len(val_loader.dataset) / (batch_size // 2))
            val_angle_loss = val_angle_loss / \
                math.ceil(len(val_loader.dataset) / (batch_size // 2))
            val_IoU_loss = val_IoU_loss / \
                math.ceil(len(val_loader.dataset) / (batch_size // 2))
            epoch_valid_loss = val_loss

            val_end = time.time()
            print(
                f"Val Epoch[{epoch+1}/{max_epoch}] "
                f"| val loss {val_loss:4.4} "
                f"| val_cls_loss {val_cls_loss:4.4} "
                f"| val_angle_loss {val_angle_loss:4.4} "
                f"| val_IoU_loss {val_IoU_loss:4.4} | eval time: "
                f"{str(timedelta(seconds=val_end - val_start)).split('.')[0]}"
            )

            wandb.log({
                    "val_loss": val_loss,
                    "val_cls_loss": val_cls_loss,
                    "val_angle_loss": val_angle_loss,
                    "val_IoU_loss": val_IoU_loss,
                    'val_rgb': wandb.Image(img[0])
                }, step=epoch+1)

            # Evaluate
            eval_start = time.time()
            file_path = os.path.join(
                root_folder, 'ufo/val_{}.json'.format(split)
            )
            with open(file_path, 'r') as f:
                anno = json.load(f)
            file_paths = sorted(anno['images'].keys())
            root_path = os.path.join(root_folder, 'img', 'train')
            for i in range(len(file_paths)):
                path = os.path.join(root_path, file_paths[i])
                image_sizes.append(Image.open(path).size)

            eval_pbar = tqdm(
                eval_loader, desc='evaluating', position=1, leave=False
            )
            for idx, (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(eval_pbar):
                with torch.no_grad():
                    _, extra_info = model.train_step(
                        img, gt_score_map, gt_geo_map, roi_mask
                    )
                    val_score_map = extra_info['score_map']
                    val_geo_map = extra_info['geo_map']
                    score_maps = val_score_map.cpu().numpy()
                    geo_maps = val_geo_map.cpu().numpy()

                    batch = batch_size // 2
                    img_sizes = image_sizes[idx*batch: idx*batch+batch]
                    predict_box.extend(convert_map_bbox(
                        score_maps, geo_maps, img_sizes, image_size))

            metric = evaluate(data_dir, split, predict_box)
            precision = metric['total']['precision']
            recall = metric['total']['recall']
            hmean = metric['total']['hmean']
            f1 = hmean

            eval_end = time.time()
            print(
                f"Eval time: "
                f"{str(timedelta(seconds=eval_end-eval_start)).split('.')[0]}"
                f' | precision : {precision:.4f} '
                f'| recall: {recall:.4f} | hmean: {hmean:.4f}'
            )

            wandb.log({
                'Precision': precision,
                'Recall': recall,
                'Hmean': hmean
            }, step=epoch+1)

        pbar.set_postfix({
            'train_loss': epoch_train_loss,
            'valid_loss': epoch_valid_loss,
            'f1-score': f1
        })

        if (epoch+1) % save_interval == 0:
            save_dir = os.path.join(model_dir, experiment_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            ckpt_fpath = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)

        if epoch_valid_loss < best_loss and epoch_valid_loss != 0:
            best_loss = epoch_valid_loss
            torch.save(
                model.state_dict(), os.path.join(save_dir, 'latest.pth')
            )


def main(cfg):
    seed_everything(cfg.seed)
    set_wandb(exp_name=cfg.experiment_name, configs=cfg)
    train_loader, val_loader, eval_loader = set_data(
        data_dir=cfg.data_dir, image_size=cfg.image_size,
        input_size=cfg.input_size, ignore_tags=cfg.ignore_tags,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        split=cfg.split)
    train(
        train_loader, val_loader, eval_loader, lr=cfg.learning_rate,
        max_epoch=cfg.max_epoch, save_interval=cfg.save_interval,
        model_dir=cfg.model_dir, batch_size=cfg.batch_size,
        split=cfg.split, data_dir=cfg.data_dir,
        experiment_name=cfg.experiment_name, root_folder=cfg.data_dir,
        image_size=cfg.image_size
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
