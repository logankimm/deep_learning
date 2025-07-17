import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import Detector, save_model
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data

import time

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 30,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    seg_weight = 1,
    depth_weight = 4,
    num_workers = 8,
    **kwargs,
):
    device = torch.device("cuda")
    start = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    train_data = load_data("drive_data/train", shuffle=True, batch_size = batch_size, num_workers = num_workers)
    val_data = load_data("drive_data/val", batch_size = batch_size, num_workers = num_workers)

    # Initialize model, losses, optimizer
    model = Detector().to(device)
    # class_weights = torch.tensor([0.3, 1.0, 1.0]).to(device) 
    # seg_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    seg_criterion = torch.nn.CrossEntropyLoss()

    # mean square error loss
    depth_criterion = torch.nn.MSELoss()
    # depth_criterion = torch.nn.L1Loss() # Mean Absolute Error (MAE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch)
    metric = DetectionMetric()

    # print("testing the save_model function")
    # best_iou = 0.0

    for epoch in range(num_epoch):
        # train_start = time.time()
        metric.reset()
        # Training loop
        # print(epoch, "training", time.time() - start)
        model.train()
        for batch in train_data:
            images = batch['image'].to(device)
            seg_labels = batch['track'].to(device)
            depth_labels = batch['depth'].to(device)
 
            seg_logits, depth_preds = model(images)
            
            loss_seg = seg_criterion(seg_logits, seg_labels)
            loss_depth = depth_criterion(depth_preds, depth_labels)
            loss = seg_weight * loss_seg + depth_weight * loss_depth
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        # break

        # Validation loop
        # print(epoch, "validation", time.time() - start)
        with torch.no_grad():
            model.eval()
            for batch in val_data:
                images = batch['image'].to(device)
                seg_labels = batch['track'].to(device)
                depth_labels = batch['depth'].to(device)
                
                seg_preds, depth_preds_norm = model.predict(images)
                metric.add(seg_preds, seg_labels, depth_preds_norm, depth_labels)

        val_metrics = metric.compute()
        # if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"IoU={val_metrics['iou']:.4f} "
            f"DepthErr={val_metrics['abs_depth_error']:.4f} "
            f"TPDepthErr={val_metrics['tp_depth_error']:.4f} "
            f"{time.time() - start}"
        )

    save_model(model)
    print("model saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))