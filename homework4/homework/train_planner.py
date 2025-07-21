import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from homework.models import MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from homework.metrics import PlannerMetric
from homework.datasets.road_dataset import load_data

def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 40,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers = 0,
    **kwargs
):
    device = torch.device("cuda")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use 'state_only' for MLP and Transformer planners, 'default' for CNN planner
    transform_pipeline = "default" if model_name == "cnn_planner" else "state_only"
    
    train_loader = load_data(
        dataset_path= "drive_data/train",
        transform_pipeline = transform_pipeline,
        num_workers = num_workers,
        batch_size = batch_size,
        shuffle = True,
    )
    val_loader = load_data(
        dataset_path = f"drive_data/val",
        transform_pipeline = transform_pipeline,
        num_workers = num_workers,
        batch_size = batch_size,
    )

    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    metric = PlannerMetric()
    
    best_val_error = float('inf')

    # Training Loop
    for epoch in range(num_epoch):
        # for test in train_loader:
        #     print(test)
        #     return

        # model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epoch} [Train]", disable = True)
        
        for batch in pbar:
            # print(batch["waypoints_mask"])
            # return
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            predictions = model(batch["track_left"], batch["track_right"])
            
            # Calculate masked loss
            loss = criterion(predictions, batch["waypoints"])
            masked_loss = loss * batch["waypoints_mask"].unsqueeze(-1)
            # Normalize by the number of valid waypoints to get mean loss
            final_loss = masked_loss.sum() / batch["waypoints_mask"].sum()

            # Backward pass and optimization
            final_loss.backward()
            optimizer.step()
            
            train_loss += final_loss.item()
            pbar.set_postfix(loss=f"{final_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        metric.reset()
        val_loss = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Val]", disable = True)
        
        with torch.no_grad():
            for batch in pbar_val:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                predictions = model(**batch)
                
                # Calculate masked loss for logging
                loss = criterion(predictions, batch["waypoints"])
                masked_loss = loss * batch["waypoints_mask"].unsqueeze(-1)
                final_loss = masked_loss.sum() / batch["waypoints_mask"].sum()
                val_loss += final_loss.item()
                
                # Update metrics
                metric.add(predictions.cpu(), batch["waypoints"].cpu(), batch["waypoints_mask"].cpu())

        avg_val_loss = val_loss / len(val_loader)
        val_metrics = metric.compute()
        
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Val Loss: {avg_val_loss:.4f} "
            f"L1 Error: {val_metrics['l1_error']:.4f} "
            f"Longitudinal: {val_metrics['longitudinal_error']:.4f} "
            f"Lateral: {val_metrics['lateral_error']:.4f}"
            # f"{time.time() - start}"
        )

        # Save the best model
        if val_metrics['l1_error'] < best_val_error:
            best_val_error = val_metrics['l1_error']
            save_model(model)
            print("saved model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="mlp_planner")
    # parser.add_argument("--num_epoch", type=int)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(**vars(parser.parse_args()))