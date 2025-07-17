import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from homework.models import Classifier, save_model
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data

import time

def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 1,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 2024,
    num_workers = 8,
    **kwargs,
):
    device = torch.device("cuda")
    # start = time.time()

    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     print("Using CUDA")

    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     device = torch.device("mps")
    # else:
    #     print("CUDA not available, using CPU")
    #     device = torch.device("cpu")
    
    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load data
    # print("loading training data")
    train_data = load_data("classification_data/train", shuffle = True, batch_size = batch_size, transform_pipeline = "aug", num_workers = num_workers)
    # print("loading validation/testing data")
    val_data = load_data("classification_data/val", shuffle=False, num_workers = num_workers)
    # print("done loading the data")
    
    # Model
    model = Classifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    val_metric = AccuracyMetric()
    train_metric = AccuracyMetric()
    global_step = 0
    
    # print("starting the training loop", time.time() - start)
    # Training loop
    for epoch in range(num_epoch):
        # print(epoch, time.time() - start)
        
        train_metric.reset()
        val_metric.reset()

        model.train()
        # print(epoch, "training", time.time() - start)

        for img, label in train_data:
            # lstart = time.time()
            img, label = img.to(device), label.to(device)
            outputs = model(img)

            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.add_scalar("train_loss", loss.item(), global_step)
            train_metric.add(model.predict(img), label)
            
            global_step += 1
        
        # print("validation", time.time() - start)
        # Validation
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                val_metric.add(model.predict(img), label)
                
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={train_metric.compute()['accuracy']:.4f} "
                f"val_acc={val_metric.compute()['accuracy']:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))