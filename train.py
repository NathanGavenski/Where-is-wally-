import os

import numpy as np
import torch
from torch.optim import Adam
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from src import get_dataloader, get_faster_rcnn
from src.utils import draw_bbox

from engine import train_one_epoch, evaluate
from utils import collate_fn

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Dataset and DataLoader")
    dataloader = get_dataloader(batch_size=1, collate_fn=collate_fn)

    print("Building models")
    model = get_faster_rcnn()
    model.to(device)

    print("Optimizer")
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-3,
        weight_decay=5e-4
    )

    board = Tensorboard("Test", "./runs", delete=True)

    pbar = tqdm(range(100))
    for epoch in pbar:
        epoch_losses = []
        train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq=10)
        evaluate(model, dataloader, device)

        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        torch.save(model.state_dict(), f"./checkpoints/{epoch}.ckpt")
