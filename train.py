import os

import numpy as np
import torch
from torch.optim import Adam
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from src import get_dataloader
from src.utils import draw_bbox
from src.network import CNN


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Dataset and DataLoader")
    dataloader = get_dataloader(batch_size=2)

    print("Building models")
    model = CNN()
    model.to(device)
    model.train()

    print("Optimizer")
    optimizer = Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-3,
        weight_decay=5e-4
    )
    criterion = torch.nn.MSELoss()

    board = Tensorboard("Test", "./runs", delete=True)

    pbar = tqdm(range(100))
    for epoch in pbar:
        epoch_losses = []
        
        for image, bbox in dataloader:
            image = image.to(device)
            
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, bbox.to(device))
            
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix_str(f"Loss: {np.mean(epoch_losses)}", refresh=True)
        
        board.add_scalars(
            train=True,
            prior="Train",
            Loss=np.mean(epoch_losses)
        )

        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        torch.save(model.state_dict(), f"./checkpoints/{epoch}.ckpt")
