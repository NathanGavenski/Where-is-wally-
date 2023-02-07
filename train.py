import os

import numpy as np
import torch
from torch.optim import Adam
from tensorboard_wrapper.tensorboard import Tensorboard
from tqdm import tqdm

from src import get_dataloader, get_faster_rcnn
from src.utils import draw_bbox


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Dataset and DataLoader")
    dataloader = get_dataloader(batch_size=1)

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
        for image, bboxes, labels in dataloader:
            image = image.to(device)
            targets = [
                {
                    "boxes": bbox[None], 
                    "labels": torch.Tensor([label]).long()
                } 
                for bbox, label 
                in zip(bboxes, labels)
            ]
            
            output = model(image, targets)
            losses = sum(loss for loss in output.values())
            epoch_losses.append(losses.item())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            pbar.set_postfix_str(s=f"Loss: {np.mean(epoch_losses)}", refresh=True)
        
            if epoch % 5 == 0 and epoch > 0:
                with torch.no_grad():
                    model.eval()
                    output = model(image[0][None])
                    if output[0]["scores"].size(0) > 0:
                        print(output)
                        exit()
                model.train()
                pass

        board.add_scalar("Losses", np.mean(epoch_losses), epoch="train")
        board.step()
    
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        torch.save(model.state_dict(), f"./checkpoints/{epoch}.ckpt")
