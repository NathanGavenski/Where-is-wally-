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
    device = "cpu"

    print("Loading Dataset and DataLoader")
    dataloader = get_dataloader(batch_size=1, shuffle=False)

    model = get_faster_rcnn()
    model.load_state_dict(torch.load("./checkpoints/40.ckpt"))
    model.to(device)
    model.eval()

    for image, bboxes, labels in dataloader:
        image = image.to(device)
        print(image.shape, bboxes, labels)

        output = model(image)
        print(output)
        # exit()