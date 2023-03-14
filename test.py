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

    for i in range(100):
        model.load_state_dict(torch.load(f"./checkpoints/checkpoints/{i}.ckpt", map_location=torch.device('cpu')))
        model.eval()
        
        for (images, target) in dataloader:
            images = list(image.to(device) for image in images)
            outputs = model(images)
            outputs = [{k: v for k, v in t.items()} for t in outputs]
            print(i, target, outputs)
            break
