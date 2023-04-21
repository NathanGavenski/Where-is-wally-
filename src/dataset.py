from typing import Tuple, List, Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WallyDataset(Dataset):
    def __init__(
        self, 
        size: Optional[List[int]] = None, 
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        train: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.size = size if size is not None else [512, 512]
        dataset = pd.read_csv("./data/solutions.csv", sep=";")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=self.size)
            ])
        self.images, self.bbox = self.__process_data(dataset)
        self.train = train
        self.dataset = dataset
        
    def __process_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        "Process DataFrame into x (images) and y (classes)"
        
        images = torch.Tensor(size=(0, 3, *self.size))
        bbox = torch.Tensor(size=(0, 4))
        for _, (idx, xmin, ymin, xmax, ymax, _) in data.iterrows():
            image = Image.open(f"./data/{int(idx)}.jpg").convert("RGB")

            tensor = self.transform(image)
            images = torch.cat((images, tensor[None]), dim=0)

            y_original, x_original, _ = np.array(image).shape
            _, x_new, y_new = tensor.shape

            xmin = xmin/(x_original/x_new)
            xmax = xmax/(x_original/x_new)
            ymin = ymin/(y_original/y_new)
            ymax = ymax/(y_original/y_new)

            bbox = torch.cat((bbox, torch.tensor([[xmin, ymin, xmax, ymax]])), dim=0)
        return images, bbox
    
    def unprocess_data(
            self, 
            index: int, 
            bbox: torch.Tensor, 
            source_shape: Tuple[int, int, int]
        ) -> Tuple[Image.Image, torch.Tensor]:
        "Undo what __process_data did for a single image"

        image = Image.open(f"./data/{index}.jpg").convert("RGB")
        y_original, x_original, _ = np.array(image).shape
        _, x_new, y_new = source_shape
        xmin, ymin, xmax, ymax = bbox

        xmin = (xmin / (x_new / x_original)).item()
        xmax = (xmax / (x_new / x_original)).item()
        ymin = (ymin / (y_new / y_original)).item()
        ymax = (ymax / (y_new / y_original)).item()

        bbox = [xmin, ymin, xmax, ymax]
        return image, bbox

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.train:
            return self.images[index], self.bbox[index].float()
        else:
            return self.images[index], self.bbox[index].float(), self.dataset.loc[index, "idx"]

    def __len__(self):
        return self.images.size(0)


def get_dataloader(batch_size: int = 1, shuffle: bool = True, collate_fn: Callable = None, train=True) -> DataLoader:
    return DataLoader(WallyDataset(train=train), batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
