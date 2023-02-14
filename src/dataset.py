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
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        super().__init__()
        self.size = size if size is not None else [512, 512]
        dataset = pd.read_csv("./data/solutions.csv", sep=";")

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.PILToTensor(),
                transforms.Resize(size=self.size)
            ])
        self.images, self.bbox = self.__process_data(dataset)
        
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

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.images[index], self.bbox[index].long(), torch.tensor(0).long()

    def __len__(self):
        return self.images.size(0)


def get_dataloader(batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(WallyDataset(), batch_size=batch_size, shuffle=shuffle)
