from typing import Tuple, List, Callable, Optional

from PIL import Image
import pandas as pd
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
            image = self.transform(image)
            images = torch.cat((images, image[None]), dim=0)
            bbox = torch.cat((bbox, torch.tensor([[xmin, ymin, xmax, ymax]])), dim=0)
        return images, bbox

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        label = torch.tensor(1) if self.bbox[index].sum() > 0 else torch.tensor(0)
        label = label.long()

        boxes = self.bbox[index].int() if label == 1 else torch.Tensor([0, 1, 2, 3])
        return self.images[index], boxes, label


    def __len__(self):
        return self.images.size(0)


def get_dataloader(batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(WallyDataset(), batch_size=batch_size, shuffle=shuffle)
