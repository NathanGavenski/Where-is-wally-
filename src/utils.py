from PIL import ImageDraw
import torch
from torchvision import transforms

def draw_bbox(image: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
    image = transforms.ToPILImage()(image)
    xmin, ymin, xmax, ymax = bbox

    draw = ImageDraw.Draw(image)
    draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 255, 255))
    return transforms.ToTensor()(image)
