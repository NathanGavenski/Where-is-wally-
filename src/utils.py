from PIL import ImageDraw
from PIL.Image import Image
import torch
from torchvision import transforms

def draw_bbox(image: Image, bbox: torch.Tensor) -> Image:
    xmin, ymin, xmax, ymax = bbox

    draw = ImageDraw.Draw(image)
    draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=5)
    return image
