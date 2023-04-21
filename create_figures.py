import os

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from tqdm import tqdm
from PIL import Image

from src import get_dataloader
from src.utils import draw_bbox
from src.network import CNN


if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Dataset and DataLoader")
    dataloader = get_dataloader(batch_size=1, train=False, shuffle=False)

    print("Building models")
    model = CNN()
    model.to(device)

    for i in tqdm(range(100)):
        model.load_state_dict(torch.load(f"./checkpoints/checkpoints/{i}.ckpt", map_location=torch.device('cpu')))
        model.eval()

        cam = EigenCAM(
            model=model, 
            target_layers=[model.feature_extractor[12]], 
            use_cuda=torch.cuda.is_available()
        )

        if not os.path.exists(f"./outputs/{i}/"):
            os.makedirs(f"./outputs/{i}/")
        
        for (images, target, index) in dataloader:
            images = images.to(device)
            outputs = model(images)
            image, bbox = dataloader.dataset.unprocess_data(
                index[0],
                outputs[0],
                images[0].shape
            )
            try:
                draw_bbox(image, bbox).save(f"./outputs/{i}/{index.item()}_bbox.png", "png")
                grayscale_cam = cam(input_tensor=images, targets=target)
                grayscale_cam = grayscale_cam[0, :]
                image = images[0].permute(1, 2, 0).numpy()
                visualization = show_cam_on_image(image, grayscale_cam, use_rgb=True)
                Image.fromarray(visualization).save(f"./outputs/{i}/{index.item()}_heatmap.png")
            except ValueError:
                continue
