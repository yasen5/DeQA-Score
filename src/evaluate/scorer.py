from PIL import Image
from typing import List

import torch
import torch.nn as nn

from src.utils import expand2square
from src.model.builder import load_pretrained_model


class Scorer(nn.Module):
    def __init__(self, pretrained="zhiyuanyou/DeQA-Score-Mix3", device="cuda:0"):
        super().__init__()
        model, image_processor = load_pretrained_model(pretrained, device=device)
        self.model = model
        self.image_processor = image_processor
        self.device = device

    def forward(self, images: List[Image.Image]) -> torch.Tensor:
        images = [
            expand2square(img, tuple(int(x * 255) for x in self.image_processor.image_mean))
            for img in images
        ]
        with torch.inference_mode():
            image_tensor = (
                self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
                .half()
                .to(self.device)
            )
            return self.model.score(image_tensor)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="zhiyuanyou/DeQA-Score-Mix3")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--img_path", type=str, default="fig/singapore_flyer.jpg")
    args = parser.parse_args()

    scorer = Scorer(pretrained=args.model_path, device=args.device)
    print(scorer([Image.open(args.img_path)]).tolist())
