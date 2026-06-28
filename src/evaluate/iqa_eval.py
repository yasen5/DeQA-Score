import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Dict, Optional

import requests
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import expand2square, process_images
from src.model.builder import load_pretrained_model


@dataclass
class IQAMetaSample:
    id: Optional[str]
    image: str
    gt_score: Optional[float] = None

    @classmethod
    def from_json_dict(cls, data):
        image = data.get("image") or data.get("img_path")
        if image is None:
            raise ValueError("IQA metadata sample must include 'image' or 'img_path'")
        return cls(
            id=data.get("id"),
            image=image,
            gt_score=data.get("gt_score"),
        )


@dataclass
class IQAPredictionResult:
    id: Optional[str]
    image: str
    gt_score: Optional[float]
    logits: Dict[str, float] = field(default_factory=dict)
    probs: Optional[Dict[str, float]] = None

    @classmethod
    def from_json_dict(cls, data):
        return cls(**data)

    def to_json_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def main(args):
    model, image_processor = load_pretrained_model(
        args.model_path,
        device=args.device,
        preprocessor_path=args.preprocessor_path,
    )

    level_names = args.level_names  # ["excellent", "good", "fair", "poor", "bad"]

    meta_paths = args.meta_paths
    root_dir = args.root_dir
    batch_size = args.batch_size
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with_prob = args.with_prob

    for meta_path in meta_paths:
        with open(meta_path) as f:
            iqadata = [IQAMetaSample.from_json_dict(sample) for sample in json.load(f)]

        image_tensors = []
        batch_data = []
        imgs_handled = []

        save_path = os.path.join(save_dir, os.path.basename(meta_path))
        if os.path.exists(save_path):
            with open(save_path) as fr:
                for line in fr:
                    meta_res = IQAPredictionResult.from_json_dict(json.loads(line))
                    imgs_handled.append(meta_res.image)

        meta_name = os.path.basename(meta_path)
        for i, llddata in enumerate(tqdm(iqadata, desc=f"Evaluating [{meta_name}]")):
            filename = llddata.image
            if filename in imgs_handled:
                continue

            image = load_image(os.path.join(root_dir, filename))
            image = expand2square(image, tuple(int(x * 255) for x in image_processor.image_mean))
            image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().to(args.device)
            image_tensors.append(image_tensor)
            batch_data.append(llddata)

            if (i + 1) % batch_size == 0 or i == len(iqadata) - 1:
                with torch.inference_mode():
                    images = torch.cat(image_tensors, 0)           # (B, C, H, W)
                    output = model(images=images)                   # ViTIQAOutput
                    logits = output.logits                          # (B, 5)
                    if with_prob:
                        probs = torch.softmax(logits, dim=-1)

                for j, xllddata in enumerate(batch_data):
                    meta_res = IQAPredictionResult(
                        id=xllddata.id,
                        image=xllddata.image,
                        gt_score=xllddata.gt_score,
                        probs={} if with_prob else None,
                    )
                    for k, tok in enumerate(level_names):
                        meta_res.logits[tok] = logits[j, k].item()
                        if with_prob:
                            meta_res.probs[tok] = probs[j, k].item()
                    with open(save_path, "a") as fw:
                        fw.write(json.dumps(meta_res.to_json_dict()) + "\n")

                image_tensors = []
                batch_data = []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--preprocessor-path", type=str, default=None)
    parser.add_argument("--meta-paths", type=str, required=True, nargs="+")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="results")
    parser.add_argument("--level-names", type=str, required=True, nargs="+")
    parser.add_argument("--with-prob", type=bool, default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.device is None:
        import torch
        if torch.cuda.is_available():
            args.device = "cuda:0"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    main(args)
