import argparse
import json
import os
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm

from src.utils import expand2square, process_images
from src.model.builder import load_pretrained_model


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
            iqadata = json.load(f)

        image_tensors = []
        batch_data = []
        imgs_handled = []

        save_path = os.path.join(save_dir, os.path.basename(meta_path))
        if os.path.exists(save_path):
            with open(save_path) as fr:
                for line in fr:
                    meta_res = json.loads(line)
                    imgs_handled.append(meta_res["image"])

        meta_name = os.path.basename(meta_path)
        for i, llddata in enumerate(tqdm(iqadata, desc=f"Evaluating [{meta_name}]")):
            filename = llddata.get("image") or llddata.get("img_path")
            if filename in imgs_handled:
                continue

            llddata["logits"] = {}
            if with_prob:
                llddata["probs"] = {}

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
                    for k, tok in enumerate(level_names):
                        xllddata["logits"][tok] = logits[j, k].item()
                        if with_prob:
                            xllddata["probs"][tok] = probs[j, k].item()
                    meta_res = {
                        "id": xllddata.get("id"),
                        "image": xllddata.get("image") or xllddata.get("img_path"),
                        "gt_score": xllddata.get("gt_score"),
                        "logits": xllddata["logits"],
                    }
                    if with_prob:
                        meta_res["probs"] = xllddata["probs"]
                    with open(save_path, "a") as fw:
                        fw.write(json.dumps(meta_res) + "\n")

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
