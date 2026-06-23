import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import expand2square, rank0_print


@dataclass
class SingleSampleItem:
    image: torch.Tensor
    task_type: str
    level_probs: List[float]


class SingleDataset(Dataset):
    """Dataset for single-image quality scoring."""

    def __init__(self, data_paths, data_weights, data_args):
        super().__init__()
        list_data_dict = []
        for data_path, data_weight in zip(data_paths, data_weights):
            data_dict = json.load(open(data_path, "r"))
            list_data_dict += data_dict * data_weight

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def next_rand(self):
        return random.randint(0, len(self) - 1)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        while True:
            try:
                sample = self.list_data_dict[i]
                image_folder = self.data_args.image_folder
                processor = self.data_args.image_processor

                image_file = sample.get("image")
                if image_file is not None:
                    image_path = os.path.join(image_folder, image_file)
                    try:
                        image = Image.open(image_path).convert("RGB")
                    except Exception as ex:
                        print(ex)
                        i = self.next_rand()
                        continue

                    if self.data_args.image_aspect_ratio == "pad":
                        image = expand2square(
                            image, tuple(int(x * 255) for x in processor.image_mean)
                        )
                    image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                else:
                    crop_size = processor.crop_size
                    image = torch.zeros(3, crop_size["height"], crop_size["width"])

                return SingleSampleItem(
                    image=image,
                    task_type=sample.get("task_type", "score"),
                    level_probs=sample.get("level_probs", [-10000.0] * 5),
                )
            except Exception as ex:
                print(ex)
                i = self.next_rand()
                continue


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate single-image samples into a batch."""

    def __call__(self, instances: Sequence[SingleSampleItem]) -> Dict:
        images = [inst.image for inst in instances]
        if all(x.shape == images[0].shape for x in images):
            images = torch.stack(images)

        return {
            "input_type": "single",
            "images": images,
            "task_types": [inst.task_type for inst in instances],
            "level_probs": torch.tensor([inst.level_probs for inst in instances]),
        }


def make_single_data_module(data_args) -> Dict:
    train_dataset = SingleDataset(
        data_paths=data_args.data_paths,
        data_weights=data_args.data_weights,
        data_args=data_args,
    )
    data_collator = DataCollatorForSupervisedDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
