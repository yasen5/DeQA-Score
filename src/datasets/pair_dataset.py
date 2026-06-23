import copy
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import expand2square, rank0_print


class PairDataset(Dataset):
    """Dataset for pairwise quality ranking."""

    def __init__(self, data_paths, data_weights, data_args):
        super().__init__()
        dataset_list = []
        for data_path, data_weight in zip(data_paths, data_weights):
            data_list = json.load(open(data_path, "r"))
            dataset_list.append(data_list * data_weight)
        self.dataset_list = dataset_list

        nums_eachdata = [len(d) for d in dataset_list]
        nums_predata = copy.deepcopy(nums_eachdata)
        for idx in range(1, len(nums_predata)):
            nums_predata[idx] += nums_predata[idx - 1]

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.nums_eachdata = nums_eachdata
        self.nums_predata = nums_predata
        self.data_args = data_args
        assert self.nums_predata[-1] == sum(self.nums_eachdata)

    def __len__(self):
        return self.nums_predata[-1]

    def next_rand(self):
        return random.randint(0, len(self) - 1)

    def __getitem__(self, i):
        while True:
            try:
                if i < self.nums_predata[0]:
                    idx_dataset, idx_sample = 0, i
                else:
                    for idx_dataset in range(1, len(self.nums_predata)):
                        if self.nums_predata[idx_dataset - 1] <= i < self.nums_predata[idx_dataset]:
                            idx_sample = i - self.nums_predata[idx_dataset - 1]
                            break

                item_A = self._get_one(idx_dataset, idx_sample)
                while True:
                    idx_sample_B = random.randint(0, self.nums_eachdata[idx_dataset] - 1)
                    if idx_sample_B != idx_sample:
                        break
                item_B = self._get_one(idx_dataset, idx_sample_B)
                return {"item_A": item_A, "item_B": item_B}
            except Exception as ex:
                print(ex)
                i = self.next_rand()
                continue

    def _get_one(self, idx_dataset, idx_sample) -> Dict[str, torch.Tensor]:
        sample = self.dataset_list[idx_dataset][idx_sample]
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor

        image_file = sample.get("image")
        if image_file is not None:
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            if self.data_args.image_aspect_ratio == "pad":
                image = expand2square(
                    image, tuple(int(x * 255) for x in processor.image_mean)
                )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            crop_size = processor.crop_size
            image = torch.zeros(3, crop_size["height"], crop_size["width"])

        return {
            "image": image,
            "task_type": sample.get("task_type", "score"),
            "gt_score": sample.get("gt_score", -10000.0),
            "std": sample.get("std", -10000.0),
            "level_probs": sample.get("level_probs", [-10000.0] * 5),
        }


@dataclass
class DataCollatorForPairDataset:
    """Collate paired samples into a batch."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances_A = [inst["item_A"] for inst in instances]
        instances_B = [inst["item_B"] for inst in instances]
        return {
            "input_type": "pair",
            "item_A": self._collate_one(instances_A),
            "item_B": self._collate_one(instances_B),
        }

    def _collate_one(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        images = [inst["image"] for inst in instances]
        if all(x.shape == images[0].shape for x in images):
            images = torch.stack(images)

        return {
            "images": images,
            "task_types": [inst["task_type"] for inst in instances],
            "gt_scores": torch.tensor([inst["gt_score"] for inst in instances]),
            "stds": torch.tensor([inst["std"] for inst in instances]),
            "level_probs": torch.tensor([inst["level_probs"] for inst in instances]),
        }


def make_pair_data_module(data_args) -> Dict:
    train_dataset = PairDataset(
        data_paths=data_args.data_paths,
        data_weights=data_args.data_weights,
        data_args=data_args,
    )
    data_collator = DataCollatorForPairDataset()
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
