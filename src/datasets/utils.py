from dataclasses import dataclass, field
from typing import List, Optional

import torch.distributed as dist
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def rank0_print(*args):
    try:
        if dist.get_rank() == 0:
            print(*args)
    except Exception:
        print(*args)


@dataclass
class DataArguments:
    data_paths: List[str] = field(default_factory=lambda: [])
    data_weights: List[int] = field(default_factory=lambda: [])
    dataset_type: str = "pair"
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "pad"
    image_grid_pinpoints: Optional[str] = field(default=None)


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_video(video_file):
    from decord import VideoReader

    vr = VideoReader(video_file)
    fps = vr.get_avg_fps()
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]
