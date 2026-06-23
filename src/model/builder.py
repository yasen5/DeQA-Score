import torch
from transformers.models.clip.image_processing_clip import CLIPImageProcessor

from src.model import ViTForIQA


def load_pretrained_model(
    model_path,
    device="cuda",
    load_8bit=False,
    load_4bit=False,
    torch_dtype=None,
    preprocessor_path=None,
):
    if torch_dtype is None:
        torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = ViTForIQA.from_pretrained(model_path, torch_dtype=torch_dtype)
    model = model.to(device)
    model.eval()

    if preprocessor_path is None:
        preprocessor_path = model_path
    image_processor = CLIPImageProcessor.from_pretrained(preprocessor_path)

    return model, image_processor
