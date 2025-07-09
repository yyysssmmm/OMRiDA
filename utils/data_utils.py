from torchvision import transforms
import torch
from PIL import ImageOps
from torchvision.transforms import functional as TF

class DynamicPadToSize:
    def __init__(self, target_size=(2339, 1654), fill=255):
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        pad_left = max((self.target_w - w) // 2, 0)
        pad_top = max((self.target_h - h) // 2, 0)
        pad_right = max(self.target_w - w - pad_left, 0)
        pad_bottom = max(self.target_h - h - pad_top, 0)

        # 🔧 fill 값 조정: RGB면 튜플로 변환
        fill_value = self.fill
        if img.mode == "RGB" and isinstance(self.fill, int):
            fill_value = (self.fill,) * 3

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill_value)
        else:
            return img


def get_formula_transform(image_type: str, transform_config: dict):
    cfg = transform_config.get(image_type)
    if cfg is None:
        raise ValueError(f"Unsupported image_type: {image_type}")

    target_size = tuple(cfg["target_size"])
    mean = cfg["mean"]
    std = cfg["std"]

    return transforms.Compose([
        DynamicPadToSize(target_size=target_size, fill=255),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def formula_collate_fn(batch, pad_idx=0):
    images = torch.stack([item["image"] for item in batch], dim=0)
    formulas = torch.nn.utils.rnn.pad_sequence(
        [item["formula"] for item in batch],
        batch_first=True,
        padding_value=pad_idx
    )
    return {"image": images, "formula": formulas}


def paired_collate_fn(batch, pad_idx=0):
    imgs_hme = torch.stack([item["img_hme"] for item in batch], dim=0)
    imgs_pme = torch.stack([item["img_pme"] for item in batch], dim=0)
    formulas = torch.nn.utils.rnn.pad_sequence(
        [item["formula"] for item in batch],
        batch_first=True,
        padding_value=pad_idx
    )
    return {
        "img_hme": imgs_hme,
        "img_pme": imgs_pme,
        "formula": formulas
    }