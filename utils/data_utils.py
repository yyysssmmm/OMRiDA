from torchvision import transforms
import torch

def get_formula_transform(image_type: str):
    if image_type == "crohme_hme":
        return transforms.Compose([
            transforms.Resize((128, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.077], std=[0.249])
        ])
    elif image_type == "crohme_pme":
        return transforms.Compose([
            transforms.Resize((64, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.11], std=[0.235])
        ])
    elif image_type == "unpaired_pme":
        return transforms.Compose([
            transforms.CenterCrop((2339, 1654)),
            transforms.Resize((128, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[3.7e-5], std=[0.0014])
        ])
    else:
        raise ValueError(f"Unsupported image_type: {image_type}")

def formula_collate_fn(batch, pad_idx=0):
    images = torch.stack([item["image"] for item in batch], dim=0)
    formulas = torch.nn.utils.rnn.pad_sequence(
        [item["formula"] for item in batch],
        batch_first=True,
        padding_value=pad_idx
    )
    return {"image": images, "formula": formulas}
