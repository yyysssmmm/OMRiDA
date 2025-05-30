# test/test_dataset.py

# test/test_dataset.py 맨 위에 추가
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))  # 상위 폴더 OMRiDA를 경로에 추가

from data.dataset import FormulaDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128, 512)),
    transforms.ToTensor()
])

dataset = FormulaDataset(
    image_dir="data/CROHME/data_crohme/2014/pme_img",
    caption_path="data/CROHME/data_crohme/2014/caption.txt",
    image_ext="png",
    transform=transform
)

print(len(dataset))
print(dataset[0][0].min(), dataset[0][0].max())  # (Tensor, LaTeX string)
