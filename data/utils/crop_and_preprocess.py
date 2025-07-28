# im2latex pme 데이터셋은 모두 (1654, 2339) 크기이지만, 수식부분은 이 중 매우 적은 부분만 차지
# 따라서 수식부분이 있는 부분만 잘라내어 학습이 용이하도록 함

from PIL import Image
import numpy as np
from pathlib import Path

def crop(image_path, output_path, ext="png"):
    
    for i, img_path in enumerate(sorted(image_path.glob(f"*.{ext}"))):
        try:
            with Image.open(img_path) as img:
                
                img = img.convert("RGB")
                img_np = np.array(img)
                text_area = (img_np == 0)

                ys, xs, _ = np.where(text_area)

                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()

                crop_box = (x0-10, y0-10, x1+10, y1+10)
                cropped_img = img.crop(crop_box)

                # 잘라낸 이미지 저장
                cropped_img.save(output_path / img_path.name, format="PNG")

        except Exception as e:
            print(f"❌ {img_path.name}: {e}")

def preprocess(image_path, output_path, ext="png"):
    
    for i, img_path in enumerate(sorted(image_path.glob(f"*.{ext}"))):
        try:
            with Image.open(img_path) as img:
                
                img = img.convert("RGB")
                img_name = img_path.name.split(".")[0]
                img_name = img_name + '.png'
                img.save(output_path / img_name, format="PNG")

        except Exception as e:
            print(f"❌ {img_path.name}: {e}")
        
crop(Path("../preprocessed/train/paired/im2latex/pme").resolve(), Path("../preprocessed/train/paired/im2latex/pme_cropped").resolve())
preprocess(Path("../preprocessed/train/paired/crohme/pme").resolve(), Path("../preprocessed/train/paired/crohme/pme_preprocessed").resolve(), ext="png")
preprocess(Path("../preprocessed/train/paired/crohme/hme").resolve(), Path("../preprocessed/train/paired/crohme/hme_preprocessed").resolve(), ext="bmp")

# tmp_path = Path("../preprocessed/train/paired/im2latex/pme/2400.png").resolve()
# tmp_path = Path("../preprocessed/train/paired/im2latex/hme/2400.png").resolve()
# tmp_path = Path("../preprocessed/train/paired/crohme/pme/2233.png").resolve()
# tmp_path = Path("../preprocessed/train/paired/crohme/hme/70_caue.bmp").resolve()

# img = Image.open(tmp_path)
# img = img.convert("RGB")
# print(img.mode)
# img_np = np.array(img)
# text_area = (img_np == 0)

# ys, xs, c = np.where(text_area)


# y0, y1 = ys.min(), ys.max()
# x0, x1 = xs.min(), xs.max()

# print(x0, y0, x1, y1)

# crop_box = (x0-1, y0-1, x1+1, y1+1)
# cropped_img = img.crop(crop_box)

# import matplotlib.pyplot as plt
# plt.imshow(cropped_img) 
# plt.show()

