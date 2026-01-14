import os
from PIL import Image, ImageFilter
import random

HR_DIR = "data/images"
LR_DIR = "data/lr"
os.makedirs(LR_DIR, exist_ok=True)

SCALE = 4  # x4 super resolution

for img_name in os.listdir(HR_DIR):
    img = Image.open(os.path.join(HR_DIR, img_name)).convert("RGB")

    w, h = img.size
    lr = img.resize((w//SCALE, h//SCALE), Image.BICUBIC)
    lr = lr.resize((w, h), Image.BICUBIC)

    # slight blur
    if random.random() < 0.5:
        lr = lr.filter(ImageFilter.GaussianBlur(radius=1))

    lr.save(os.path.join(LR_DIR, img_name), quality=70)
