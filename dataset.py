import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.images = sorted(os.listdir(lr_dir))
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lr = Image.open(os.path.join(self.lr_dir, self.images[idx])).convert("RGB")
        hr = Image.open(os.path.join(self.hr_dir, self.images[idx])).convert("RGB")
        return self.to_tensor(lr), self.to_tensor(hr)
