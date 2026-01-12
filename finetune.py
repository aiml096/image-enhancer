import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import Generator

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
BATCH_SIZE = 100

EPOCHS = 200
LR = 1e-4

# ---------------- DATASET ----------------
class SingleFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png",".jpeg"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        label = 0  # dummy label (replace later)
        return img, label


# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

dataset = SingleFolderDataset("data/images", transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- MODEL ----------------
model = Generator(num_classes=2).to(DEVICE)

state = torch.load("RealESRGAN_x4.pth", map_location=DEVICE)

# 🔥 IMPORTANT: load safely
missing, unexpected = model.load_state_dict(state, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

# ---------------- FREEZE BACKBONE ----------------
for param in model.features.parameters():
    param.requires_grad = False

# ---------------- TRAIN SETUP ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# ---------------- TRAIN LOOP ----------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss:.4f}")

# ---------------- SAVE ----------------
torch.save(model.state_dict(), "finetuned_model.pth")
print("✅ Fine-tuning complete")
