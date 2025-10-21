import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
import numpy as np
import os
import re
from PIL import Image

# --- Configuration ---
DATA_DIR = "./training"
OUTPUT_MODEL_PATH = "fcn_road_segmentation_v2.pth"
NUM_EPOCHS = 25
LEARNING_RATE_DECODER = 0.001
LEARNING_RATE_ENCODER = 1e-5
BATCH_SIZE = 8
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 576
VALIDATION_SPLIT = 0.2

# --- FCN Model Definition ---
class FCN(nn.Module):
    def __init__(self, num_classes=2):
        super(FCN, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.encoder = vgg.features
        for param in self.encoder.parameters():
            param.requires_grad = False
        for layer in self.encoder[24:]:
            for param in layer.parameters():
                param.requires_grad = True
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, num_classes, 3, 2, 1, 1))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# --- Dataset Definitions (at top level to avoid pickling errors) ---
class KittiRoadDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        def get_key(f): return os.path.splitext(f)[0].replace('_road','').replace('_lane','')
        image_map = {get_key(f): f for f in os.listdir(image_dir)}
        label_map = {get_key(f): f for f in os.listdir(label_dir)}
        keys = sorted(list(set(image_map.keys()).intersection(set(label_map.keys()))))
        self.image_files = [image_map[k] for k in keys]
        self.label_files = [label_map[k] for k in keys]
        if len(self.image_files) == 0: raise ValueError("No matching pairs found.")
        print(f"Successfully matched {len(self.image_files)} pairs.")
        self.road_color = np.array([255, 0, 255])
    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        lbl_path = os.path.join(self.label_dir, self.label_files[idx])
        image = Image.open(img_path).convert("RGB")
        label_image = Image.open(lbl_path).convert("RGB")
        mask = np.all(np.array(label_image) == self.road_color, axis=-1)
        mask = Image.fromarray(mask.astype(np.uint8) * 255)
        return image, mask

class SplitDataset(Dataset):
    def __init__(self, subset, image_transform, label_transform):
        self.subset = subset
        self.image_transform = image_transform
        self.label_transform = label_transform
    def __getitem__(self, idx):
        image, mask = self.subset.dataset[self.subset.indices[idx]]
        return self.image_transform(image), self.label_transform(mask).squeeze(0).long()
    def __len__(self):
        return len(self.subset)

def check_iou(loader, model, device="cpu"):
    model.eval()
    total_iou = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = torch.argmax(model(x), dim=1)
            intersection = (preds & y).float().sum((1, 2))
            union = (preds | y).float().sum((1, 2))
            iou = (intersection + 1e-6) / (union + 1e-6)
            total_iou += iou.mean().item()
    avg_iou = total_iou / len(loader)
    print(f"Validation IoU: {avg_iou*100:.2f}%")
    model.train()

def main():
    image_folder = os.path.join(DATA_DIR, 'image_2')
    label_folder = os.path.join(DATA_DIR, 'gt_image_2')
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    label_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    full_dataset = KittiRoadDataset(image_folder, label_folder)
    train_size = int((1-VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_dataset = SplitDataset(train_subset, train_transform, label_transform)
    val_dataset = SplitDataset(val_subset, val_transform, label_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FCN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.decoder.parameters(), 'lr': LEARNING_RATE_DECODER},
        {'params': model.encoder[24:].parameters(), 'lr': LEARNING_RATE_ENCODER}
    ])

    print(f"\n--- Starting Training on {device} ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")
        check_iou(val_loader, model, device)
        
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
    print(f"\n--- Training Finished. Model saved to {OUTPUT_MODEL_PATH} ---")

if __name__ == "__main__":
    main()
