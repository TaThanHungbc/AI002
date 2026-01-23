# main.py
import os
import argparse
from pathlib import Path
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision import transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import process_image_resize
from model import get_resnet18

# -------------------------
# Dataset
# -------------------------
class HandwritingDataset(Dataset):
    def __init__(self, paths, labels, transform=None, debug=False):
        self.paths = paths
        self.labels = labels
        self.debug = debug
        # đảm bảo transform luôn hợp lệ (dùng ImageNet normalization mặc định nếu None)
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        label = int(self.labels[idx])
        try:
            img_np, lap_var = process_image_resize(p, target_size=(224,224), debug=False)
            if img_np is None:
                raise ValueError("process_image_resize returned None")
            # convert numpy to PIL Image
            img_pil = Image.fromarray(img_np)
            img_t = self.transform(img_pil)  # tensor, shape (C,H,W), dtype=float
            # ensure dtype/shape
            if not torch.is_tensor(img_t):
                img_t = T.ToTensor()(img_pil)
            # make sure shape is correct
            if img_t.ndim == 3 and img_t.shape[0] == 1:
                # single channel -> replicate to 3 channels
                img_t = img_t.repeat(3,1,1)
            if img_t.shape[1] != 224 or img_t.shape[2] != 224:
                # resize fallback (rare)
                img_t = T.Resize((224,224))(img_pil)
                img_t = self.transform(Image.fromarray(np.array(img_t)))
            return img_t, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            # Log lỗi (worker vẫn in được khi num_workers=0; nếu >0, xem log trên Colab console)
            print(f"[Dataset ERROR] idx={idx} path={p} error={e}")
            # Trả về dummy sample (không crash) — có thể gây ảnh hưởng huấn luyện nhẹ, nhưng tránh dừng toàn bộ
            dummy = torch.zeros((3,224,224), dtype=torch.float32)
            return dummy, torch.tensor(label, dtype=torch.long)

# -------------------------
# Utilities
# -------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -------------------------
# Train & Eval loops
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="eval", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

# -------------------------
# Main
# -------------------------
def main(args):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # tìm tất cả ảnh trong 2 thư mục
    human_dir = Path(args.dataset) / "human"
    ai_dir = Path(args.dataset) / "ai"

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    human_paths = [str(p) for p in human_dir.rglob("*") if p.suffix.lower() in exts]
    ai_paths = [str(p) for p in ai_dir.rglob("*") if p.suffix.lower() in exts]

    print(f"Found {len(human_paths)} human images, {len(ai_paths)} ai images.")

    paths = human_paths + ai_paths
    labels = [0]*len(human_paths) + [1]*len(ai_paths)

    # chia train/test
    train_p, test_p, train_y, test_y = train_test_split(paths, labels, test_size=args.test_size,
                                                        stratify=labels, random_state=42)
    # optional small val split from train
    train_p, val_p, train_y, val_y = train_test_split(train_p, train_y, test_size=args.val_size,
                                                      stratify=train_y, random_state=42)

    print(f"Train {len(train_p)}, Val {len(val_p)}, Test {len(test_p)}")

    # transforms: dùng ImageNet normalization cho ResNet pretrained
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    train_ds = HandwritingDataset(train_p, train_y, transform=transform)
    val_ds = HandwritingDataset(val_p, val_y, transform=transform)
    test_ds = HandwritingDataset(test_p, test_y, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = get_resnet18(num_classes=2, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0
    best_epoch = -1
    save_path = args.save_path

    for epoch in range(1, args.epochs+1):
        print(f"Epoch {epoch}/{args.epochs}")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train loss {train_loss:.4f} acc {train_acc:.4f}")
        print(f"  Val   loss {val_loss:.4f} acc {val_acc:.4f}  time {(time.time()-t0):.1f}s")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({'model_state_dict': model.state_dict(),
                        'args': vars(args)}, save_path)
            print(f"  Saved best model to {save_path}")

    print(f"Best val acc {best_val_acc:.4f} at epoch {best_epoch}")

    # Load best and evaluate on test
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc, test_labels, test_preds = eval_model(model, test_loader, criterion, device)
    print("=== Test results ===")
    print(f"Test loss {test_loss:.4f} acc {test_acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))
    print("Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Human","AI"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./dataset", help="Folder chứa thư mục human/ và ai/")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
