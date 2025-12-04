import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os


# ============================================================
# Settings
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
train_dir = "data/train"
val_dir = "data/val"

num_classes = 13
batch_size = 32
num_epochs = 29
model_save_path = "chess_square_resnet18_v2.pth"


# ============================================================
# Data Augmentation
# ============================================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# ============================================================
# Datasets & Dataloaders
# ============================================================
train_data = datasets.ImageFolder(train_dir, transform=transform_train)
val_data = datasets.ImageFolder(val_dir, transform=transform_val)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


# ============================================================
# Model Setup
# ============================================================
model = models.resnet18(pretrained=True)

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Strategy:
# Freeze first half of the network, unfreeze last 2 residual blocks + classifier
for name, param in model.named_parameters():
    param.requires_grad = False  # freeze all

for name, param in model.layer3.named_parameters():
    param.requires_grad = True   # unfreeze layer3
for name, param in model.layer4.named_parameters():
    param.requires_grad = True   # unfreeze layer4

for param in model.fc.parameters():
    param.requires_grad = True   # unfreeze classifier

model = model.to(device)

print("Trainable parameters:")
for name, p in model.named_parameters():
    if p.requires_grad:
        print("  ", name)


# ============================================================
# Loss, Optimizer, Scheduler
# ============================================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Differential learning rates:
optimizer = optim.Adam([
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

scaler = GradScaler()  # mixed precision


# ============================================================
# Training Loop
# ============================================================
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():  # mixed precision
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_data)
    scheduler.step(epoch)  # Update LR schedule


    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f}")

    # -------------------------
    # Save best model
    # -------------------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"🔥 New best model saved! Acc = {best_val_acc:.4f}")


print("Training complete.")
print(f"Best model saved as {model_save_path}")
