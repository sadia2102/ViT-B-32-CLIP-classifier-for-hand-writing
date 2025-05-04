import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import clip
import os

# Use MPS (Mac GPU) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load CLIP model and preprocessing
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.visual.half()  # Convert CLIP encoder to float16 for MPS compatibility


# Freeze CLIP weights
for p in clip_model.parameters():
    p.requires_grad = False

# Dataset paths
train_dir = "data/train"
val_dir = "data/val"
test_dir = "data/test"

# Datasets
train_dataset = ImageFolder(root=train_dir, transform=preprocess)
val_dataset = ImageFolder(root=val_dir, transform=preprocess)
test_dataset = ImageFolder(root=test_dir, transform=preprocess)

# Dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Classifier model
class CLIPLinearClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.half()  # Convert input to float16 to match model weights
        with torch.no_grad():
            features = self.image_encoder(x)
        return self.classifier(features.float())  # Convert back to float32 for classifier


num_classes = len(train_dataset.classes)
model = CLIPLinearClassifier(clip_model, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

# Training loop
best_val_acc = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total * 100
    val_loss_avg = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val   Loss: {val_loss_avg:.4f}, Val   Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_clip_classifier.pth")
        print("  Saved best model.")

# Final test accuracy
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

print(f"\nTest Accuracy: {100 * test_correct / test_total:.2f}%")
