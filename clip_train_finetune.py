import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model and preprocessing
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model = clip_model.to(device)
clip_model.visual.float()  # Ensure model is float32 for MPS
clip_model.eval()  # We'll remove this shortly

# Define fine-tuning classifier
class CLIPFineTuneClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.image_encoder = clip_model.visual  # Make this trainable
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.image_encoder(x)  # (B, 512)
        return self.classifier(features)

# Use CLIP's own preprocessing
transform = preprocess

def train():
    # Load datasets
    train_dataset = ImageFolder("data/train", transform=transform)
    val_dataset = ImageFolder("data/val", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    model = CLIPFineTuneClassifier(clip_model, num_classes=len(train_dataset.classes)).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": model.image_encoder.parameters(), "lr": 1e-5},{"params": model.classifier.parameters(), "lr": 1e-4}])  # Lower LR for fine-tuning

    best_val_acc = 0.0
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
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
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        val_loss /= len(val_loader)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "finetuned_clip_classifier.pth")
            print("  ✅ Saved best model.\n")

if __name__ == "__main__":
    train()
