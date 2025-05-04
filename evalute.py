import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import clip
import numpy as np
import os

from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load model
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.visual.half()

# Define classifier model
class CLIPLinearClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.half()
        with torch.no_grad():
            features = self.image_encoder(x)
        return self.classifier(features.float())

# Load test dataset
test_dataset = ImageFolder(root="data/test", transform=preprocess)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = CLIPLinearClassifier(clip_model, num_classes=2).to(device)
model.load_state_dict(torch.load("best_clip_classifier.pth", map_location=device))
model.eval()

# Evaluation
all_preds = []
all_labels = []
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Save misclassified file names
        for i, (p, l) in enumerate(zip(preds, labels)):
            if p != l:
                img_path, _ = test_dataset.samples[len(all_preds) - len(labels) + i]
                misclassified.append(img_path)

# Metrics
print("✅ Test Accuracy: {:.2f}%".format(100 * np.mean(np.array(all_preds) == np.array(all_labels))))
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_dataset.classes, yticklabels=test_dataset.classes, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Optional: save misclassified image paths
with open("misclassified.txt", "w") as f:
    for path in misclassified:
        f.write(path + "\n")
print(f"\nMisclassified {len(misclassified)} images. Saved list to misclassified.txt")
