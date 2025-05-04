import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import clip
from tqdm import tqdm
import numpy as np

# Set device
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.visual.float()  # Ensure it's in float32 for MPS

# Define your fine-tuned classifier
class CLIPFineTuneClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = self.image_encoder(x)
        return self.classifier(features)

# Load model with saved weights
model = CLIPFineTuneClassifier(clip_model, num_classes=2).to(device)
model.load_state_dict(torch.load("finetuned_clip_classifier.pth", map_location=device))
model.eval()

# Load field dataset
field_dataset = ImageFolder("data/field", transform=preprocess)
field_loader = DataLoader(field_dataset, batch_size=32, shuffle=False)

class_names = field_dataset.classes
print("Class Mapping:", field_dataset.class_to_idx)

# Evaluate
all_preds = []
all_labels = []
misclassified = []

with torch.no_grad():
    for images, labels in tqdm(field_loader, desc="Evaluating on field data"):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Track misclassified images
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                path, _ = field_dataset.samples[len(all_labels) - len(preds) + i]
                misclassified.append(path)

# Results
acc = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\n✅ Field Accuracy: {acc:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\n🧱 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# Save misclassified
if misclassified:
    with open("misclassified_clip_field.txt", "w") as f:
        for path in misclassified:
            f.write(f"{path}\n")
    print(f"\n❌ Misclassified {len(misclassified)} images. Saved to misclassified_clip_field.txt")
else:
    print("\n✅ No misclassifications!")

