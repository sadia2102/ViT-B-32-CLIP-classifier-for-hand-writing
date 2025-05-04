# CLIP ViT-B/32 Classifier (Linear Probing)

This project uses OpenAI's CLIP (ViT-B/32) image encoder with a frozen encoder and a trainable linear head for binary classification (`real` vs `fake`).

## 🧠 Methodology

- Used CLIP's pretrained `ViT-B/32` visual encoder
- Encoder kept frozen (no gradient updates)
- Trained a single `Linear(512, 2)` classification head
- Framework: PyTorch

## 📁 Dataset Structure

data/
├── train/
│ ├── real/
│ └── fake/
├── val/
│ ├── real/
│ └── fake/
└── test/
  ├── real/
  └── fake/

bash command
	python clip_train.py


    -Batch size: 32
    -Epochs: 5–10
    -Optimizer: Adam
    -Learning rate: 1e-3
    -Frozen encoder (clip_model.visual)
    -Trainable linear head (nn.Linear(512, 2))
## 📊 Results

Set		Accuracy	Misclassified
Test		99.23%		24 images
data/field	Not evaluated (in this stage)

	
Confusion Matrix (Test Set)
		[[1993    7]
 		[  17 1111]]

## 🧪 Evaluation

	bash command:
		python evalute.py


Shows accuracy, precision, recall, and F1
Saves misclassified paths to misclassified.txt

## Fine-Tuned CLIP ViT-B/32 Classifier

This project fine-tunes OpenAI's CLIP (ViT-B/32) image encoder for binary image classification: `real` vs `fake`.

## 📦 Model

- Base: `ViT-B/32` from OpenAI CLIP
- Pretrained weights: Contrastive image-text training on 400M dataset
- Fine-tuned: Full encoder + classification head (`Linear(512, 2)`)
- Framework: PyTorch

## 📁 Dataset Structure

data/
├── train/
│ ├── real/
│ └── fake/
├── val/
│ ├── real/
│ └── fake/
└── field/
  ├── real/
  └── fake/


## 🚀 Training

```bash command
	python clip_train_finetune.py

    -Optimizer: Adam
    -Learning rate: 1e-4 (same for encoder and head)
    -Epochs: 5
    -Progress tracked with tqdm

## 📊 Results

Set	Accuracy	Misclassified
Validation	99.38%	-
data/field	99.86%	2 images
Confusion Matrix (Field Set)
[[698   2]
 [  0 774]]
## 🧪 Evaluation

```bash command:
	python evalute_clip_finetune.py


Outputs:
Accuracy, precision, recall, f1-score
Misclassified image paths saved to misclassified_clip_field.txt

