import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
from selective_optimizers.load_store import write_summary_to_disk
from selective_optimizers.wrap import get_selective_optimizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model

import random
import numpy as np

# Random seed initialization
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "google/vit-base-patch16-224-in21k"
num_labels = 100
batch_size = 32
epochs = 2
lr = 1e-4

# Feature extractor
feature_extractor = ViTImageProcessor.from_pretrained(model_name)

# Transforms (match ViT expected input)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# Dataset and Dataloader
train_dataset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.01,
)

# Load ViT model
model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels, device_map="auto")

model = get_peft_model(model, config)

# Need to explicitly set the classifier as trainable since LoRA sets its requires_grad to False.
for p in model.base_model.model.classifier.parameters():
    p.requires_grad = True

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

device = next(model.parameters()).device

opt = AdamW
peft_to_use = "id3"
optimizer_class = get_selective_optimizer(opt, peft_to_use)

classifier_params = []
pretrained_params = []

for name, param in model.named_parameters():
    if name.startswith("classifier"):
        classifier_params.append(param)
    else:
        pretrained_params.append(param)

params = [
    {"params": classifier_params, "choose_all": True},
    {"params": pretrained_params, "choose_all": False},
]


optimizer = optimizer_class(
    params=params,
    lr=1e-4,
    budget=100000,
    exp=1,
    eps=1e-3,
    max_steps=1563*epochs,
    verify=True
)

# Training loop
for epoch in range(epochs):
    model.train()
    cnt = 0
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(pixel_values=inputs)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        cnt += 1
    acc = correct / total * 100
    loss = total_loss / cnt
    print(f"Train Loss: {loss:.4f}, Train Accuracy: {acc:.2f}%")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(pixel_values=inputs)
        _, preds = outputs.logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

test_acc = correct / total * 100
print(f"Test Accuracy: {test_acc:.2f}%")

optimizer.post_train_work()

print("Budget used:", optimizer.get_budget_used())

summary = optimizer.get_summary(model)
write_summary_to_disk("summary.pt", summary)