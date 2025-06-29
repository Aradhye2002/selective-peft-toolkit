from transformers import ViTForImageClassification, ViTImageProcessorFast, Trainer, TrainingArguments
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from selective_optimizers.load_store import write_summary_to_disk
from selective_optimizers.wrap import get_selective_optimizer
from torch.optim import AdamW
import torch


model_name = "google/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessorFast.from_pretrained(model_name)

dataset = load_dataset("cifar100")
normalize = Normalize(mean=[0.5]*3, std=[0.5]*3)   # or use feature_extractor stats

tx = Compose([
    Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    ToTensor(),
    normalize
])

class CIFAR100(torch.utils.data.Dataset):
    def __init__(self, tx, dataset, is_train=True):
        self.tx = tx
        self.dataset = dataset["train"] if is_train else dataset["test"]
    
    def __getitem__(self, idx):
        pixels = self.dataset[idx]["img"]
        pixels = tx(pixels)
        label = self.dataset[idx]["fine_label"]
        return {"pixel_values" : pixels, "label": label}
    
    def __len__(self):
        return len(self.dataset)
    
train_dataset = CIFAR100(tx, dataset)
test_dataset = CIFAR100(tx, dataset, is_train=False)

# 5. Define model
model = ViTForImageClassification.from_pretrained(model_name, num_labels=100)

opt = AdamW
peft_to_use = "pafi"
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
    {"params": pretrained_params, "choose_all": False}
]

num_train_epochs = 1
per_device_train_batch_size = 32
per_device_eval_batch_size = 32

optimizer = optimizer_class(
    params=params,
    lr=1e-4,
    budget=100000,
    # exp=1,
    # eps=1e-3,
    # max_steps=1563*epochs
)

# 6. Define training arguments
training_args = TrainingArguments(
    output_dir="./vit-finetuned",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_train_epochs,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="accuracy",
    report_to="none",
)

# 7. Metric
from evaluate import load as load_metric
accuracy = load_metric("accuracy")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)

# 8. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    optimizers=(optimizer, None),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# 9. Train
trainer.train()

# 10. Perform post-train checks
optimizer.post_train_work()

# 11. Get budget used
# print("Budget used:", optimizer.get_budget_used())

# 12. Construct summary and write to disk
summary = optimizer.get_summary(model)
write_summary_to_disk(f"summary.pt", summary)