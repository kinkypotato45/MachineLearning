import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoImageProcessor, TrainingArguments, Trainer, AutoModelForSequenceClassification, ResNetForImageClassification, DefaultDataCollator
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import huggingface_hub
from torch.utils.data import DataLoader


signs = load_dataset("bazyl/GTSRB")
labels = signs["train"].features["ClassId"].names
# print(labels)

label2id, id2label = {}, {}

for i, label in enumerate(labels):

    label2id[label] = str(i)

    id2label[str(i)] = label


checkpoint = "microsoft/resnet-50"
# checkpoint = "google/vit-base-patch16-224-in21k"
model = ResNetForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    ignore_mismatched_sizes=True,
    id2label=id2label,
    label2id=label2id,
)
print(id2label)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
# print(image_processor)

normalize = Normalize(mean=image_processor.image_mean,
                      std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


def transforms(examples):
    examples["pixel_values"] = [_transforms(
        img.convert("RGB")) for img in examples["Path"]]
    del examples["Path"]
    return examples


data_collator = DefaultDataCollator()
signs = signs.with_transform(transforms)
train_dataloader = DataLoader(
    signs["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    signs["test"], batch_size=8, collate_fn=data_collator
)
# for batch in train_dataloader:
#     print({k: v.shape for k, v in batch.items()})
# print(signs)
# print(signs["train"][0])
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
print(torch.cuda.is_available())

# accuracy = evaluate.load("accuracy")
#
#
# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)
#
#
# training_args = TrainingArguments(
#     output_dir="roadSigns",
#     remove_unused_columns=False,
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=5e-5,
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=4,
#     per_device_eval_batch_size=16,
#     num_train_epochs=1,
#     warmup_ratio=0.1,
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="accuracy",
#     # push_to_hub=True,
#     # push_to_hub_model_id="roadSigns"
#     hub_token="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",
# )
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
#     train_dataset=signs["train"],
#     eval_dataset=signs["test"],
#     tokenizer=image_processor,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
# trainer.push_to_hub("Bliu3/roadSigns")
