import torch
from transformers import AutoTokenizer, AutoConfig
from model import (
    BertForSentenceClassification,
    BertMoCoForSentenceClassification,
)

import json
import os
from transformers import AutoTokenizer, AutoConfig

model_dir = "/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/genai_poc/main/lic_poc/InfoCL/model_save/fast_model"

# ---- Load raw config.json manually ----
config_path = os.path.join(model_dir, "config.json")
with open(config_path, "r") as f:
    config_dict = json.load(f)

# ---- Fix id2label if it's a list ----
if isinstance(config_dict.get("id2label"), list):
    config_dict["id2label"] = {i: label for i, label in enumerate(config_dict["id2label"])}

# ---- Fix label2id if necessary ----
if isinstance(config_dict.get("label2id"), list):
    config_dict["label2id"] = {label: i for i, label in enumerate(config_dict["label2id"])}

# ---- Save fixed config to a temp file ----
fixed_config_path = os.path.join(model_dir, "fixed_config.json")
with open(fixed_config_path, "w") as f:
    json.dump(config_dict, f, indent=2)

# ---- Now load safely ----
config = AutoConfig.from_pretrained(fixed_config_path)

from types import SimpleNamespace

# Convert config.global_args back to SimpleNamespace if it's a dict
if isinstance(config.global_args, dict):
    config.global_args = SimpleNamespace(**config.global_args)
# -------------------------
# 4. Load model
# -------------------------
model = BertMoCoForSentenceClassification.from_pretrained(model_dir, config=config)
model.eval()

# -------------------------
# 5. Prepare text for inference
# -------------------------
sentence = "I want to update my LIC policy details."
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
inputs = tokenizer(
    sentence,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

# -------------------------
# 6. Run model
# -------------------------
with torch.no_grad():
    outputs = model(**inputs)

# -------------------------
# 7. Decode prediction
# -------------------------
if hasattr(outputs, "logits"):
    logits = outputs.logits
else:
    logits = outputs[0]  # fallback for tuple outputs

predicted_class_id = torch.argmax(logits, dim=-1).item()
predicted_label = config.id2label[predicted_class_id]

print(f"\nSentence: {sentence}")
print(f"Predicted label: {predicted_label}")
