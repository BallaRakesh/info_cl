import torch
from transformers import AutoTokenizer, AutoConfig
from model import (
    BertForSentenceClassification,
    BertMoCoForSentenceClassification,
    BertMoCoForSentenceClassificationDistbert
)

import json
import os
from transformers import AutoTokenizer, AutoConfig

model_dir = "/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/genai_poc/main/lic_poc/model_save/slow_model"

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
model = BertMoCoForSentenceClassificationDistbert.from_pretrained(model_dir, config=config)#, ignore_mismatched_sizes=True)
model.eval()

# -------------------------
# 5. Prepare text for inference
# -------------------------
sentence = """From: Priya Sharma [priyasharma@gmail.com]
Date: 15 October 2024
Subject: Discrepancy in Premium Payment Statement

Dear Customer Service,

I hope this email finds you well. I am writing to express my concern regarding my policy number 987654321. I recently received my premium payment statement, and I noticed that my payments for the last two quarters have not been accurately reflected. 

According to my records, I made payments of Rs 5,000 on 15th July 2024 and Rs 5,000 on 15th September 2024, but these amounts are missing from the statement you provided. I rely on this statement for my tax records, and it is crucial that it reflects the correct payment history.

I would appreciate it if you could look into this matter urgently and provide me with an updated statement that accurately reflects all my payments. Thank you for your attention to this issue.

Best regards,
Priya Sharma
Mobile: 9876543210

"""
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
