import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)

# === Config ===
CSV_PATH = "pulled_grants.csv"      # your CSV file
MODEL_DIR = "./classifier"                # path to model dir (with safetensors)
TEXT_COLUMNS = ['Recipient', 'Agreement', 'Description']  # columns to combine

# === Load CSV (first 100 rows) ===
df = pd.read_csv(CSV_PATH)
df['text'] = df[TEXT_COLUMNS].fillna('').astype(str).agg(' '.join, axis=1)

# Create Hugging Face Dataset
hf_dataset = Dataset.from_pandas(df[['text']])

# === Load tokenizer & model ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True
)

# === Set up pipeline ===
pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    device=0 if torch.cuda.is_available() else -1
)

# Load clean label mapping
with open("classifier/label_encoder_classes.txt") as f:
    labels = [line.strip() for line in f if line.strip()]
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

# Inject into model config
model.config.id2label = id2label
model.config.label2id = label2id

# === Run inference ===
texts = [str(x) for x in hf_dataset['text']]
results = pipeline(texts)

# === Display results ===
for i, (text, res) in enumerate(zip(hf_dataset['text'], results)):
    print(f"{i+1:03}: {res['label']} ({res['score']:.4f}) — {text[:60]}...")