import os
import pandas as pd
import hashlib
import logging
import torch
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline
)
import dotenv

dotenv.load_dotenv()

# === Logging Setup ===
log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_file_handler = logging.FileHandler("logs/uploader.log")
log_file_handler.setFormatter(log_formatter)

error_file_handler = logging.FileHandler("logs/error.log")
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(log_file_handler)
logger.addHandler(error_file_handler)

# --- Config ---
CSV_FILE_PATH = 'pulled_grants.csv'
DATABASE_URL = os.getenv("DATABASE_URL")
TABLE_NAME = "grants"
HASH_TABLE = "grants_hashes"
TEXT_COLUMNS = ['Recipient', 'Agreement', 'Description']
MODEL_DIR = "./classifier"
LABELS_FILE = "./classifier/label_encoder_classes.txt"

def hash_row(row):
    row_str = '|'.join(str(v) for v in row)
    return hashlib.sha256(row_str.encode()).hexdigest()

def load_pipeline():
    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        trust_remote_code=True
    )
    with open(LABELS_FILE) as f:
        labels = [line.strip() for line in f if line.strip()]
    id2label = {i: label for i, label in enumerate(labels)}
    label2id = {label: i for i, label in enumerate(labels)}
    model.config.id2label = id2label
    model.config.label2id = label2id

    pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        return_all_scores=False,
        device=0 if torch.cuda.is_available() else -1
    )
    return pipe

def add_predictions(df, pipeline):
    logger.info("Running inference...")
    df['text'] = df[TEXT_COLUMNS].fillna('').astype(str).agg(' '.join, axis=1)
    hf_dataset = Dataset.from_pandas(df[['text']])
    texts = list(hf_dataset['text'])
    results = pipeline(texts)
    df['predicted_label'] = [r['label'] for r in results]
    df['predicted_score'] = [r['score'] for r in results]
    return df.drop(columns=['text'])

def load_and_hash_data():
    logger.info(f"Loading CSV: {CSV_FILE_PATH}")
    df = pd.read_csv(CSV_FILE_PATH)
    df['row_hash'] = df.apply(hash_row, axis=1)
    logger.info(f"Loaded {len(df)} total rows.")
    return df

def get_existing_hashes(engine):
    logger.info("Fetching existing hashes...")
    with engine.connect() as conn:
        conn.execute(text(f"CREATE TABLE IF NOT EXISTS {HASH_TABLE} (hash TEXT PRIMARY KEY);"))
        result = conn.execute(text(f"SELECT hash FROM {HASH_TABLE};"))
        return set(row[0] for row in result)

def upload_new_rows(df_new, engine):
    logger.info(f"Overwriting data in tables: {TABLE_NAME} and {HASH_TABLE}...")

    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM {TABLE_NAME};"))
        conn.execute(text(f"DELETE FROM {HASH_TABLE};"))

    df_to_upload = df_new.drop(columns=['row_hash'])

    df_to_upload.to_sql(TABLE_NAME, engine, if_exists='append', index=False)
    pd.DataFrame(df_new['row_hash'], columns=['hash']).to_sql(HASH_TABLE, engine, if_exists='append', index=False)

    logger.info(f"Overwrote {len(df_new)} rows in both tables.")

def main():
    start_time = time.time()
    logger.info("Starting daily uploader...")

    try:
        engine = create_engine(DATABASE_URL)
        pipeline = load_pipeline()

        df = load_and_hash_data()
        existing_hashes = get_existing_hashes(engine)
        df_new = df[~df['row_hash'].isin(existing_hashes)]

        logger.info(f"Identified {len(df_new)} new rows for processing.")

        if not df_new.empty:
            df_new = add_predictions(df_new, pipeline)

        upload_new_rows(df_new, engine)

    except Exception as e:
        logger.error("An error occurred:", exc_info=True)
    finally:
        duration = time.time() - start_time
        logger.info(f"Done. Duration: {duration:.2f} seconds.")

if __name__ == "__main__":
    main()