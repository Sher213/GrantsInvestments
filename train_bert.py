import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_and_prepare_data(csv_path):
    logging.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    assert 'category' in df.columns, "CSV must contain a 'category' column"

    CATEGORIES = [
    "Housing & Shelter",
    "Education & Training",
    "Employment & Entrepreneurship",
    "Business & Innovation",
    "Health & Wellness",
    "Environment & Energy",
    "Community & Nonprofits",
    "Research & Academia",
    "Indigenous Programs",
    "Public Safety & Emergency Services",
    "Agriculture & Rural Development",
    "Arts, Culture & Heritage",
    "Civic & Democratic Engagement"
    ]

    # Drop rows with 'category not in CATEGORIES'
    df = df[df['category'].apply(lambda x: x.strip('"\n')).isin(CATEGORIES)].reset_index(drop=True)

    print(f"Filtered dataset size: {len(df)} rows")

    logging.info("Combining text features into single field")
    df['text'] = (
        df['title'].fillna('') + ' ' +
        df['agreement_title'].fillna('') + ' ' +
        df['description'].fillna('')
    )

    logging.info("Encoding labels")
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    logging.info("Splitting into train/test sets")
    train_df, test_df = train_test_split(
        df[['text', 'label']],
        test_size=0.1,
        random_state=42
    )

    logging.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    return train_ds, test_ds, le

def tokenize_and_encode(dataset, tokenizer):
    logging.info("Tokenizing dataset")
    def fn(examples):
        return tokenizer(examples['text'], truncation=True)
    return dataset.map(fn, batched=True)

def main():
    DATA_PATH = os.getenv('GRANT_CSV_PATH', 'categorized_grants_sample.csv')
    MODEL_NAME = 'bert-base-uncased'
    OUTPUT_DIR = './grant_classifier'
    NUM_EPOCHS = 1
    BATCH_SIZE = 8

    logging.info("Starting training pipeline")
    train_ds, test_ds, label_encoder = load_and_prepare_data(DATA_PATH)
    num_labels = len(label_encoder.classes_)
    logging.info(f"Detected {num_labels} categories: {label_encoder.classes_}")

    logging.info(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    train_tok = tokenize_and_encode(train_ds, tokenizer)
    test_tok = tokenize_and_encode(test_ds, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer)

    logging.info("Setting up metrics")
    metric = evaluate.load('accuracy')
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return metric.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy='epoch', # Changed from evaluation_strategy
        save_strategy='epoch',       # Changed from save_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True, # Load the best model (based on metric_for_best_model) at the end of training
        metric_for_best_model='accuracy', # Metric to monitor for best model selection
        logging_dir=f"{OUTPUT_DIR}/logs", # Directory for TensorBoard logs
        logging_steps=100, # Log every 100 steps
        save_total_limit=3, # Keep only the last 3 checkpoints
        report_to="none", # Disable reporting to external services like Weights & Biases
        fp16=True
    )

    logging.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=test_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    logging.info("Beginning training")
    trainer.train()
    logging.info("Training complete")

    logging.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)

    label_map_path = os.path.join(OUTPUT_DIR, 'label_encoder_classes.txt')
    logging.info(f"Saving label mapping to {label_map_path}")
    with open(label_map_path, 'w') as f:
        for cls in label_encoder.classes_:
            f.write(cls + '\n')

    logging.info("Pipeline finished successfully")

if __name__ == '__main__':
    main()
