import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_device(0)
torch.cuda.empty_cache()

MODEL = "ai4bharat/indic-bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL)


def tokenize_data(texts):
    tokenized_inputs = tokenizer(
        texts, padding="max_length", max_length=512, truncation=True
    )
    return tokenized_inputs


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)

    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def train(args):
    df = pd.read_csv(args.train_data)
    df = df.dropna()

    labels = sorted(list(set(df["Label"])))
    labels_to_ids = {k: v for v, k in enumerate(sorted(labels))}

    X = list(df["Text"])
    y = list(df["Label"])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tokenized = tokenizer(
        X_train, padding=True, truncation=True, max_length=512
    )
    X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels)
    )
    train_dataset = utils.Dataset(X_train_tokenized, y_train, labels_to_ids)
    val_dataset = utils.Dataset(X_val_tokenized, y_val, labels_to_ids)

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=args.epochs,
        seed=0,
        save_steps=500,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    with open(os.path.join(args.output_dir, "labels-dict.pkl"), "wb") as f:
        pickle.dump(labels_to_ids, f)

    train_metrics = trainer.train()
    eval_metrics = trainer.evaluate()

    print(train_metrics)
    print(eval_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train(args)
