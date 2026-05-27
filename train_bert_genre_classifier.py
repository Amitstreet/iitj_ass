"""
Training and Fine-Tuning BERT for Classification
Classifying Goodreads Reviews By Book Genre

By Maria Antoniak, Melanie Walsh, and the AI for Humanists Team
Updated: 2024-11-05

This script fine-tunes a DistilBERT model on Goodreads reviews
to predict book genre (poetry, comics, fantasy, history, mystery, romance, young adult).
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import json
import gzip
import random
import pickle
from collections import defaultdict

# ── Third-party ───────────────────────────────────────────────────────────────
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

import wandb
from huggingface_hub import login, whoami

sns.set(style="ticks", font_scale=1.2)


# ── Parameters ────────────────────────────────────────────────────────────────
MODEL_NAME              = "distilbert-base-cased"
DEVICE_NAME             = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH              = 512
CACHED_MODEL_DIR        = "distilbert-reviews-genres"
REVIEWS_PICKLE          = "genre_reviews_dict.pickle"
HF_REPO                 = "Ami12cbdfhdxf/distilbert-goodreads-genres"
WANDB_PROJECT           = "mlops-assignment2"
WANDB_RUN_NAME          = "distilbert-run-1"

GENRE_URL_DICT = {
    "poetry":                 "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children":               "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic":         "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal":     "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography":      "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance":                "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult":            "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ── Dataset ───────────────────────────────────────────────────────────────────
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_reviews(url: str, head: int = 10_000, sample_size: int = 2_000) -> list[str]:
    """Stream reviews from a gzipped JSON-lines URL and return a random sample."""
    reviews = []
    response = requests.get(url, stream=True)
    print(f"  HTTP {response.status_code}")
    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            reviews.append(d["review_text"])
            if head is not None and i + 1 >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))


def get_genre_reviews(pickle_path: str = REVIEWS_PICKLE) -> dict:
    """Load reviews from cache if available, otherwise download and cache."""
    if os.path.exists(pickle_path):
        print(f"Loading cached reviews from {pickle_path}")
        return pickle.load(open(pickle_path, "rb"))

    genre_reviews_dict = {}
    for genre, url in GENRE_URL_DICT.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews_dict[genre] = load_reviews(url, head=10_000, sample_size=2_000)

    pickle.dump(genre_reviews_dict, open(pickle_path, "wb"))
    print(f"Saved reviews to {pickle_path}")
    return genre_reviews_dict


# ── Train / test split ────────────────────────────────────────────────────────
def split_data(genre_reviews_dict: dict, per_genre: int = 1000, train_frac: float = 0.8):
    train_texts, train_labels = [], []
    test_texts,  test_labels  = [], []

    split_idx = int(per_genre * train_frac)  # e.g. 800

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(per_genre, len(reviews)))
        for review in sampled[:split_idx]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[split_idx:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(
        f"Train: {len(train_texts)} texts / {len(train_labels)} labels  |  "
        f"Test: {len(test_texts)} texts / {len(test_labels)} labels"
    )
    return train_texts, train_labels, test_texts, test_labels


# ── Baseline model ────────────────────────────────────────────────────────────
def run_baseline(train_texts, train_labels, test_texts, test_labels):
    print("\n── TF-IDF + Logistic Regression baseline ──")
    vec   = TfidfVectorizer()
    X_tr  = vec.fit_transform(train_texts)
    X_te  = vec.transform(test_texts)
    model = LogisticRegression(max_iter=1_000).fit(X_tr, train_labels)
    preds = model.predict(X_te)
    print(classification_report(test_labels, preds))


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


# ── Visualisation ─────────────────────────────────────────────────────────────
def plot_confusion_heatmap(test_labels, predicted_labels, remove_diagonal: bool = False,
                           title: str = "Genre Classifications"):
    genre_classifications_dict = defaultdict(int)
    for true, pred in zip(test_labels, predicted_labels):
        if remove_diagonal and true == pred:
            continue
        genre_classifications_dict[(true, pred)] += 1

    rows = [
        {"True Genre": t, "Predicted Genre": p, "Number of Classifications": c}
        for (t, p), c in genre_classifications_dict.items()
    ]
    df_wide = (
        pd.DataFrame(rows)
        .pivot_table(index="True Genre", columns="Predicted Genre",
                     values="Number of Classifications")
    )

    plt.figure(figsize=(9, 7))
    sns.heatmap(df_wide, linewidths=1, cmap="Purples")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fname = title.lower().replace(" ", "_") + ".png"
    plt.savefig(fname, dpi=150)
    print(f"Saved heatmap → {fname}")
    plt.show()


# ── Secrets helper (Kaggle) ───────────────────────────────────────────────────
def load_secrets_from_kaggle():
    """Load WANDB_API_KEY and HF_TOKEN from Kaggle Secrets (no-op outside Kaggle)."""
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
        os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
        print("Secrets loaded from Kaggle.")
    except Exception:
        print("Not running on Kaggle — skipping secret loading.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 0. Secrets
    load_secrets_from_kaggle()

    # 1. Data
    genre_reviews_dict = get_genre_reviews()

    # Quick preview
    for genre, reviews in genre_reviews_dict.items():
        print(genre, "|", random.sample(reviews, 1)[0][:80], "...")

    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews_dict)

    # 2. Baseline
    run_baseline(train_texts, train_labels, test_texts, test_labels)

    # 3. Label maps
    unique_labels   = sorted(set(train_labels))
    label2id        = {label: idx for idx, label in enumerate(unique_labels)}
    id2label        = {idx: label for label, idx in label2id.items()}

    # 4. Tokenise
    print("\n── Tokenising ──")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
    test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=MAX_LENGTH)

    train_labels_encoded = [label2id[y] for y in train_labels]
    test_labels_encoded  = [label2id[y] for y in test_labels]

    train_dataset = MyDataset(train_encodings, train_labels_encoded)
    test_dataset  = MyDataset(test_encodings,  test_labels_encoded)

    # 5. Load model
    print("\n── Loading pre-trained DistilBERT ──")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label)
    ).to(DEVICE_NAME)

    # 6. W&B init
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model":         MODEL_NAME,
            "epochs":        3,
            "batch_size":    8,
            "learning_rate": 3e-5,
            "max_length":    MAX_LENGTH,
            "dataset":       "UCSD Goodreads",
        },
    )

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir                = "./results",
        num_train_epochs          = 3,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        warmup_steps              = 100,
        weight_decay              = 0.01,
        logging_steps             = 50,
        eval_strategy             = "epoch",
        save_strategy             = "epoch",
        load_best_model_at_end    = True,
        logging_dir               = "./logs",
        report_to                 = "wandb",
        run_name                  = WANDB_RUN_NAME,
    )

    # 8. Fine-tune
    print("\n── Fine-tuning ──")
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = test_dataset,
        compute_metrics = compute_metrics,
    )
    trainer.train()

    # 9. Save model
    trainer.save_model(CACHED_MODEL_DIR)
    print(f"Model saved to {CACHED_MODEL_DIR}")

    # 10. Evaluate
    print("\n── Evaluation ──")
    eval_results = trainer.evaluate()
    print(eval_results)

    wandb.log({
        "final/loss":     eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1":       eval_results["eval_f1"],
    })

    # 11. Predictions & classification report
    predicted_results  = trainer.predict(test_dataset)
    predicted_labels   = predicted_results.predictions.argmax(-1).flatten().tolist()
    predicted_labels   = [id2label[l] for l in predicted_labels]

    print(classification_report(test_labels, predicted_labels))

    # 12. Save eval report
    preds_int = predicted_results.predictions.argmax(-1)
    true_int  = [item["labels"].item() for item in test_dataset]
    report    = classification_report(
        true_int, preds_int,
        target_names=list(id2label.values()),
        output_dict=True,
    )
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)

    # 13. Visualise
    print("\n── Plotting confusion heatmaps ──")
    plot_confusion_heatmap(test_labels, predicted_labels,
                           remove_diagonal=False, title="All Genre Classifications")
    plot_confusion_heatmap(test_labels, predicted_labels,
                           remove_diagonal=True,  title="Misclassifications Only")

    # Print sample correct / incorrect predictions
    print("\nSample correct predictions:")
    for true, pred, text in random.sample(list(zip(test_labels, predicted_labels, test_texts)), 20):
        if true == pred:
            print(f"  LABEL: {true}\n  REVIEW: {text[:100]}...\n")

    print("\nSample misclassifications:")
    for true, pred, text in random.sample(list(zip(test_labels, predicted_labels, test_texts)), 20):
        if true != pred:
            print(f"  TRUE: {true}  |  PREDICTED: {pred}\n  REVIEW: {text[:100]}...\n")

    # 14. Push to HuggingFace Hub
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print(whoami())
        model.push_to_hub(HF_REPO)
        tokenizer.push_to_hub(HF_REPO)
        wandb.run.summary["huggingface_model"] = f"https://huggingface.co/{HF_REPO}"
    else:
        print("HF_TOKEN not set — skipping Hub push.")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
