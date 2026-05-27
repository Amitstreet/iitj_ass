Goodreads Genre Classification using DistilBERT

A machine learning and NLP project that fine-tunes a DistilBERT model to classify Goodreads book reviews into multiple genres using the Hugging Face Transformers library.

Project Overview

This project performs multi-class text classification on Goodreads reviews.
The model predicts the genre of a book review from the following categories:

Poetry
Children
Comics & Graphic
Fantasy & Paranormal
History & Biography
Mystery / Thriller / Crime
Romance
Young Adult

The project also includes:

TF-IDF + Logistic Regression baseline
DistilBERT fine-tuning
Evaluation metrics (Accuracy + F1 Score)
Confusion matrix visualizations
Weights & Biases experiment tracking
Hugging Face Hub integration
Model Used
distilbert-base-cased

Using:

PyTorch
Hugging Face Transformers
Scikit-learn
Dataset

Dataset source: Goodreads Reviews Dataset by UCSD McAuley Lab

Each genre dataset is downloaded automatically from public URLs.

Features
Automatic dataset downloading
Random sampling and preprocessing
Baseline ML model comparison
Transformer fine-tuning
Evaluation and classification reports
Confusion matrix heatmaps
Hugging Face model upload support
W&B experiment logging
Installation

Clone the repository:

git clone https://github.com/your-username/goodreads-genre-classifier.git
cd goodreads-genre-classifier

Install dependencies:

pip install -r requirements.txt
Required Libraries
transformers
torch
scikit-learn
pandas
numpy
matplotlib
seaborn
wandb
huggingface_hub
requests
Running the Project

Run the training script:

python train_bert_genre_classifier.py
Training Pipeline

The script performs the following steps:

Load Goodreads review datasets
Create train/test split
Train TF-IDF baseline model
Tokenize text using DistilBERT tokenizer
Fine-tune DistilBERT
Evaluate model performance
Generate classification report
Plot confusion matrices
Save trained model
Push model to Hugging Face Hub
Example Results
Metric	Score
Accuracy	0.605
Weighted F1 Score	0.606
Hugging Face Integration

Login using your HF token:

from huggingface_hub import login

login(token="YOUR_HF_TOKEN")

Push model:

model.push_to_hub("your-username/distilbert-goodreads-genres")
tokenizer.push_to_hub("your-username/distilbert-goodreads-genres")
Weights & Biases Tracking

The project logs:

Training loss
Evaluation loss
Accuracy
F1 Score
Artifacts

Initialize W&B:

wandb.init(
    project="mlops-assignment2",
    name="distilbert-run-1"
)
Output Files

Generated during training:

eval_report.json
Confusion matrix heatmaps
Saved DistilBERT model
W&B logs
Future Improvements
Hyperparameter tuning
Larger training dataset
More transformer architectures
Better class balancing
Deployment with FastAPI or Streamlit
Author

Amit Singh

License

This project is for educational and research purposes.
