 Goodreads Genre Classification using DistilBERT

A Natural Language Processing (NLP) project that fine-tunes a DistilBERT model to classify Goodreads book reviews into different literary genres using the Hugging Face Transformers library.

 Project Overview

This project performs multi-class text classification on Goodreads reviews.

Given a review, the model predicts its corresponding genre from the following categories:

Poetry
Children
Comics & Graphic Novels
Fantasy & Paranormal
History & Biography
Mystery / Thriller / Crime
Romance
Young Adult

The project also compares a traditional Machine Learning baseline with a Transformer-based deep learning model.

🧠 Model Architecture

This project uses:

DistilBERT (distilbert-base-cased)
PyTorch
Hugging Face Transformers

DistilBERT is a lightweight and faster version of BERT that retains most of BERT’s language understanding capabilities while reducing computation cost.

📂 Dataset

Dataset Source:
Goodreads Review Datasets from the UCSD McAuley Lab.

The datasets are automatically downloaded from public Goodreads review archives.

Each genre contains thousands of reviews used for training and evaluation.


# Links

| Resource | Link |
|---|---|
| Hugging Face Model | https://huggingface.co/Ami12cbdfhdxf/distilbert-goodreads-genres |
| Weights & Biases Dashboard | https://wandb.ai/amitsinghyyj-iit-/mlops-assignment2 |
| Kaggle Notebook | https://www.kaggle.com/code/amitsinghxfxc/iit-as |
| GitHub Repository |https://github.com/Amitstreet/iitj_ass |

---

✨ Features

✅ Goodreads review genre classification
✅ TF-IDF + Logistic Regression baseline
✅ DistilBERT fine-tuning
✅ Accuracy and F1-score evaluation
✅ Confusion matrix visualization
✅ Weights & Biases experiment tracking
✅ Hugging Face Hub integration
✅ Automated dataset downloading
✅ Classification reports generation

🛠️ Tech Stack
Category	Tools / Libraries
Language	Python
Deep Learning	PyTorch
Transformers	Hugging Face Transformers
ML Baseline	Scikit-learn
Visualization	Matplotlib, Seaborn
Experiment Tracking	Weights & Biases
Model Hosting	Hugging Face Hub
📦 Installation

Clone the repository:

git clone https://github.com/Amitstreet/iitj_ass
cd goodreads-genre-classifier

Install dependencies:

pip install -r requirements.txt
📋 Required Libraries
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
▶️ Running the Project

Run the training script:

python train_bert_genre_classifier.py
🔄 Project Workflow

The script performs the following pipeline:

1. Dataset Downloading
Downloads Goodreads review datasets
Caches reviews locally
2. Data Preprocessing
Random review sampling
Train-test splitting
Label encoding
3. Baseline Model
TF-IDF Vectorization
Logistic Regression Classification
4. Transformer Fine-Tuning
Tokenization using DistilBERT tokenizer
Fine-tuning DistilBERT on Goodreads reviews
5. Evaluation
Accuracy Score
Weighted F1 Score
Classification Report
6. Visualization
Confusion Matrix Heatmaps
Misclassification Analysis
7. Model Deployment
Save trained model locally
Push model to Hugging Face Hub
📊 Example Results
Metric	Score
Accuracy	0.605
Weighted F1 Score	0.606

These results were achieved using DistilBERT fine-tuning on Goodreads review data.

📈 Weights & Biases Integration

This project uses Weights & Biases (W&B) for experiment tracking.

Tracked metrics include:

Training Loss
Validation Loss
Accuracy
F1 Score
Evaluation Artifacts

Example initialization:

wandb.init(
    project="mlops-assignment2",
    name="distilbert-run-1"
)
🤗 Hugging Face Hub Integration

Login using your Hugging Face token:

from huggingface_hub import login

login(token="YOUR_HF_TOKEN")

Push model and tokenizer:

model.push_to_hub("your-username/distilbert-goodreads-genres")
tokenizer.push_to_hub("your-username/distilbert-goodreads-genres")
📁 Generated Outputs

The project generates:

eval_report.json
Confusion matrix heatmaps
Fine-tuned DistilBERT model
W&B experiment logs
Classification reports
📌 Future Improvements
Hyperparameter tuning
Larger training datasets
Additional transformer architectures
Better class balancing
FastAPI deployment
Streamlit web app
Docker support
👨‍💻 Author

Amit Singh

📜 License

This project is intended for educational and research purposes.

⭐ Acknowledgements
Goodreads Dataset
UCSD McAuley Lab
Hugging Face
PyTorch
Weights & Biases
