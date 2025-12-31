# Spam Email Classifier (Python, TF-IDF, Logistic Regression)

This project is an end-to-end spam email classification system built using
classical machine learning techniques in Python.

It demonstrates the complete ML workflow:
data preprocessing, feature extraction, model training, evaluation,
model persistence, and command-line inference.

---

## Features

- Text preprocessing using TF-IDF
- Logistic Regression classifier
- Train/test evaluation with accuracy and classification report
- Model persistence using pickle
- Command Line Interface (CLI) for real-time predictions

---

## Model Overview

- **Vectorizer**: TF-IDF
  - Word and bigram features
  - Stopword removal
  - Frequency-based filtering
- **Classifier**: Logistic Regression
- **Labels**:
  - `0` → Spam
  - `1` → Ham (Not Spam)

---

## Project Structure

```
spam-classifier/
├── data/
│   └── mail_data.csv
├── .gitignore
├── predict.py
├── README.md
├── requirements.txt
├── spam_classifier.pkl
└── train.py
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

This script:

- Loads and preprocesses the dataset
- Trains the model using TF-IDF + Logistic Regression
- Evaluates performance on a test set
- Saves the trained model to spam_classifier.pkl

---

## Run CLI Spam Detector

```bash
python predict.py
```

### Example Usage

```
Spam Detector CLI
Type 'exit' to quit.

> You are a winner U have been specially selected to receive £1000 cash
Prediction: Spam mail

> Thanks for your birthday wishes!
Prediction: Ham mail
```

---

## Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

Due to the nature of text data, some borderline messages may be misclassified.
This project focuses on correctness, clarity, and explainability rather than
perfect accuracy.

---

## Technologies Used

- Python
- Pandas
- scikit-learn

---

## Author

Adil Khan

---
