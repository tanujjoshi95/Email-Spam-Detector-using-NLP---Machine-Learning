# 📬 Email Spam Detector using NLP & Machine Learning

An end-to-end **NLP project** that classifies messages as **Spam** or **Not Spam** using traditional machine learning techniques on top of text preprocessing and TF–IDF features.

---

## 🧠 Project Overview

This project applies the NLP concepts I learned (text preprocessing, TF–IDF, classical ML models) to build an **Email/SMS Spam Detector**.

The goal is to:
- Clean raw text messages
- Convert them into numerical features
- Train models to classify them as **spam** or **ham (not spam)**
- Provide a simple way to test new messages

---

## 🧰 Tech Stack & Libraries

**Language**
- Python

**Data & Utilities**
- `pandas`
- `numpy`
- `re` (regular expressions)

**NLP**
- `nltk` (stopwords)
- `nltk.stem.PorterStemmer` (stemming)

**Feature Extraction**
- `sklearn.feature_extraction.text.TfidfVectorizer`

**Models**
- `sklearn.naive_bayes.MultinomialNB`

**Model Evaluation**
- `sklearn.model_selection.train_test_split`
- `sklearn.metrics`  
  - `accuracy_score`
  - `classification_report`
  - `confusion_matrix`

**Model Persistence**
- `joblib`

---

## 🧱 Project Structure (example)

```bash
.
├── mail.data.csv              # Dataset (label, text)
├── Model.ipynb    
├── model.joblib         # Saved trained model
├── tfidf_vectorizer.joblib   # Saved TF-IDF vectorizer
├── spam_ham_detector.mp4     # Project demo video
└── README.md                 # This file
