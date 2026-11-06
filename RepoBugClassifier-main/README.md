# RepoTriage AI

An AI-powered GitHub Issue Triage System that classifies GitHub issues as **Bug** or **Not Bug** using repository-specific machine learning models.

> 🚧 Current Status: Version 1 (Machine Learning & JavaScript specialist models completed)
>
> 🔜 Planned Version 2: Automatic repository detection and intelligent routing to repository-specific expert models.

---

# Motivation

Open-source repositories receive hundreds of GitHub issues every month. These issues include:

- Bug reports
- Feature requests
- Documentation requests
- Questions
- Discussions

Maintainers spend a significant amount of time manually triaging these issues before assigning them to developers.

Automating bug identification can significantly reduce this effort.

---

# Problem Statement

Initially, the goal was to train a **single machine learning model** capable of classifying GitHub issues from multiple repositories into **Bug** or **Not Bug**.

However, during dataset analysis an important observation emerged.

Every repository follows its own issue-labeling conventions and uses highly domain-specific terminology.

For example:

### Machine Learning repositories

- CUDA
- Tensor
- Gradient
- Loss
- Inference
- Training

### JavaScript repositories

- React
- Hook
- Component
- Webpack
- Node
- NPM

Similarly, bug labels varied significantly across repositories.

Examples include:

- bug
- Type: Bug
- type:bug
- confirmed-bug
- module:bug
- bug:beetle

Using one common mapping for every repository produced noisy labels and reduced prediction quality.

---

# Our Observation

Instead of forcing one universal classifier, repository-specific specialist models provide much better performance because they learn repository-specific terminology and issue patterns.

This observation led to the current project architecture.

---

# Current Solution (Version 1)

The current implementation trains **independent specialist models** for each repository.

Currently implemented:

- Machine Learning repositories
- JavaScript repositories

Each repository has its own:

- preprocessing pipeline
- TF-IDF vectorizer
- XGBoost classifier

During inference, the user selects the repository and the corresponding specialist model performs bug classification.

---

# Dataset Preparation

The dataset was collected from GitHub Issues.

Each issue contains:

- Repository
- Title
- Body
- User Type
- Author Association
- Labels

The target variable was not directly available.

Therefore, repository-specific bug labels were manually analyzed and converted into a binary target.

```
Bug -> 1
Not Bug -> 0
```

---

# Text Preprocessing

Each issue is represented as:

```
Title + Body
```

The preprocessing pipeline performs:

- Emoji removal
- HTML removal
- Lowercasing
- Stopword removal
- Punctuation removal
- Lemmatization
- Whitespace normalization

SpaCy was used for linguistic preprocessing.

---

# Feature Engineering

Multiple feature extraction methods were evaluated.

## TF-IDF

- max_features = 10000
- ngram_range = (1,2)

## SBERT

Sentence Transformer embeddings were also evaluated.

---

# Model Comparison

Several combinations were experimentally evaluated.

### Vectorizers

- TF-IDF
- SBERT

### Classifiers

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Performance was compared using:

- Accuracy
- Precision
- Recall
- F1 Score

---

# Final Model

The best performing pipeline was:

```
TF-IDF
        ↓
XGBoost
```

The final model provided the best balance between precision and recall while maintaining strong bug detection capability.

Since missing a genuine bug is costly, recall was treated as an important evaluation metric during model selection.

---

# Current Architecture

```
User

↓

Select Repository

↓

Preprocess Text

↓

Repository-specific TF-IDF

↓

Repository-specific XGBoost

↓

Bug / Not Bug
```

---

# Project Structure

```
repo-triage-ai/

│

├── models/
│   ├── ML_tfidf.pkl
│   ├── ML_xgb_tfidf.pkl
│   ├── javascript_tfidf.pkl
│   └── javascript_xgb_tfidf.pkl
│
├── preprocessing.py
├── app.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SpaCy
- Joblib
- FastAPI
- Streamlit

---

# Current Progress

✅ Dataset collection

✅ Repository-specific target engineering

✅ Text preprocessing pipeline

✅ Feature engineering

✅ TF-IDF implementation

✅ SBERT comparison

✅ Model comparison

✅ Repository-specific XGBoost models

✅ Model serialization (.pkl)

🚧 FastAPI deployment

🚧 Streamlit frontend

---

# Future Work (Version 2)

The current system requires the user to manually choose the repository.

The next version will eliminate this requirement by introducing a **Repository Router**.

Proposed architecture:

```
GitHub Issue

↓

Repository Router

↓

Machine Learning Expert

JavaScript Expert

Future Repository Experts...

↓

Bug / Not Bug
```

Instead of asking the user to select the repository, a lightweight routing model will automatically identify the repository domain and forward the issue to the appropriate specialist model.

This architecture is scalable and allows new repository specialists to be added without modifying the existing expert models.

---

