# 🤖 Day 06 — Advanced ML Pipelines, NLP & Anomaly Detection

![Day](https://img.shields.io/badge/Day-06-darkblue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Wine%20Quality%20+%20Text%20Docs-purple?style=flat-square)
![Models](https://img.shields.io/badge/Models-NaiveBayes%20%7C%20LR%20%7C%20RF%20%7C%20GB%20%7C%20IsolationForest-blue?style=flat-square)

---

## 🎯 Tasks Completed

| # | Task | Status |
|---|------|--------|
| 1 | Build an end-to-end machine learning pipeline | ✅ Done |
| 2 | Create a document classification system | ✅ Done |
| 3 | Develop a text and information extraction solution | ✅ Done |
| 4 | Implement advanced text preprocessing techniques | ✅ Done |
| 5 | Create anomaly detection models | ✅ Done |
| 6 | Build a recommendation engine prototype | ✅ Done |
| 7 | Develop predictive analytics for user behavior | ✅ Done |
| 8 | Compare multiple machine learning approaches | ✅ Done |
| 9 | Generate automated insights from uploaded datasets | ✅ Done |
| 10 | Prepare technical documentation | ✅ Done |

---

## 📂 Datasets Used

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| Red Wine Quality | UCI (copied from Day05) | 1599 rows, 12 cols | ML pipeline, anomaly detection, recommendation |
| Text Documents | Inline (15 samples) | 15 docs, 3 categories | NLP document classification |

---

## 🔤 NLP — Advanced Text Preprocessing (Task 4)

Applied a 4-step text preprocessing pipeline to all 15 documents before classification:

1. **Lowercase** — standardise all text
2. **Punctuation & number removal** — using regex `re.sub(r'[^a-z\s]', '')`
3. **Stopword removal** — filtered common English words using NLTK stopwords
4. **Stemming** — reduced words to root form using Porter Stemmer (e.g. "learning" → "learn", "science" → "scienc")

**Example:**
```
Original : Python is a powerful programming language for data science and machine learning
Cleaned  : python power program languag data scienc machin learn
```

---

## 📄 Document Classification (Task 2 & 3)

**15 sample documents across 3 categories (5 each):** Tech, Sports, Food

Used a TF-IDF + Multinomial Naive Bayes pipeline for classification.

| Metric | Value |
|--------|-------|
| Algorithm | TF-IDF Vectorizer + Multinomial Naive Bayes |
| Train / Test Split | 67% / 33% (10 train, 5 test) |
| Test Accuracy | 0.60 (3 out of 5 test docs correct) |
| Best performing class | Food (precision 1.0, recall 1.0) |

> **Note:** 60% accuracy is expected with only 5 test samples. With such a small dataset, a single misclassification significantly affects the score.

### Top Keywords per Category (TF-IDF)

| Tech | Sports | Food |
|------|--------|------|
| technolog | player | flavor |
| learn | team | dish |
| threat | perform | fresh |
| connect | footbal | celebr |
| grow | final | chocol |

---

## ⚙️ End-to-End Advanced ML Pipeline (Task 1, 7, 8)

Advanced 3-stage sklearn Pipeline: **StandardScaler → SelectKBest (top 8 features) → Model**

### Model Comparison Results

| Model | Test Accuracy | CV Mean Accuracy |
|-------|--------------|-----------------|
| Logistic Regression | 0.73 | **0.74** |
| Random Forest | **0.79** | **0.74** |
| Gradient Boosting | 0.76 | 0.73 |

**Best model:** Logistic Regression and Random Forest tied on CV accuracy (0.74). Random Forest achieved the highest test accuracy (0.79).

---

## 🔴 Anomaly Detection (Task 5)

Used **IsolationForest** (contamination=0.05) on wine features to detect outlier wines.

| Result | Value |
|--------|-------|
| Algorithm | IsolationForest |
| Features Used | alcohol, volatile acidity, sulphates, quality |
| Normal Wines | 1520 (95.1%) |
| Anomaly Wines | 79 (4.9%) |

Anomalies represent wines with unusual combinations of alcohol, acidity, and sulphate levels that don't fit the expected distribution.

---

## 🎯 Recommendation Engine (Task 6)

Used **cosine similarity** on scaled wine features to find the most similar wines.

| Recommendation | Wine # | Similarity Score | Quality | Alcohol |
|---------------|--------|-----------------|---------|---------|
| Reference | #0 | — | 5 | 9.4% |
| Match 1 | #4 | 1.000 | 5 | 9.4% |
| Match 2 | #5 | 0.992 | 5 | 9.4% |
| Match 3 | #28 | 0.979 | 5 | 9.4% |

---

## 💡 Automated Business Insights (Task 9)

```
Total wines analysed : 1,599
Good wine rate       : 53.5%
Anomaly rate         : 4.9%
Avg alcohol (good)   : 10.86%
Avg alcohol (bad)    : 9.93%
Best CV model        : Logistic Regression (0.74)
NLP classifier acc   : 0.60
```

---

## 📈 Visualizations (Dashboard)

| Chart | Type | Insight |
|-------|------|---------|
| Category Distribution | Bar Chart | Equal distribution — 5 docs per category |
| Top Keywords per Category | Horizontal Bar (3 subplots) | TF-IDF clearly separates domain vocabulary |
| Anomaly Detection | Scatter Plot | 79 anomaly wines visible as red dots in outlier regions |
| Pipeline Model Comparison | Bar Chart | LR and RF tied on CV accuracy at 0.74 |
| Recommendation Engine | Bar Chart | Wine #4 found with perfect similarity score of 1.0 |

---

## 📚 Learning Outcomes

- Learned advanced sklearn Pipelines with SelectKBest feature selection
- Learned TF-IDF vectorisation and Naive Bayes for text classification
- Learned NLTK text preprocessing: stopword removal and stemming
- Learned IsolationForest for unsupervised anomaly detection
- Built a content-based recommendation engine using cosine similarity

---

## ▶️ How to Run

```bash
pip install nltk
cd Day06
python analysis.py
```

---

> ⬅️ [Day 05](../Day05/README.md) · 🏠 [Main README](../README.md)