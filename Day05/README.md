# 🍷 Day 05 — NLP, Dashboards, Pipelines & Model Comparison

![Day](https://img.shields.io/badge/Day-05-crimson?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Wine%20Quality-red?style=flat-square)
![Models](https://img.shields.io/badge/Models-LR%20%7C%20RF%20%7C%20GradientBoosting-blue?style=flat-square)

---

## 🎯 Tasks Completed

- ✅ Build a dataset from unstructured documents (NLP → structured table)
- ✅ Perform sentiment / text analysis on sample data
- ✅ Create automated reports with charts and insights
- ✅ Develop data preprocessing pipelines
- ✅ Compare different machine learning algorithms
- ✅ Generate business insights from raw data
- ✅ Create prediction models using real-world datasets
- ✅ Create a dashboard with visualizations
- ✅ Document the complete workflow and findings

---

## 📂 Dataset
**Red Wine Quality Dataset** — 1599 rows, 12 columns
Target: Classify wine as Good (quality >= 6 = 1) or Bad (quality < 6 = 0)

---

## 🔄 What Was Done

**Preprocessing Pipeline**
- Created binary target: quality >= 6 = Good (855 wines), quality < 6 = Bad (744 wines)
- Built reusable sklearn `Pipeline` with StandardScaler + model for clean workflow

**NLP — Sentiment Analysis**
- Created 12 sample wine reviews inline
- Used TextBlob to compute polarity scores and classify sentiment
- Built structured dataset from raw unstructured text

**Feature Engineering Insight**
- Good wines have significantly higher average alcohol (10.86%) vs bad wines (9.93%)
- Bad wines have higher volatile acidity (0.59) vs good wines (0.474)
- Overall good wine rate: 53.5% — dataset is well balanced

---

## 📈 Visualizations (Dashboard)

| Chart | Type | Insight |
|-------|------|---------|
| Quality Distribution | Bar Chart | Most wines score 5 or 6 — very few reach 8 |
| Alcohol vs Quality | Box Plot | Higher quality wines consistently have higher alcohol content |
| Sentiment Analysis | Bar Chart | TextBlob found 7 Positive and 5 Negative reviews from 12 samples |
| Correlation Heatmap | Heatmap | Alcohol has the strongest positive correlation with wine quality |
| Model Comparison | Bar Chart | All 3 models achieved similar CV accuracy (~0.73) |

---

## 🤖 ML Model Comparison

| Model | Test Accuracy | CV Mean Accuracy |
|-------|--------------|------------------|
| Logistic Regression | 0.74 | 0.73 |
| Random Forest | **0.80** | 0.73 |
| Gradient Boosting | 0.79 | 0.73 |

**Best model:** Random Forest — highest test accuracy (0.80), all models tied on CV (0.73)

---

## 💡 Business Insights

- Good wines average **10.86% alcohol** vs bad wines at **9.93%** — alcohol is the top differentiator
- Good wines have lower volatile acidity (**0.474** vs **0.59**) — less acetic acid = better taste
- **53.5%** of wines in this dataset qualify as good quality
- All 3 ML models performed similarly (CV: 0.73) — the dataset's features have a natural accuracy ceiling around this range

---

## ▶️ How to Run

```bash
pip install textblob
cd Day05
python analysis.py
```

---

> ⬅️ [Day 04](../Day04/README.md) · 🏠 [Main README](../README.md)