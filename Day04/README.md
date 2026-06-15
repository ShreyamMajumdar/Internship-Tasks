# 🔍 Day 04 — Model Comparison, Feature Selection & Tuning

![Day](https://img.shields.io/badge/Day-04-indigo?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Heart%20Disease-red?style=flat-square)
![Models](https://img.shields.io/badge/Models-Logistic%20%7C%20Decision%20Tree%20%7C%20Random%20Forest-blue?style=flat-square)

---

## 🎯 Tasks
- ✅ Train and compare multiple machine learning models
- ✅ Perform feature selection on a dataset
- ✅ Improve model accuracy through parameter tuning
- ✅ Create visual reports and insights
- ✅ Prepare project documentation

---

## 📂 Dataset
**Heart Disease Dataset (UCI)** — 303 rows, 14 columns
Target: Whether a patient has heart disease (0 = No, 1 = Yes)

---

## 🔄 What Was Done

**Exploratory Data Analysis**
Checked target distribution and correlation between all features — dataset was already clean with no missing values.

**Model Comparison (5-Fold Cross Validation)**
Trained and compared 3 models: Logistic Regression, Decision Tree, and Random Forest.

**Feature Selection**
Used Random Forest feature importance to identify and select the top 6 most predictive features.

**Hyperparameter Tuning**
Used GridSearchCV to search for the best Random Forest parameters on the selected features.

---

## 📈 Visualizations

| Chart | Type | Insight |
|-------|------|---------|
| Target Distribution | Bar Chart | Dataset is fairly balanced between disease and no-disease cases |
| Correlation Heatmap | Heatmap | `cp`, `thalach`, `exang`, and `oldpeak` show the strongest correlation with target |
| Model Comparison | Bar Chart | Random Forest and Logistic Regression performed best (~0.83-0.84 CV accuracy) |
| Feature Importance | Horizontal Bar Chart | `cp` (chest pain type) is the single most important feature |
| Before vs After Tuning | Bar Chart | Tuning on selected features slightly reduced accuracy (0.84 → 0.79) |

---

## 🤖 Model Comparison Results

| Model | Test Accuracy | CV Mean Accuracy |
|-------|---------------|-------------------|
| Logistic Regression | 0.80 | 0.83 |
| Decision Tree | 0.70 | 0.76 |
| Random Forest | **0.84** | **0.84** |

---

## 🔬 Feature Selection

**Top 6 features (by Random Forest importance):**
`cp`, `thalach`, `oldpeak`, `thal`, `chol`, `ca`

| Feature | Importance |
|---------|------------|
| cp | 0.157 |
| thalach | 0.117 |
| oldpeak | 0.113 |
| thal | 0.107 |
| chol | 0.087 |
| ca | 0.086 |

---

## ⚙️ Hyperparameter Tuning

| Detail | Value |
|--------|-------|
| Method | GridSearchCV (5-fold CV) |
| Best Parameters | `max_depth=5, min_samples_split=5, n_estimators=50` |
| Accuracy — Default RF (all features) | 0.84 |
| Accuracy — Tuned RF (top 6 features) | 0.79 |

---

## 🔍 Key Findings

- Random Forest was the best overall model (0.84 test accuracy, 0.84 CV accuracy)
- `cp` (chest pain type) is the strongest predictor of heart disease
- Logistic Regression performed almost as well as Random Forest (0.83 CV accuracy) despite being a much simpler model
- Decision Tree alone performed the worst (0.76 CV accuracy) — likely overfitting on the full feature set
- **Interesting result:** reducing to the top 6 features and tuning actually lowered accuracy (0.84 → 0.79) — showing that feature selection and tuning don't always improve performance, and the full feature set with default settings worked best here

---

## 📚 Learning Outcomes

- Learned the difference between supervised and unsupervised learning
- Learned to compare multiple models using cross-validation
- Learned how to interpret feature importance from Random Forest
- Learned hyperparameter tuning with GridSearchCV
- Learned that more "optimization" doesn't always mean better results — testing and comparing is essential

---

## ▶️ How to Run

```bash
cd Day04
python analysis.py
```

---

> ⬅️ [Day 03](../Day03/README.md) · 🏠 [Main README](../README.md)