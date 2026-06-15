# 📈 Day 03 — Feature Engineering & Model Comparison

![Day](https://img.shields.io/badge/Day-03-teal?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Auto%20MPG-yellow?style=flat-square)
![Model](https://img.shields.io/badge/Models-Linear%20Regression%20vs%20Random%20Forest-orange?style=flat-square)

---

## 🎯 Tasks
- ✅ Select a dataset and perform complete preprocessing
- ✅ Train at least two machine learning models
- ✅ Compare model performance using evaluation metrics
- ✅ Create a dashboard with visualizations
- ✅ Document observations and conclusions

---

## 📂 Dataset
**Auto MPG Dataset** — 398 rows, 9 columns
Target: Predict a car's fuel efficiency in miles per gallon (Regression)

---

## 🔄 What Was Done

**Preprocessing**
- Filled missing `horsepower` values with median
- Dropped `name` column (too many unique values)
- Encoded `origin` column as numbers (usa=0, europe=1, japan=2)

**Feature Engineering**
- Created `power_to_weight` ratio (horsepower / weight)

---

## 📈 Visualizations (Dashboard)

| Chart | Type | Insight |
|-------|------|---------|
| MPG Distribution | Histogram | Most cars average around 23.5 mpg |
| Weight vs MPG | Scatter Plot | Strong negative relationship — heavier cars are less efficient |
| MPG by Origin | Box Plot | Japanese/European cars are more fuel-efficient than US cars |
| MPG by Cylinders | Bar Chart | Fewer cylinders means better fuel efficiency |
| Correlation Heatmap | Heatmap | Weight is the strongest predictor of mpg |

---

## 🤖 Model Comparison

| Model | MAE | RMSE | R2 |
|-------|-----|------|-----|
| Linear Regression | 2.26 | 2.86 | 0.85 |
| Random Forest Regressor | 1.58 | 2.14 | 0.91 |

**Best model:** Random Forest Regressor — lower error and higher R2, explaining 91% of the variance in mpg.

---

## ▶️ How to Run

```bash
cd Day03
python analysis.py
```

---

> ⬅️ [Day 02](../Day02/README.md) · 🏠 [Main README](../README.md) · ➡️ [Day 04](../Day04/README.md)