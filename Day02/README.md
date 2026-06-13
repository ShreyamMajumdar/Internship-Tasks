# 🧠 Day 02 — Data Cleaning, EDA & Machine Learning

![Day](https://img.shields.io/badge/Day-02-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Diabetes-red?style=flat-square)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-orange?style=flat-square)

---

## 🎯 Tasks
- ✅ Clean and preprocess a real-world dataset
- ✅ Perform detailed EDA
- ✅ Create 5 meaningful visualizations
- ✅ Build a Machine Learning model
- ✅ Document findings and model performance

---

## 📂 Dataset
**Pima Indians Diabetes Dataset** — 768 rows, 9 columns
Target: Whether a patient has diabetes (0 = No, 1 = Yes)

---

## 🔄 What Was Done

**Data Cleaning**
Replaced biologically impossible 0 values in Glucose, BloodPressure, SkinThickness, Insulin, and BMI columns with column medians.

**Feature Engineering**
- Created `AgeGroup` column (Under 30 / 30–45 / 45–60 / Over 60)
- Created `BMICategory` column (Underweight / Normal / Overweight / Obese)

---

## 📈 Visualizations

| Chart | Type | Insight |
|-------|------|---------|
| Outcome Distribution | Bar Chart | 65% no diabetes vs 35% diabetic |
| Glucose by Outcome | Histogram | Diabetic patients have higher glucose levels |
| BMI by Outcome | Box Plot | Diabetic patients have higher BMI on average |
| Diabetes Rate by Age | Bar Chart | Risk increases significantly with age |
| Correlation Heatmap | Heatmap | Glucose is the strongest predictor of diabetes |

---

## 🤖 ML Model

| Detail | Value |
|--------|-------|
| Algorithm | Logistic Regression |
| Features | Glucose, BMI, Age, Insulin, BloodPressure, Pregnancies |
| Train / Test Split | 80% / 20% |
| Accuracy | ~77% |

---

## ▶️ How to Run

```bash
cd Day02
python analysis.py
```

---

> ⬅️ [Day 01](../Day01/README.md) · 🏠 [Main README](../README.md)