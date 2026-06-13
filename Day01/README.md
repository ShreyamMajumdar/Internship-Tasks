# 📊 Day 01 — Exploratory Data Analysis

![Day](https://img.shields.io/badge/Day-01-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Titanic-orange?style=flat-square)

---

## 🎯 Tasks
- ✅ Analyse a CSV dataset
- ✅ Perform data exploration and preprocessing
- ✅ Create 3 meaningful visualizations

---

## 📂 Dataset
**Titanic Passenger Dataset** — 891 rows, 12 columns
Target: Whether a passenger survived (0 = No, 1 = Yes)

---

## 🔄 What Was Done

**Exploration**
Loaded the dataset and checked shape, data types, statistics, and missing values.

**Preprocessing**
- Filled missing `Age` values with median
- Filled missing `Embarked` values with mode
- Dropped `Cabin` column (77% missing)
- Encoded `Sex` column as numbers (male=0, female=1)

---

## 📈 Visualizations

| Chart | Type | Insight |
|-------|------|---------|
| Survival by Gender | Bar Chart | Women survived at ~74% vs men at ~18% |
| Age Distribution | Histogram | Most passengers were aged 20–40 |
| Correlation Heatmap | Heatmap | Sex and Pclass are strongest survival predictors |

---

## ▶️ How to Run

```bash
cd Day01
python analysis.py
```

---

> 🏠 [Main README](../README.md) · ➡️ [Day 02](../Day02/README.md)