# Day 01 - AI/Data Science Task
**Name:** Shreyam Majumdar  
**Date:** 12-06-2026

## Dataset Used
Titanic dataset — 891 rows, 12 columns.  
Contains passenger info: age, gender, class, fare, and survival status.

## Tools Used
- Python 3
- pandas, numpy, matplotlib, seaborn

## Steps Performed

### 1. Data Exploration
- Loaded CSV using pandas
- Checked shape, data types, missing values
- Used df.describe() for basic statistics

### 2. Preprocessing
- Filled missing Age values with median
- Filled missing Embarked with mode
- Dropped Cabin column (77% missing)
- Encoded Sex column as numbers

### 3. Visualizations

**Chart 1 — Survival by Gender (Bar Chart)**  
Insight: Women survived at a much higher rate than men (~74% vs ~18%).

**Chart 2 — Age Distribution (Histogram)**  
Insight: Most passengers were between 20–40 years old.

**Chart 3 — Correlation Heatmap**  
Insight: Sex and Pclass are the strongest predictors of survival.