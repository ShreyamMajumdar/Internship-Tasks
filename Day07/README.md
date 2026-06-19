# 🏆 Day 07 — Final Capstone Project: Customer Churn Prediction

![Day](https://img.shields.io/badge/Day-07-gold?style=flat-square)
![Final](https://img.shields.io/badge/Final-Day-red?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Telco%20Customer%20Churn-blue?style=flat-square)
![Models](https://img.shields.io/badge/Models-LR%20%7C%20RF%20%7C%20GB%20%7C%20KNN-purple?style=flat-square)

---

## 🎯 Tasks Completed

| # | Task | Status |
|---|------|--------|
| 1 | Select a real-world dataset and build a complete solution | ✅ Done |
| 2 | Perform data cleaning, feature engineering, and analysis | ✅ Done |
| 3 | Train and evaluate suitable machine learning models | ✅ Done |
| 4 | Create meaningful visualizations and dashboard (7 charts) | ✅ Done |
| 5 | Generate actionable business insights from the data | ✅ Done |
| 6 | Compare model performance and justify selection | ✅ Done |
| 7 | Prepare a technical report (PDF) | ✅ Done |
| 8 | Create project presentation (README) | ✅ Done |
| 9 | Upload source code with proper documentation | ✅ Done |
| 10 | Present final work | ✅ Done |

---

## 📂 Dataset — Telco Customer Churn (IBM)

| Field | Details |
|-------|---------|
| **Source** | IBM / telco-customer-churn-on-icp4d (GitHub) |
| **Rows** | 7,043 customers |
| **Columns** | 21 features |
| **Target** | Churn (Yes = left, No = stayed) |
| **Churn Rate** | 26.5% (1,869 churned out of 7,043) |

---

## 🔄 Data Cleaning & Preprocessing

- Dropped `customerID` — not useful for prediction
- Converted `TotalCharges` from string to numeric (had hidden blank values)
- Filled 11 blank `TotalCharges` values with the column median
- Encoded all 16 categorical columns using LabelEncoder

---

## ⚙️ Feature Engineering

| New Feature | Formula | Insight |
|-------------|---------|---------|
| `avg_monthly_spend` | TotalCharges / (tenure + 1) | Average spend per month — captures true billing pattern |
| `num_services` | Count of 6 add-on services | How many extras the customer subscribes to |

---

## 📈 Visualizations (Dashboard — 7 Charts)

| Chart | Type | Key Insight |
|-------|------|-------------|
| Churn Distribution | Bar Chart | 5,174 stayed vs 1,869 churned — 26.5% churn rate |
| Monthly Charges by Churn | Box Plot | Churned customers pay significantly more ($74.44 avg vs $61.27) |
| Tenure by Churn | Histogram | Churned customers leave early — avg 18 months vs 37.6 for loyal customers |
| Churn by Contract Type | Bar Chart | Month-to-month contracts have the highest churn rate by far |
| Churn by Services | Bar Chart | Customers with fewer add-on services churn more |
| Correlation Heatmap | Heatmap | Tenure and TotalCharges are most negatively correlated with churn |
| Model Comparison | Bar Chart | Logistic Regression achieved the best CV accuracy (0.81) |

---

## 🤖 ML Model Comparison (4 Models)

| Model | Test Accuracy | CV Mean Accuracy |
|-------|--------------|-----------------|
| **Logistic Regression** | **0.80** | **0.81** |
| Gradient Boosting | 0.80 | 0.80 |
| Random Forest | 0.78 | 0.79 |
| KNN | 0.75 | 0.75 |

### Best Model: Logistic Regression (CV: 0.81)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| No Churn | 0.84 | 0.90 | 0.87 |
| Churn | 0.66 | 0.52 | 0.59 |
| **Overall** | — | — | **0.80** |

---

## 💡 Business Insights & Recommendations

| Insight | Finding | Recommendation |
|---------|---------|----------------|
| Tenure matters most | Churned customers avg only 18 months vs 37.6 for loyal ones | Target new customers (<12 months) with retention offers |
| High bills drive churn | Churned customers pay $74.44/month vs $61.27 for loyal | Offer discounts or loyalty pricing to high-bill customers |
| Contract type is critical | Month-to-month customers churn far more | Incentivize customers to switch to annual contracts |
| Fewer services = more churn | Customers with 1-2 services churn more | Bundle more services — increases stickiness |
| Model accuracy | Best model (LR) achieves 81% CV accuracy | Model is ready for deployment in retention campaigns |

---

## 📚 Learning Outcomes (Full 7-Day Summary)

| Day | Skill Learned |
|-----|--------------|
| Day 01 | EDA, data exploration, basic visualizations |
| Day 02 | Data cleaning, handling missing values, Logistic Regression |
| Day 03 | Feature engineering, regression (Linear Regression, Random Forest) |
| Day 04 | Feature selection, hyperparameter tuning with GridSearchCV |
| Day 05 | NLP basics, TextBlob, sklearn Pipelines, Gradient Boosting |
| Day 06 | TF-IDF, Naive Bayes, IsolationForest, cosine similarity |
| Day 07 | End-to-end ML project, KNN, business storytelling, final presentation |

---

## ▶️ How to Run

```bash
cd Day07
python analysis.py
```

---

> ⬅️ [Day 06](../Day06/README.md) · 🏠 [Main README](../README.md)