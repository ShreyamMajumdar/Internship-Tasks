import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("Day04/data/heart-disease.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget counts:")
print(df['target'].value_counts())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7,5))
df['target'].value_counts().plot(kind='bar', color=['#E74C3C','#2ECC71'], edgecolor='white')
plt.title('Heart Disease Distribution')
plt.xlabel('Target (0 = No Disease, 1 = Disease)')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart1_target_distribution.png', dpi=150)
plt.show()
print("Chart 1 saved!")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart2_correlation_heatmap.png', dpi=150)
plt.show()
print("Chart 2 saved!")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

models = {
"Logistic Regression": LogisticRegression(max_iter=1000),
"Decision Tree": DecisionTreeClassifier(random_state=42),
"Random Forest": RandomForestClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))
    cv_scores = cross_val_score(model, X, y, cv=5)
    results[name] = {"test_acc": test_acc, "cv_mean": cv_scores.mean()}
    print(f"\n{name}")
    print(f"  Test Accuracy : {round(test_acc, 2)}")
    print(f"  CV Mean Acc   : {round(cv_scores.mean(), 2)}")

plt.figure(figsize=(7,5))
names = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]

plt.bar(names, cv_means, color=['#3498DB','#E67E22','#2ECC71'], edgecolor='white')
plt.title('Model Comparison (Cross-Validation Accuracy)')
plt.xlabel('Model')
plt.ylabel('Average Accuracy (5-Fold CV)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('chart3_model_comparison.png', dpi=150)
plt.show()
print("Chart 3 saved!")

rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\nFeature Importance:")
print(importances)

plt.figure(figsize=(8,6))
importances.plot(kind='barh', color='steelblue', edgecolor='white')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('chart4_feature_importance.png', dpi=150)
plt.show()
print("Chart 4 saved!")

top_features = importances.head(6).index.tolist()
print("\nTop 6 features selected:", top_features)

X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42),
                     param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_sel, y_train)

print("\nBest parameters found:", grid.best_params_)

best_model = grid.best_estimator_
tuned_pred = best_model.predict(X_test_sel)
tuned_acc = accuracy_score(y_test, tuned_pred)

print("Tuned Random Forest Accuracy:", round(tuned_acc, 2))

plt.figure(figsize=(7,5))
labels = ['Default Random Forest', 'Tuned Random Forest\n(Selected Features)']
scores = [results['Random Forest']['test_acc'], tuned_acc]

plt.bar(labels, scores, color=['#95A5A6','#27AE60'], edgecolor='white')
plt.title('Accuracy: Before vs After Optimization')
plt.ylabel('Test Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('chart5_before_after_tuning.png', dpi=150)
plt.show()
print("Chart 5 saved!")