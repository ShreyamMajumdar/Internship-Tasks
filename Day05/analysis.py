import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "wine.csv"), sep=';')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nQuality distribution:")
print(df['quality'].value_counts().sort_index())

df['good_wine'] = (df['quality'] >= 6).astype(int)

print("\nBinary target distribution:")
print(df['good_wine'].value_counts())
print("Good wines:", df['good_wine'].sum(), "| Bad wines:", (df['good_wine'] == 0).sum())

X = df.drop(columns=['quality', 'good_wine'])
y = df['good_wine']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

def make_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

print("\nPreprocessing pipeline ready!")
print("Training samples:", len(X_train), "| Testing samples:", len(X_test))

reviews = [
    "This wine is absolutely fantastic, very smooth and delicious",
    "Terrible taste, very bitter and disappointing quality",
    "Amazing aroma and great flavor, would definitely recommend",
    "Poor quality wine, not worth the price at all",
    "Excellent wine with a beautiful rich taste and finish",
    "Very bad experience, too acidic and way too harsh",
    "Outstanding vintage, perfectly balanced and wonderful",
    "Decent wine but nothing particularly special",
    "Love the fruity notes and the incredibly smooth finish",
    "Worst wine I have ever tried, absolutely disgusting",
    "Rich and complex flavors, truly a great bottle",
    "Flat taste, no character, completely average and boring",
]

# Analyse sentiment using TextBlob
sentiments = []
polarities = []
for review in reviews:
    score = TextBlob(review).sentiment.polarity
    polarities.append(round(score, 3))
    if score > 0:
        sentiments.append('Positive')
    elif score < 0:
        sentiments.append('Negative')
    else:
        sentiments.append('Neutral')

nlp_df = pd.DataFrame({
    'Review': reviews,
    'Polarity': polarities,
    'Sentiment': sentiments
})

print("\nSentiment Analysis Results:")
print(nlp_df[['Polarity', 'Sentiment']])
print("\nSentiment Counts:")
print(nlp_df['Sentiment'].value_counts())

models = {
    "Logistic Regression":  make_pipeline(LogisticRegression(max_iter=1000)),
    "Random Forest":        make_pipeline(RandomForestClassifier(random_state=42)),
    "Gradient Boosting":    make_pipeline(GradientBoostingClassifier(random_state=42)),
}

results = {}
print("\n== Model Comparison ==========================================")
for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    results[name] = {
        "test_acc": round(test_acc, 2),
        "cv_mean":  round(cv_scores.mean(), 2)
    }
    print(f"\n{name}")
    print(f"Test Accuracy : {round(test_acc, 2)}")
    print(f"CV Mean Acc : {round(cv_scores.mean(), 2)}")

best_model = max(results, key=lambda n: results[n]['cv_mean'])
print(f"\nBest model: {best_model} (CV Mean: {results[best_model]['cv_mean']})")

print("\n== Business Insights =========================================")
print("Average alcohol in good wines:", round(df[df['good_wine']==1]['alcohol'].mean(), 2))
print("Average alcohol in bad wines :", round(df[df['good_wine']==0]['alcohol'].mean(), 2))
print("Average volatile acidity (good):", round(df[df['good_wine']==1]['volatile acidity'].mean(), 3))
print("Average volatile acidity (bad) :", round(df[df['good_wine']==0]['volatile acidity'].mean(), 3))
print("Overall good wine rate:", round(df['good_wine'].mean() * 100, 1), "%")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
quality_counts = df['quality'].value_counts().sort_index()
colors = ['#E74C3C' if q < 6 else '#2ECC71' for q in quality_counts.index]
quality_counts.plot(kind='bar', color=colors, edgecolor='white')
plt.title('Wine Quality Score Distribution')
plt.xlabel('Quality Score  (Red = Bad  |  Green = Good)')
plt.ylabel('Number of Wines')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart1_quality_distribution.png'), dpi=150)
plt.show()
print("Chart 1 saved!")

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='quality', y='alcohol',
            hue='quality', palette='RdYlGn', legend=False)
plt.title('Alcohol Level by Wine Quality Score')
plt.xlabel('Quality Score')
plt.ylabel('Alcohol (%)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart2_alcohol_vs_quality.png'), dpi=150)
plt.show()
print("Chart 2 saved!")

plt.figure(figsize=(7, 5))
sentiment_counts = nlp_df['Sentiment'].value_counts()
color_map = {'Positive': '#2ECC71', 'Negative': '#E74C3C', 'Neutral': '#95A5A6'}
bar_colors = [color_map[s] for s in sentiment_counts.index]
bars = plt.bar(sentiment_counts.index, sentiment_counts.values,
               color=bar_colors, edgecolor='white')
for bar, val in zip(bars, sentiment_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.1, str(val),
             ha='center', fontweight='bold')
plt.title('Sentiment Analysis of Wine Reviews (NLP)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart3_sentiment_analysis.png'), dpi=150)
plt.show()
print("Chart 3 saved!")

plt.figure(figsize=(10, 8))
sns.heatmap(df.drop(columns=['good_wine']).corr(),
            annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart4_correlation_heatmap.png'), dpi=150)
plt.show()
print("Chart 4 saved!")

plt.figure(figsize=(7, 5))
names    = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]
bars = plt.bar(names, cv_means,
               color=['#3498DB', '#2ECC71', '#E67E22'], edgecolor='white')
for bar, val in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.005, str(val),
             ha='center', fontweight='bold')
plt.title('ML Model Comparison (5-Fold CV Accuracy)')
plt.ylabel('CV Mean Accuracy')
plt.ylim(0, 1)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart5_model_comparison.png'), dpi=150)
plt.show()
print("Chart 5 saved!")