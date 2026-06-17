import os
import re
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "wine.csv"), sep=';')
df['good_wine'] = (df['quality'] >= 6).astype(int)
print("Dataset loaded:", df.shape)
print("Good wines:", df['good_wine'].sum(), "| Bad wines:", (df['good_wine']==0).sum())

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

documents = [
    "Python is a powerful programming language for data science and machine learning",
    "Artificial intelligence and deep learning are transforming modern technology",
    "Cloud computing enables scalable and flexible infrastructure for developers",
    "Cybersecurity threats are growing as more devices connect to the internet",
    "Blockchain technology provides decentralized and secure transaction systems",

    "The football team won the championship after an incredible final performance",
    "Basketball players need both speed and strength to dominate the game",
    "Cricket is one of the most popular sports in South Asia with millions of fans",
    "The tennis player won the grand slam after five sets of intense competition",
    "Swimming requires excellent technique and endurance to compete at elite level",

    "Italian pasta dishes are loved worldwide for their rich and authentic flavors",
    "Sushi is a traditional Japanese food made with seasoned rice and fresh fish",
    "Chocolate cake with cream frosting is a classic dessert for celebrations",
    "Indian curry spices create incredibly flavorful and aromatic dishes",
    "Fresh salads with vegetables and olive oil are perfect for healthy eating",
]
labels      = [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2]
label_names = ['Tech', 'Sports', 'Food']

cleaned_docs = [preprocess_text(d) for d in documents]
print("\nSample preprocessed text:")
print("Original :", documents[0])
print("Cleaned :", cleaned_docs[0])

nlp_pipeline = Pipeline([('tfidf',  TfidfVectorizer()), ('model',  MultinomialNB())])

X_train_nlp, X_test_nlp, y_train_nlp, y_test_nlp = train_test_split( cleaned_docs, labels, test_size=0.33, random_state=42)

nlp_pipeline.fit(X_train_nlp, y_train_nlp)
nlp_pred = nlp_pipeline.predict(X_test_nlp)
nlp_acc = accuracy_score(y_test_nlp, nlp_pred)

print("\nDocument Classification Results (TF-IDF + Naive Bayes):")
print("Accuracy:", round(nlp_acc, 2))
print(classification_report(y_test_nlp, nlp_pred, target_names=label_names, zero_division=0))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(cleaned_docs)
feature_names = vectorizer.get_feature_names_out()

print("\nTop Keywords per Category:")
for cat_idx, cat_name in enumerate(label_names):
    cat_docs = [cleaned_docs[i] for i, l in enumerate(labels) if l == cat_idx]
    cat_tfidf = vectorizer.transform(cat_docs).toarray().mean(axis=0)
    top_idx   = cat_tfidf.argsort()[::-1][:5]
    keywords  = [feature_names[i] for i in top_idx]
    print(f"  {cat_name}: {', '.join(keywords)}")

X = df.drop(columns=['quality', 'good_wine'])
y = df['good_wine']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def advanced_pipeline(model, k=8):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=k)),
        ('model', model)
    ])

models = {
    "Logistic Regression": advanced_pipeline(LogisticRegression(max_iter=1000)),
    "Random Forest": advanced_pipeline(RandomForestClassifier(random_state=42)),
    "Gradient Boosting": advanced_pipeline(GradientBoostingClassifier(random_state=42)),
}

results = {}
print("\n== End-to-End Pipeline Model Comparison ====================")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    test_acc  = accuracy_score(y_test, pipe.predict(X_test))
    cv_scores = cross_val_score(pipe, X, y, cv=5)
    results[name] = {"test_acc": round(test_acc, 2), "cv_mean": round(cv_scores.mean(), 2)}
    print(f"\n{name}")
    print(f" Test Accuracy : {round(test_acc, 2)}")
    print(f" CV Mean Acc : {round(cv_scores.mean(), 2)}")

best = max(results, key=lambda n: results[n]['cv_mean'])
print(f"\nBest model: {best} (CV Mean: {results[best]['cv_mean']})")

X_anomaly = df[['alcohol', 'volatile acidity', 'sulphates', 'quality']].copy()

iso = IsolationForest(contamination=0.05, random_state=42)
df['anomaly'] = iso.fit_predict(X_anomaly)
df['anomaly_label'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

print("\nAnomaly Detection Results:")
print(df['anomaly_label'].value_counts())
anomaly_count = (df['anomaly'] == -1).sum()
print(f"Anomalies detected: {anomaly_count} out of {len(df)} wines ({round(anomaly_count/len(df)*100,1)}%)")

scaler_rec  = StandardScaler()
wine_feats  = df.drop(columns=['quality', 'good_wine', 'anomaly', 'anomaly_label']).values
wine_scaled = scaler_rec.fit_transform(wine_feats)

def recommend_wines(wine_idx, n=3):
    sims    = cosine_similarity([wine_scaled[wine_idx]], wine_scaled)[0]
    top_idx = sims.argsort()[::-1][1:n+1]
    return top_idx, sims[top_idx]

sample_idx = 0
sim_idx, sim_scores = recommend_wines(sample_idx)
print(f"\nRecommendation Engine: Wines similar to Wine #{sample_idx}")
print(f"  Reference Wine: Quality={df['quality'].iloc[sample_idx]}, Alcohol={df['alcohol'].iloc[sample_idx]}")
for i, (idx, score) in enumerate(zip(sim_idx, sim_scores), 1):
    print(f"{i}. Wine #{idx} | Similarity: {round(score,3)} | Quality: {df['quality'].iloc[idx]} | Alcohol: {df['alcohol'].iloc[idx]}")

print("\n== Automated Business Insights ==============================")
print(f"Total wines analysed : {len(df)}")
print(f"Good wine rate : {round(df['good_wine'].mean()*100, 1)}%")
print(f"Anomaly rate : {round((df['anomaly']==-1).mean()*100, 1)}%")
print(f"Avg alcohol (good) : {round(df[df['good_wine']==1]['alcohol'].mean(), 2)}%")
print(f"Avg alcohol (bad) : {round(df[df['good_wine']==0]['alcohol'].mean(), 2)}%")
print(f"Best CV model : {best} ({results[best]['cv_mean']} CV accuracy)")
print(f"NLP classifier acc : {round(nlp_acc, 2)}")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
cat_counts = pd.Series(labels).value_counts().sort_index()
plt.bar(label_names, cat_counts.values,
        color=['#3498DB', '#2ECC71', '#E67E22'], edgecolor='white')
plt.title('Document Classification — Category Distribution')
plt.xlabel('Category')
plt.ylabel('Number of Documents')
for i, v in enumerate(cat_counts.values):
    plt.text(i, v + 0.05, str(v), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart1_category_distribution.png'), dpi=150)
plt.show()
print("Chart 1 saved!")

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
colors = ['#3498DB', '#2ECC71', '#E67E22']
for cat_idx, (cat_name, ax, color) in enumerate(zip(label_names, axes, colors)):
    cat_docs  = [cleaned_docs[i] for i, l in enumerate(labels) if l == cat_idx]
    cat_tfidf = vectorizer.transform(cat_docs).toarray().mean(axis=0)
    top_idx   = cat_tfidf.argsort()[::-1][:6]
    top_words = [feature_names[i] for i in top_idx]
    top_vals  = [cat_tfidf[i] for i in top_idx]
    ax.barh(top_words[::-1], top_vals[::-1], color=color, edgecolor='white')
    ax.set_title(f'{cat_name} — Top Keywords')
    ax.set_xlabel('TF-IDF Score')
plt.suptitle('Top Keywords per Category (TF-IDF)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart2_top_keywords.png'), dpi=150)
plt.show()
print("Chart 2 saved!")

plt.figure(figsize=(8, 5))
colors_map = {'Normal': '#2ECC71', 'Anomaly': '#E74C3C'}
for label, group in df.groupby('anomaly_label'):
    plt.scatter(group['alcohol'], group['volatile acidity'],
                c=colors_map[label], label=label, alpha=0.5, s=20)
plt.title('Anomaly Detection — Alcohol vs Volatile Acidity')
plt.xlabel('Alcohol (%)')
plt.ylabel('Volatile Acidity')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart3_anomaly_detection.png'), dpi=150)
plt.show()
print("Chart 3 saved!")

plt.figure(figsize=(7, 5))
names    = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]
bars = plt.bar(names, cv_means,
               color=['#3498DB', '#2ECC71', '#E67E22'], edgecolor='white')
for bar, val in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005, str(val),
             ha='center', fontweight='bold')
plt.title('Advanced Pipeline — Model Comparison (5-Fold CV)')
plt.ylabel('CV Mean Accuracy')
plt.ylim(0, 1)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart4_pipeline_comparison.png'), dpi=150)
plt.show()
print("Chart 4 saved!")

plt.figure(figsize=(7, 5))
rec_labels = [f"Wine #{i}" for i in sim_idx]
plt.bar(rec_labels, sim_scores,
        color=['#9B59B6', '#1ABC9C', '#F39C12'], edgecolor='white')
for i, (label, val) in enumerate(zip(rec_labels, sim_scores)):
    plt.text(i, val + 0.001, str(round(val, 3)), ha='center', fontweight='bold')
plt.title(f'Recommendation Engine — Wines Similar to Wine #{sample_idx}')
plt.xlabel('Recommended Wine')
plt.ylabel('Cosine Similarity Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart5_recommendations.png'), dpi=150)
plt.show()
print("Chart 5 saved!")