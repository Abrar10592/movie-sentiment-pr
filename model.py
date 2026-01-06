import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# NLTK setup
nltk.download('stopwords')
stop_words = stopwords.words('english')

def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 1. Load data
df = pd.read_csv('data/IMDB Dataset.csv')   # columns: review, sentiment
df['review'] = df['review'].apply(preprocess_text)

X = df['review']
y = df['sentiment']   # 'positive' / 'negative'

# 2. Train/val/test split  (e.g. 70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 3. Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=1000))
])

# 4. Train on TRAIN only
pipeline.fit(X_train, y_train)

# 5. Evaluate on VAL set (optional – for tuning)
y_val_pred = pipeline.predict(X_val)
print("\nValidation performance:")
print("Accuracy :", accuracy_score(y_val, y_val_pred))
print("Precision:", precision_score(y_val, y_val_pred, pos_label='positive'))
print("Recall   :", recall_score(y_val, y_val_pred, pos_label='positive'))
print("F1-score :", f1_score(y_val, y_val_pred, pos_label='positive'))

# 6. Final evaluation on TEST set
y_test_pred = pipeline.predict(X_test)
print("\nTest performance:")
print("Accuracy :", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred, pos_label='positive'))
print("Recall   :", recall_score(y_test, y_test_pred, pos_label='positive'))
print("F1-score :", f1_score(y_test, y_test_pred, pos_label='positive'))

print("\nClassification report (test):")
print(classification_report(y_test, y_test_pred))

# 7. Confusion matrix
cm = confusion_matrix(y_test, y_test_pred, labels=['negative', 'positive'])
print("\nConfusion matrix (rows=true, cols=pred):")
print(pd.DataFrame(cm,
                   index=['true_negative', 'true_positive'],
                   columns=['pred_negative', 'pred_positive']))

# 8. Save model trained on FULL data (optional but common for deployment)
pipeline.fit(X, y)
joblib.dump(pipeline, 'model.pkl')
print("\nModel retrained on full data and saved to model.pkl")
