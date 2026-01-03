import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def preprocess_text(text):
    # Simple cleaning
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load data and train model
df = pd.read_csv('data/IMDB Dataset.csv')
df['review'] = df['review'].apply(preprocess_text)

# Create pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression())
])

pipeline.fit(df['review'], df['sentiment'])

# Save model
joblib.dump(pipeline, 'model.pkl')
print("Model trained and saved!")
