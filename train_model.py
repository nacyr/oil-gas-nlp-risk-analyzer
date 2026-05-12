import pandas as pd
import re
import joblib
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("dataset.csv")

# NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for word in tokens:

        if word not in stop_words:

            lemma = lemmatizer.lemmatize(word)

            cleaned_tokens.append(lemma)

    return " ".join(cleaned_tokens)

# Apply preprocessing
df["cleaned"] = df["report"].apply(preprocess)

# Features and labels
X = df["cleaned"]

y = df["risk"]

# Vectorization
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()

model.fit(X_vectorized, y)

# Save pipeline
joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer
    },
    "nlp_pipeline.pkl"
)

print("Model trained and saved successfully!")