import streamlit as st
import joblib
import re
import nltk
import spacy
import subprocess
import sys

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# =========================
# DOWNLOAD REQUIRED NLTK RESOURCES (SAFE FOR CLOUD)
# =========================

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("vader_lexicon")

# =========================
# SPAcy MODEL LOADING (OPTION 2 FIX)
# =========================

try:
    nlp = spacy.load("en_core_web_sm")

except OSError:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Oil & Gas Incident Intelligence System",
    page_icon="🛢️",
    layout="centered"
)

# =========================
# SIDEBAR
# =========================

st.sidebar.title("About")

st.sidebar.info(
    """
    NLP Project for Oil & Gas Industry

    Technologies Used:
    - Python
    - Scikit-learn
    - NLTK
    - spaCy
    - Streamlit
    """
)

# =========================
# LOAD ML MODEL
# =========================

data = joblib.load("nlp_pipeline.pkl")

model = data["model"]
vectorizer = data["vectorizer"]

# =========================
# NLP TOOLS
# =========================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# =========================
# PREPROCESSING FUNCTION
# =========================

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

# =========================
# RISK PREDICTION
# =========================

def predict_risk(text):

    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    return prediction

# =========================
# SENTIMENT ANALYSIS
# =========================

def analyze_sentiment(text):

    score = sia.polarity_scores(text)
    compound = score["compound"]

    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# =========================
# NER FUNCTION
# =========================

def extract_entities(text):

    doc = nlp(text)

    return [(ent.text, ent.label_) for ent in doc.ents]

# =========================
# MAIN UI
# =========================

st.title("🛢️ Oil & Gas Incident Intelligence System")

st.markdown("""
This AI-powered NLP application analyzes industrial incident reports using:

- Risk Classification
- Sentiment Analysis
- Named Entity Recognition (NER)

Enter an incident report below to begin analysis.
""")

# =========================
# USER INPUT
# =========================

user_input = st.text_area(
    "Enter Incident Report",
    height=180,
    placeholder="Example: Gas leak detected near offshore platform"
)

# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze Report"):

    if not user_input.strip():
        st.warning("Please enter an incident report.")

    else:

        # ANALYSIS
        risk = predict_risk(user_input)
        sentiment = analyze_sentiment(user_input)
        entities = extract_entities(user_input)

        # =========================
        # RISK DISPLAY
        # =========================

        st.subheader("Risk Prediction")

        if risk == "High":
            st.error(f"⚠️ {risk} Risk")

        elif risk == "Medium":
            st.warning(f"⚠️ {risk} Risk")

        else:
            st.success(f"✅ {risk} Risk")

        # =========================
        # SENTIMENT DISPLAY
        # =========================

        st.subheader("Sentiment Analysis")
        st.info(f"Detected Sentiment: {sentiment}")

        # =========================
        # ENTITY DISPLAY
        # =========================

        st.subheader("Named Entities")

        if entities:
            for entity, label in entities:
                st.write(f"- {entity} ({label})")
        else:
            st.write("No entities detected.")

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("Built with Streamlit, Scikit-learn, NLTK, and spaCy")