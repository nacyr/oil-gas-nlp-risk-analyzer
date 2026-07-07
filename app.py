import streamlit as st
import joblib
import re
import nltk
import spacy
import subprocess
import sys

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# ==========================================================
# SAFE NLTK DOWNLOADS (STREAMLIT CLOUD FRIENDLY)
# ==========================================================

REQUIRED_NLTK = [
    "stopwords",
    "wordnet",
    "omw-1.4",
    "vader_lexicon",
]

for resource in REQUIRED_NLTK:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ==========================================================
# LOAD SPACY MODEL
# ==========================================================

try:
    nlp = spacy.load("en_core_web_sm")

except OSError:
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=False
    )
    nlp = spacy.load("en_core_web_sm")

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Oil & Gas Incident Intelligence System",
    page_icon="🛢️",
    layout="centered",
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("About")

st.sidebar.info(
    """
    NLP Project for Oil & Gas Industry

    Technologies Used

    • Python
    • Scikit-learn
    • NLTK
    • spaCy
    • Streamlit
    """
)

# ==========================================================
# LOAD MODEL
# ==========================================================

data = joblib.load("nlp_pipeline.pkl")

model = data["model"]
vectorizer = data["vectorizer"]

# ==========================================================
# NLP TOOLS
# ==========================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# ==========================================================
# PREPROCESS FUNCTION
# ==========================================================

def preprocess(text: str) -> str:
    """
    Clean text without using NLTK word_tokenize().
    This avoids the punkt/punkt_tab LookupError on Streamlit Cloud.
    """

    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Regex tokenizer
    tokens = re.findall(r"\b[a-z]+\b", text)

    cleaned_tokens = []

    for token in tokens:
        if token not in stop_words:
            cleaned_tokens.append(lemmatizer.lemmatize(token))

    return " ".join(cleaned_tokens)

# ==========================================================
# RISK PREDICTION
# ==========================================================

def predict_risk(text):

    cleaned = preprocess(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    return prediction

# ==========================================================
# SENTIMENT ANALYSIS
# ==========================================================

def analyze_sentiment(text):

    scores = sia.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        return "Positive"

    elif compound <= -0.05:
        return "Negative"

    return "Neutral"

# ==========================================================
# NAMED ENTITY RECOGNITION
# ==========================================================

def extract_entities(text):

    doc = nlp(text)

    return [(ent.text, ent.label_) for ent in doc.ents]

# ==========================================================
# MAIN PAGE
# ==========================================================

st.title("🛢️ Oil & Gas Incident Intelligence System")

st.markdown(
    """
This AI-powered NLP application analyzes industrial incident reports using:

- Risk Classification
- Sentiment Analysis
- Named Entity Recognition (NER)

Enter an incident report below.
"""
)

# ==========================================================
# USER INPUT
# ==========================================================

user_input = st.text_area(
    "Enter Incident Report",
    height=180,
    placeholder="Example: Gas leak detected near offshore platform."
)

# ==========================================================
# ANALYSIS
# ==========================================================

if st.button("Analyze Report"):

    if not user_input.strip():

        st.warning("Please enter an incident report.")

    else:

        risk = predict_risk(user_input)

        sentiment = analyze_sentiment(user_input)

        entities = extract_entities(user_input)

        # -------------------------------
        # Risk
        # -------------------------------

        st.subheader("Risk Prediction")

        risk_text = str(risk).strip().lower()

        if risk_text == "high":
            st.error(f"⚠️ {risk}")

        elif risk_text == "medium":
            st.warning(f"⚠️ {risk}")

        else:
            st.success(f"✅ {risk}")

        # -------------------------------
        # Sentiment
        # -------------------------------

        st.subheader("Sentiment Analysis")

        st.info(f"Detected Sentiment: **{sentiment}**")

        # -------------------------------
        # Entities
        # -------------------------------

        st.subheader("Named Entities")

        if entities:

            for entity, label in entities:
                st.write(f"• **{entity}** ({label})")

        else:

            st.write("No entities detected.")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.caption("Built with Streamlit, Scikit-learn, NLTK, and spaCy")