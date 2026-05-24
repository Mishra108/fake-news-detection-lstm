import streamlit as st
import joblib
import re
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download stopwords
nltk.download('stopwords')

# Load tokenizer
tokenizer = joblib.load("tokenizer.pkl")

# Load model
model = load_model("fake_news_lstm_model.h5", compile=False)

# Stopwords
stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Predict function
def predict_news(news):

    news = clean_text(news)

    seq = tokenizer.texts_to_sequences([news])

    padded = pad_sequences(seq, maxlen=300)

    prediction = model.predict(padded)

    return prediction[0][0]

# UI
st.title("📰 Fake News Detection System")

st.write("Enter a news article to check whether it is Real or Fake.")

news_input = st.text_area("Enter News Article")

if st.button("Predict"):

    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:

        pred = predict_news(news_input)

        if pred > 0.5:
            st.success(f"🟢 REAL NEWS\n\nConfidence: {pred:.4f}")
        else:
            st.error(f"🔴 FAKE NEWS\n\nConfidence: {pred:.4f}")