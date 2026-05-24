import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("fake_news_lstm_model.h5")

# Load tokenizer
tokenizer = joblib.load("tokenizer.pkl")

# Stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Prediction function
def predict_news(news):

    news = clean_text(news)

    seq = tokenizer.texts_to_sequences([news])

    padded = pad_sequences(seq, maxlen=300)

    prediction = model.predict(padded)[0][0]

    return prediction

# Streamlit UI
st.title("📰 Fake News Detection System")

news_input = st.text_area("Enter News Article")

if st.button("Predict"):

    pred = predict_news(news_input)

    if pred > 0.5:
        st.success(f"🟢 REAL NEWS\nConfidence: {pred:.4f}")
    else:
        st.error(f"🔴 FAKE NEWS\nConfidence: {pred:.4f}")