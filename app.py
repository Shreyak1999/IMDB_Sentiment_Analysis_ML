import streamlit as st
from src.predict import predict_sentiment

st.title("🧠 Text Sentiment Analyzer")

user_input = st.text_area("Enter text to analyze:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, prob = predict_sentiment(user_input)
        st.success(f"Sentiment: {label}")
        st.write(f"Confidence: {prob:.2f}")
