import pickle
from src.preprocess import clean_text

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector).max()

    label = "Positive 😊" if prediction == 1 else "Negative 😠"
    return label, probability
