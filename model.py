import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load artifacts (tfidf_vectorizer, label_encoder, and model)
def load_model_artifacts():
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    with open("resume_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, label_encoder, model

# Predict resume category
def predict_category(resume_text, vectorizer, model, label_encoder):
    tfidf_vector = vectorizer.transform([resume_text])
    prediction = model.predict(tfidf_vector)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return predicted_label
