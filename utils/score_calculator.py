# utils/score_calculator.py

import joblib
import re

# Load model artifacts
model = joblib.load("model/job_role_classifier.pkl")
embedder = joblib.load("model/embedding_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")


def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9 ]", " ", text.lower())


def predict_job_role(resume_text):
    cleaned = clean_text(resume_text)
    embedding = embedder.encode([cleaned])
    pred_class = model.predict(embedding)[0]
    confidence = max(model.predict_proba(embedding)[0])
    role = label_encoder.inverse_transform([pred_class])[0]
    return role,confidence

# def calculate_ats_score(resume_text, job_description):
#     resume_words = set(clean_text(resume_text).split())
#     jd_words = set(clean_text(job_description).split())
#     match_count = len(resume_words & jd_words)
#     return round((match_count / len(jd_words)) * 100, 2) if jd_words else 0.0

def calculate_ats_score(resume_text, job_description):
    resume_words = set(clean_text(resume_text).split())
    jd_words = set(clean_text(job_description).split())
    match_count = len(resume_words & jd_words)
    return round((match_count / len(jd_words)) * 100, 2) if jd_words else 0.0

