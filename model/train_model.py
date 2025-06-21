# model/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sentence_transformers import SentenceTransformer
import joblib
import os

# Load training data
df = pd.read_csv("data/training_data.csv")

X = df['text']
y = df['role']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
X_embeddings = embedder.encode(X.tolist())

# Balance the dataset
ros = RandomOverSampler()
X_balanced, y_balanced = ros.fit_resample(X_embeddings, y_encoded)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_balanced, y_balanced)

# Save model artifacts
os.makedirs("model", exist_ok=True)
joblib.dump(clf, "model/job_role_classifier.pkl")
joblib.dump(embedder, "model/embedding_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")
