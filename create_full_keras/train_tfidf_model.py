import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# === Load Dataset ===
df = pd.read_csv("data/dataset.csv")

# === Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english",
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(df["text"]).toarray()

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Keras Model ===
model = Sequential([
    Dense(1024, activation="relu", input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(512, activation="relu"),
    Dropout(0.3),
    Dense(5, activation="softmax")  # Five sentiment classes (0–4)
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# === Train ===
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=128,
    callbacks=[early_stop],
    verbose=2
)

# === Save Model + Vectorizer ===
os.makedirs("models", exist_ok=True)
model.save("models/full_keras_model.keras")

with open("models/tfidf_vectorizer.pkl", "wb") as f:
    import pickle
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved to /models/")