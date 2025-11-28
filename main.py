# ---------- main.py ----------
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

print("▶️ Running:", __file__)

# 🧹 Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Load CSVs
true = pd.read_csv("True.csv")
fake = pd.read_csv("Fake.csv")
true["label"] = 1
fake["label"] = 0

# Combine
data = pd.concat([true, fake], ignore_index=True)
data["text"] = data["text"].astype(str).apply(clean_text)

# Split
X = data["text"]
y = data["label"]

# ✨ BEST VECTOR SETTINGS (BIGRAMS + LIMITED FEATURES)
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.75,
    ngram_range=(1, 2),
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

# Train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# ✨ BEST MODEL: Linear SVM (super accurate for text)
model = LinearSVC()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, pred) * 100, 2)
print("✅ Model accuracy:", accuracy, "%")

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("✅ Saved model & vectorizer")
