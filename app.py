# ---------- app.py ----------
import streamlit as st
import pickle
import os
import re
import numpy as np

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detector")
st.caption("Type or paste a news headline / paragraph and press *Check News*")

# small helper: same cleaning used in training
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

# Load model & vectorizer
model = None
vectorizer = None
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    st.error(
        "Model files not found. Make sure `model.pkl` and `vectorizer.pkl` are in this folder.\n\n"
        "Run `python main.py` first to train & save them."
    )
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model/vectorizer: {e}")
    st.stop()

# Input area
text_input = st.text_area("Enter news text (headline or paragraph):", height=160)

# Quick example buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Example: Real"):
        text_input = "Government launches new renewable energy project to reduce emissions."
        st.session_state['text_input'] = text_input
with col2:
    if st.button("Example: Fake"):
        text_input = "Aliens found living under the sea with secret cities, viral report claims."
        st.session_state['text_input'] = text_input
with col3:
    if st.button("Clear"):
        text_input = ""
        st.session_state['text_input'] = text_input

# If using session_state to preserve example load for text_area:
if "text_input" in st.session_state:
    text_input = st.session_state["text_input"]
    st.experimental_rerun() if st.session_state.get("_rerun_once", False) else None

# Action
if st.button("Check News"):
    if not text_input or text_input.strip() == "":
        st.warning("Please enter some text to check.")
    else:
        cleaned = clean_text(text_input)
        try:
            X = vectorizer.transform([cleaned])
        except Exception as e:
            st.error(f"Error transforming input text with the vectorizer: {e}")
            st.stop()

        # Predict
        try:
            label = model.predict(X)[0]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # Get confidence (works for classifiers with predict_proba or decision_function)
        confidence = None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                # assume label 1 = real, 0 = fake
                confidence = float(probs[label])
            elif hasattr(model, "decision_function"):
                df = model.decision_function(X)
                # convert to probability-like score via sigmoid
                score = float(df[0]) if isinstance(df, (list, np.ndarray)) else float(df)
                prob = 1 / (1 + np.exp(-score))
                # for binary, map to chosen label confidence
                confidence = float(prob) if label == 1 else float(1 - prob)
            else:
                confidence = None
        except Exception:
            confidence = None

        # Show result
        if label == 1:
            st.success("✅ This looks like **Real News**")
            if confidence is not None:
                st.info(f"Confidence: {round(confidence * 100, 2)}%")
        else:
            st.error("🚨 This seems like **Fake News**")
            if confidence is not None:
                st.info(f"Confidence: {round(confidence * 100, 2)}%")

        # Show cleaned text & raw model output for debugging/inspection
        with st.expander("🛠️ Debug / Details"):
            st.write("**Cleaned input:**", cleaned)
            try:
                if hasattr(model, "predict_proba"):
                    st.write("**Predict_proba:**", model.predict_proba(X)[0].tolist())
                if hasattr(model, "decision_function"):
                    st.write("**Decision function:**", model.decision_function(X).tolist())
            except Exception as e:
                st.write("Could not fetch debug details:", e)

st.markdown("---")
st.caption("Built by Kushal — Fake News Detector (TF-IDF + SVM).")
