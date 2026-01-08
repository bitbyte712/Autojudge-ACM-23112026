import streamlit as st
import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re

# ---------------- LOAD MODELS ----------------
tfidf = joblib.load("models/tfidf_vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

clf = joblib.load("models/xgb_classifier_final.pkl")
reg = joblib.load("models/xgb_regressor_final.pkl")
scaler = joblib.load("models/numeric_scaler.pkl")

# ---------------- FEATURE FUNCTIONS ----------------
def extract_numeric_features(text):
    text = text.lower()

    total_text_len = len(text.split())
    title_len = 0  # no title in UI

    math_symbols = len(re.findall(r"[=<>+\-*/]", text))
    has_constraints = int("constraint" in text or "limit" in text)
    multi_case = int("multiple test" in text or "multiple cases" in text)

    numbers = re.findall(r"\d+", text)
    constraint_density = len(numbers) / max(total_text_len, 1)

    algo_keywords = [
        "dp", "dynamic programming", "graph", "tree",
        "dfs", "bfs", "greedy", "heap",
        "segment tree", "binary search"
    ]
    algo_keyword_count = sum(1 for kw in algo_keywords if kw in text)

    return np.array([[ 
        total_text_len,
        title_len,
        math_symbols,
        has_constraints,
        multi_case,
        constraint_density,
        algo_keyword_count
    ]], dtype=float)

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AutoJudge", layout="centered")

st.title("🧠 AutoJudge")
st.subheader("Predict Programming Problem Difficulty")

st.markdown("### 📝 Problem Description")
desc_text = st.text_area("Description", height=200)

st.markdown("### 📥 Input Description")
input_text = st.text_area("Input Format", height=150)

st.markdown("### 📤 Output Description")
output_text = st.text_area("Output Format", height=150)

# ---------------- PREDICTION ----------------
if st.button("Predict Difficulty"):
    if not desc_text or not input_text or not output_text:
        st.warning("Please fill all three sections.")
    else:
        # -------- MERGE TEXT (MATCH TRAINING) --------
        merged_text = (
            desc_text + " " +
            input_text + " " +
            output_text
        )

        # -------- TEXT FEATURES --------
        X_text = tfidf.transform([merged_text])

        # -------- CLASSIFICATION --------
        class_pred_enc = clf.predict(X_text)[0]
        class_pred = label_encoder.inverse_transform([class_pred_enc])[0]

        # -------- REGRESSION --------
        X_num = extract_numeric_features(merged_text)
        X_num_scaled = scaler.transform(X_num)

        X_reg = hstack([X_text, X_num_scaled])
        X_reg = csr_matrix(X_reg)

        score_pred = reg.predict(X_reg)[0]

        # -------- OUTPUT --------
        st.success("Prediction Complete ✅")
        st.markdown(f"### → Difficulty Class: **{class_pred.capitalize()}**")
        st.markdown(f"### → Difficulty Score: **{score_pred:.2f}**")
