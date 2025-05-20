# streamlit_app.py

import streamlit as st
import pickle
import numpy as np

# Load vectorizer and models
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

models = {
    "Logistic Regression": pickle.load(open("models/logistic_regression.pkl", "rb")),
    "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
    "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
    "SVC": pickle.load(open("models/svc.pkl", "rb")),
}

# UI
st.title("📧 Email Classifier - Spam or Ham")
email_text = st.text_area("Enter the email content:")
model_name = st.selectbox("Choose a model", list(models.keys()))

if st.button("Classify"):
    if email_text.strip() == "":
        st.warning("Please enter some email content.")
    else:
        X = vectorizer.transform([email_text])
        prediction = models[model_name].predict(X)[0]
        prob = models[model_name].predict_proba(X)[0][1] if hasattr(models[model_name], 'predict_proba') else None
        
        if prediction == 1:
            st.error("🚫 This email is SPAM!")
        else:
            st.success("✅ This email is HAM (Not Spam).")

        if prob is not None:
            st.metric("Spam Probability", f"{prob*100:.2f}%")
