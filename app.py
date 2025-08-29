import streamlit as st
from transformers import pipeline
import joblib
import os

# Load BERT model
@st.cache_resource
def load_abuse_model():
    return pipeline("text-classification", model="bert-base-uncased", return_all_scores=True)

abuse_model = load_abuse_model()

# Load baseline model
@st.cache_resource
def load_baseline_model():
    return joblib.load("models/baseline_model.pkl")

baseline_model = load_baseline_model()

# Crisis keywords
CRISIS_KEYWORDS = [
    "kill myself", "i want to die", "suicide", "end my life", "cant go on", "hopeless"
]

# Age restrictions
AGE_RESTRICTIONS = {
    "child": ["violence", "drugs", "sex", "hate", "kill"],
    "teen": ["drugs", "sex", "hate"],
    "adult": []
}

# Session state for escalation
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Streamlit UI
st.set_page_config(page_title="AI Safety POC", page_icon="🛡️")
st.title("🛡️ AI Safety Chat Simulator")
st.write("Real-time AI Safety checks: Abuse Detection, Escalation, Crisis, Age-Appropriate Filtering")

user_input = st.text_area("Enter your message:")
user_age_group = st.selectbox("Select user age group:", ["child", "teen", "adult"])

if st.button("Analyze"):

    if user_input.strip() != "":
        # -------------------------------
        # Baseline prediction
        # -------------------------------
        baseline_pred = baseline_model.predict([user_input])[0]  # 0=safe, 1=unsafe
        st.info(f"Baseline model predicts: {'⚠️ Unsafe' if baseline_pred else '✅ Safe'}")

        # -------------------------------
        # BERT prediction
        # -------------------------------
        results = abuse_model(user_input)[0]
        unsafe_score = next((r['score'] for r in results if r['label'] == 'LABEL_1'), 0)
        bert_pred = 1 if unsafe_score > 0.5 else 0
        st.info(f"BERT model predicts: {'⚠️ Unsafe' if bert_pred else '✅ Safe'} (prob {unsafe_score:.2f})")

        # -------------------------------
        # Ensemble decision
        # -------------------------------
        if baseline_pred == bert_pred:
            final_label = baseline_pred
        else:
            final_label = baseline_pred  # baseline tie-breaker

        st.success(f"🔹 Final Safety Decision: {'⚠️ Unsafe' if final_label else '✅ Safe'}")

        # -------------------------------
        # Escalation detection
        # -------------------------------
        st.session_state.message_history.append(unsafe_score)
        if len(st.session_state.message_history) >= 3:
            last_three = st.session_state.message_history[-3:]
            if all(score > 0.5 for score in last_three):
                st.warning("⚠️ Escalation detected! Multiple consecutive unsafe messages.")

        # -------------------------------
        # Crisis intervention
        # -------------------------------
        if any(kw in user_input.lower() for kw in CRISIS_KEYWORDS):
            st.error("🚨 Crisis alert! Potential self-harm message detected.")

        # -------------------------------
        # Age-appropriate filtering
        # -------------------------------
        restricted_words = AGE_RESTRICTIONS.get(user_age_group, [])
        if any(word in user_input.lower() for word in restricted_words):
            st.info("🔒 Age restriction violation: message contains content not suitable for this age group.")

    else:
        st.warning("Please enter a message to classify.")



