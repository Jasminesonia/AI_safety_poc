# import streamlit as st
# import joblib
# import os

# # Path to model
# MODEL_PATH = "models/baseline_model.pkl"

# # Load model
# @st.cache_resource
# def load_model():
#     if os.path.exists(MODEL_PATH):
#         return joblib.load(MODEL_PATH)
#     else:
#         st.error("Model not found! Please train the model first.")
#         return None

# model = load_model()

# # Streamlit UI
# st.set_page_config(page_title="AI Safety Model", page_icon="🤖")

# st.title("🛡️ AI Safety Text Classifier")
# st.write("This app classifies input text as **Safe** or **Unsafe** using the trained ML model.")

# # User input
# user_input = st.text_area("Enter a message for analysis:", "")

# if st.button("Classify"):
#     if model is not None and user_input.strip() != "":
#         prediction = model.predict([user_input])[0]
#         label = "✅ Safe" if prediction == 0 else "⚠️ Unsafe"
        
#         st.markdown(f"### Prediction: {label}")
#     else:
#         st.warning("Please enter some text to classify.")


# import streamlit as st
# from transformers import pipeline
# import os

# # -------------------------------
# # Paths (optional for later integration)
# # -------------------------------
# MODEL_PATH = "models/baseline_model.pkl"

# # -------------------------------
# # Load Pretrained BERT Model for Abuse Detection
# # -------------------------------
# @st.cache_resource
# def load_abuse_model():
#     return pipeline("text-classification", model="bert-base-uncased", return_all_scores=True)

# abuse_model = load_abuse_model()

# # -------------------------------
# # Crisis Keywords
# # -------------------------------
# CRISIS_KEYWORDS = [
#     "kill myself", "i want to die", "suicide", "end my life", "cant go on", "hopeless"
# ]

# # -------------------------------
# # Age-Appropriate Filtering Words
# # -------------------------------
# AGE_RESTRICTIONS = {
#     "child": ["violence", "drugs", "sex", "hate", "kill"],
#     "teen": ["drugs", "sex", "hate"],
#     "adult": []
# }

# # -------------------------------
# # Session State: Message History for Escalation
# # -------------------------------
# if "message_history" not in st.session_state:
#     st.session_state.message_history = []

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.set_page_config(page_title="AI Safety POC", page_icon="🛡️")
# st.title("🛡️ AI Safety Chat Simulator")
# st.write("Real-time AI Safety checks: Abuse Detection, Escalation, Crisis, Age-Appropriate Filtering")

# user_input = st.text_area("Enter your message:")
# user_age_group = st.selectbox("Select user age group:", ["child", "teen", "adult"])

# if st.button("Analyze"):

#     if user_input.strip() != "":
#         # -------------------------------
#         # 1️⃣ Abuse Detection
#         # -------------------------------
#         results = abuse_model(user_input)[0]
#         unsafe_score = next((r['score'] for r in results if r['label'] == 'LABEL_1'), 0)
#         safe_score = next((r['score'] for r in results if r['label'] == 'LABEL_0'), 0)

#         if unsafe_score > safe_score:
#             st.warning(f"⚠️ Unsafe message detected! Probability: {unsafe_score:.2f}")
#         else:
#             st.success(f"✅ Safe message. Probability: {safe_score:.2f}")

#         # -------------------------------
#         # 2️⃣ Escalation Detection
#         # -------------------------------
#         st.session_state.message_history.append(unsafe_score)
#         if len(st.session_state.message_history) >= 3:
#             last_three = st.session_state.message_history[-3:]
#             if all(score > 0.5 for score in last_three):
#                 st.warning("⚠️ Escalation detected! Multiple consecutive unsafe messages.")

#         # -------------------------------
#         # 3️⃣ Crisis Intervention
#         # -------------------------------
#         crisis_detected = any(kw in user_input.lower() for kw in CRISIS_KEYWORDS)
#         if crisis_detected:
#             st.error("🚨 Crisis alert! Potential self-harm message detected.")

#         # -------------------------------
#         # 4️⃣ Age-Appropriate Filtering
#         # -------------------------------
#         restricted_words = AGE_RESTRICTIONS.get(user_age_group, [])
#         age_violation = any(word in user_input.lower() for word in restricted_words)
#         if age_violation:
#             st.info("🔒 Age restriction violation: message contains content not suitable for this age group.")

#     else:
#         st.warning("Please enter a message to classify.")


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
