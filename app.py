import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")


@st.cache_resource
def load_model():
    return joblib.load("loan_model.pkl")


@st.cache_data
def load_features():
    with open("loan_features.json") as f:
        return json.load(f)


model = load_model()
feature_order = load_features()

st.title("🏦 Loan Approval Predictor")
st.warning(
    "Demo only. The model was trained on a synthetic dataset whose patterns differ from "
    "real lending, so it must not be used for real loan decisions."
)

edu_map = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "Doctorate": 5}

col1, col2 = st.columns(2)
education = col1.selectbox("Education", list(edu_map.keys()), index=2)
income = col2.number_input("Annual income", min_value=8000, max_value=1000000, value=60000, step=1000)

experience = col1.number_input("Work experience (years)", min_value=0, max_value=60, value=3)
credit_score = col2.number_input("Credit score", min_value=300, max_value=850, value=640)

loan_amnt = col1.number_input("Loan amount", min_value=500, max_value=35000, value=10000, step=500)
int_rate = col2.number_input("Interest rate (%)", min_value=5.0, max_value=20.0, value=11.0, step=0.1)

home = st.selectbox("Home ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
intent = st.selectbox("Loan purpose", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
                                       "MEDICAL", "PERSONAL", "VENTURE"])
default = st.selectbox("Previous loan default on file", ["No", "Yes"])

threshold = st.slider("Approval threshold", 0.1, 0.9, 0.5, 0.05,
                      help="Higher = stricter (fewer wrong approvals, more good applicants rejected)")

if st.button("Check application"):
    # same encoding as the notebook: MORTGAGE and DEBTCONSOLIDATION are the "left out" base groups
    row = {
        "person_education": edu_map[education],
        "person_income": income,
        "person_emp_exp": experience,
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": round(loan_amnt / income, 2),
        "credit_score": credit_score,
        "previous_loan_defaults_on_file": 1 if default == "Yes" else 0,
        "person_home_ownership_OTHER": int(home == "OTHER"),
        "person_home_ownership_OWN": int(home == "OWN"),
        "person_home_ownership_RENT": int(home == "RENT"),
        "loan_intent_EDUCATION": int(intent == "EDUCATION"),
        "loan_intent_HOMEIMPROVEMENT": int(intent == "HOMEIMPROVEMENT"),
        "loan_intent_MEDICAL": int(intent == "MEDICAL"),
        "loan_intent_PERSONAL": int(intent == "PERSONAL"),
        "loan_intent_VENTURE": int(intent == "VENTURE"),
    }
    X = pd.DataFrame([row])[feature_order]
    prob = model.predict_proba(X)[0, 1]

    st.metric("Approval probability", f"{prob:.1%}")
    if prob >= threshold:
        st.success("Prediction: APPROVED")
    else:
        st.error("Prediction: REJECTED")
    st.caption(f"Loan is {row['loan_percent_income']:.0%} of income. Threshold used: {threshold:.2f}.")
