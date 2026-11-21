from pathlib import Path

import pandas as pd
import streamlit as st
from lifelines import CoxPHFitter

DATA = (Path(__file__).resolve().parent.parent
        / "data" / "heart_failure_clinical_records_dataset.csv")


@st.cache_resource
def load_model():
    df = pd.read_csv(DATA)
    cox = CoxPHFitter(penalizer=0.1).fit(df, "time", "DEATH_EVENT")
    scores = cox.predict_partial_hazard(df)
    low_cut, high_cut = scores.quantile([0.33, 0.66])
    return cox, df, low_cut, high_cut


cox, df, low_cut, high_cut = load_model()

st.title("Heart Failure Follow-up Prioritiser")
st.caption("Student project. Not for real medical decisions.")

yes_no = {"No": 0, "Yes": 1}

with st.form("patient"):
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", 40, 95, 60)
        ef = st.number_input("Ejection fraction (%)", 10, 80, 38)
        creat = st.number_input("Serum creatinine (mg/dL)", 0.5, 10.0, 1.1, step=0.1)
        sodium = st.number_input("Serum sodium (mEq/L)", 100, 150, 137)
        cpk = st.number_input("Creatinine phosphokinase (mcg/L)", 20, 8000, 250)
        platelets = st.number_input("Platelets (kiloplatelets/mL)", 25000, 850000, 260000)
    with c2:
        sex = st.selectbox("Sex", ["Female", "Male"])
        anaemia = st.selectbox("Anaemia", list(yes_no))
        diabetes = st.selectbox("Diabetes", list(yes_no))
        hbp = st.selectbox("High blood pressure", list(yes_no))
        smoking = st.selectbox("Smoking", list(yes_no))
    submitted = st.form_submit_button("Predict")

if submitted:
    patient = pd.DataFrame([{
        "age": float(age),
        "anaemia": yes_no[anaemia],
        "creatinine_phosphokinase": cpk,
        "diabetes": yes_no[diabetes],
        "ejection_fraction": ef,
        "high_blood_pressure": yes_no[hbp],
        "platelets": float(platelets),
        "serum_creatinine": float(creat),
        "serum_sodium": sodium,
        "sex": 1 if sex == "Male" else 0,
        "smoking": yes_no[smoking],
    }])

    score = float(cox.predict_partial_hazard(patient).iloc[0])
    if score < low_cut:
        band = "LOW"
        st.success("Risk band: LOW. Routine follow-up.")
    elif score < high_cut:
        band = "MEDIUM"
        st.warning("Risk band: MEDIUM. Consider a somewhat earlier visit.")
    else:
        band = "HIGH"
        st.error("Risk band: HIGH. Prioritise an earlier follow-up visit.")

    prob90 = float(cox.predict_survival_function(patient, times=[90]).iloc[0, 0])
    st.metric("Estimated survival at 90 days", f"{prob90:.0%}")

    st.subheader("Predicted survival curve")
    curve = cox.predict_survival_function(patient)
    curve.columns = ["Survival probability"]
    curve.index.name = "Days since discharge"
    st.line_chart(curve)

    st.caption("Risk bands compare this patient with the 299 patients in the "
               "dataset: lowest third, middle third, highest third.")