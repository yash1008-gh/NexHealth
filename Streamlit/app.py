from __future__ import annotations

import pandas as pd
import requests
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.schemas import (
    A1C_RESULT_OPTIONS,
    AGE_BUCKET_OPTIONS,
    CHANGE_OPTIONS,
    DIABETES_MED_OPTIONS,
    GENDER_OPTIONS,
    MAX_GLU_SERUM_OPTIONS,
    MEDICATION_FIELDS,
    MEDICATION_STATUS_OPTIONS,
    RACE_OPTIONS,
    WEIGHT_BUCKET_OPTIONS,
)

# ---------------- CONFIG ----------------

st.set_page_config(page_title="NexHealth Inference", layout="wide")
st.title("NexHealth Patient Risk Inference")

api_base_url = st.sidebar.text_input("FastAPI base URL", value="http://127.0.0.1:8000")
explain = st.sidebar.checkbox("Include SHAP explanation", value=True)

# ---------------- HELPERS ----------------

def _optional_select(label, options, key):
    val = st.selectbox(label, ("",) + options, key=key)
    return val or None


def _optional_text(label, key):
    val = st.text_input(label, key=key).strip()
    return val or None


# ---------------- DUMMY DATA ----------------

def get_low_risk_patient():
    return {
        "encounter_id": 1,
        "patient_number": 1,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[30-40)",
        "weight": "[50-75)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 1,
        "time_in_hospital": 2,
        "payer_code": "MC",
        "medical_specialty": "InternalMedicine",
        "num_lab_procedures": 10,
        "num_procedures": 0,
        "num_medications": 3,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 0,
        "diag_1": "250",
        "diag_2": None,
        "diag_3": None,
        "number_diagnoses": 2,
        "max_glu_serum": "Norm",
        "A1Cresult": "Norm",
        "change": "No",
        "diabetesMed": "No",
        **{med: "No" for med in MEDICATION_FIELDS},
    }


def get_high_risk_patient():
    return {
        "encounter_id": 999,
        "patient_number": 999,
        "race": "AfricanAmerican",
        "gender": "Male",
        "age": "[70-80)",
        "weight": "[75-100)",
        "admission_type_id": 3,
        "discharge_disposition_id": 2,
        "admission_source_id": 7,
        "time_in_hospital": 10,
        "payer_code": "HM",
        "medical_specialty": "Cardiology",
        "num_lab_procedures": 80,
        "num_procedures": 3,
        "num_medications": 15,
        "number_outpatient": 2,
        "number_emergency": 3,
        "number_inpatient": 5,
        "diag_1": "250",
        "diag_2": "401",
        "diag_3": "428",
        "number_diagnoses": 9,
        "max_glu_serum": ">300",
        "A1Cresult": ">8",
        "change": "Ch",
        "diabetesMed": "Yes",
        **{med: "Up" for med in MEDICATION_FIELDS},
    }


# ---------------- QUICK TEST ----------------

st.subheader("Quick Test")

col1, col2 = st.columns(2)

if col1.button("Low Risk Patient"):
    st.session_state["payload"] = get_low_risk_patient()

if col2.button("High Risk Patient"):
    st.session_state["payload"] = get_high_risk_patient()


# ---------------- FORM ----------------

with st.form("patient_form"):

    st.subheader("Identifiers")
    col1, col2 = st.columns(2)
    encounter_id = col1.number_input("Encounter ID", 0, step=1)
    patient_number = col2.number_input("Patient Number", 0, step=1)

    st.subheader("Demographics")
    race = _optional_select("Race", RACE_OPTIONS, "race")
    gender = st.selectbox("Gender", GENDER_OPTIONS)
    age = st.selectbox("Age", AGE_BUCKET_OPTIONS)
    weight = _optional_select("Weight", WEIGHT_BUCKET_OPTIONS, "weight")

    st.subheader("Admission")
    col1, col2, col3, col4 = st.columns(4)
    admission_type_id = col1.number_input("Admission Type ID", 0)
    discharge_disposition_id = col2.number_input("Discharge ID", 0)
    admission_source_id = col3.number_input("Source ID", 0)
    time_in_hospital = col4.number_input("Time in Hospital", 0)

    st.subheader("Utilization")
    num_lab_procedures = st.number_input("Lab Procedures", 0)
    num_procedures = st.number_input("Procedures", 0)
    num_medications = st.number_input("Medications", 0)
    number_diagnoses = st.number_input("Diagnoses", 0)

    st.subheader("Emergency/Visits")
    number_outpatient = st.number_input("Outpatient", 0)
    number_emergency = st.number_input("Emergency", 0)
    number_inpatient = st.number_input("Inpatient", 0)

    st.subheader("Diagnosis")
    diag_1 = _optional_text("Diag 1", "d1")
    diag_2 = _optional_text("Diag 2", "d2")
    diag_3 = _optional_text("Diag 3", "d3")

    st.subheader("Labs")
    max_glu_serum = _optional_select("Max Glu", MAX_GLU_SERUM_OPTIONS, "glu")
    A1Cresult = _optional_select("A1C", A1C_RESULT_OPTIONS, "a1c")

    st.subheader("Medications")
    medication_payload = {}
    for med in MEDICATION_FIELDS:
        medication_payload[med] = st.selectbox(med, MEDICATION_STATUS_OPTIONS)

    st.subheader("Flags")
    change = st.selectbox("Change", CHANGE_OPTIONS)
    diabetesMed = st.selectbox("Diabetes Med", DIABETES_MED_OPTIONS)

    submitted = st.form_submit_button("Predict")


# ---------------- PAYLOAD ----------------

if submitted:
    if "payload" in st.session_state:
        payload = st.session_state["payload"]
    else:
        payload = {
            "encounter_id": encounter_id,
            "patient_number": patient_number,
            "race": race,
            "gender": gender,
            "age": age,
            "weight": weight,
            "admission_type_id": admission_type_id,
            "discharge_disposition_id": discharge_disposition_id,
            "admission_source_id": admission_source_id,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult,
            **medication_payload,
            "change": change,
            "diabetesMed": diabetesMed,
        }

    payload = {k: v for k, v in payload.items() if v is not None}

    st.json(payload)

    try:
        res = requests.post(
            f"{api_base_url}/predict",
            params={"explain": explain},
            json=payload,
        )
        res.raise_for_status()
        result = res.json()
    except Exception as e:
        st.error(str(e))
    else:
        st.success("Prediction complete")

        prediction = result["prediction"]
        probability = result["probability"]
        threshold = result["threshold_used"]

        if prediction == 1:
            risk_label = "🔴 High Risk"
        else:
            risk_label = "🟢 Low Risk"

        st.subheader("Prediction")

        st.markdown(f"### {risk_label}")
        st.write(f"Threshold: **{threshold:.2f}**")
        st.metric(
            label="Risk Level",
            value=f"{probability:.2f} prob",
            delta=f"{probability:.2f} prob"
        )


        if result.get("shap_values"):
                st.subheader("SHAP Explanation")

                df = pd.DataFrame(result["shap_values"])

                # Bar chart
                st.bar_chart(df.set_index("feature")["impact"])

                # Table (sorted)
                st.subheader("Feature Contributions")

                df_sorted = df.sort_values(by="impact", key=abs, ascending=False)

                st.dataframe(df_sorted)
        df_sorted["direction"] = df_sorted["impact"].apply(lambda x: "↑ increases risk" if x > 0 else "↓ decreases risk")
