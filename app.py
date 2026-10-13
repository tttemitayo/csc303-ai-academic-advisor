import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Academic Advisor", page_icon="🎓", layout="centered")

MODEL_PATH = Path(__file__).parent / "academic_advisor_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.title("🎓 AI Academic Advisor System")
st.write("A student-support tool that uses a machine-learning model to suggest an academic support level.")
st.info("This is an educational prototype. Its recommendations are guidance, not an official university decision.")

with st.form("advisor_form"):
    st.subheader("Enter your academic information")
    gpa = st.slider("Current GPA (4.0 scale)", min_value=0.0, max_value=4.0, value=2.8, step=0.1)
    attendance = st.slider("Class attendance (%)", min_value=0, max_value=100, value=80, step=1)
    study_hours = st.slider("Average independent study hours per week", min_value=0, max_value=40, value=10, step=1)
    difficult_courses = st.slider("Number of courses you currently find difficult", min_value=0, max_value=10, value=2, step=1)
    failed_courses = st.slider("Number of failed courses to retake", min_value=0, max_value=10, value=0, step=1)
    submitted = st.form_submit_button("Generate Academic Advice", type="primary")

if submitted:
    model = load_model()
    features = pd.DataFrame([{
        "gpa": gpa,
        "attendance": attendance,
        "study_hours": study_hours,
        "difficult_courses": difficult_courses,
        "failed_courses": failed_courses
    }])
    prediction = model.predict(features)[0]
    st.subheader("Your academic support recommendation")
    if prediction == "Strong":
        st.success("**Maintain and build on your performance.**")
        advice = [
            "Keep a weekly study timetable and review lecture notes consistently.",
            "Start assignments early and practise past questions before tests.",
            "Continue attending classes and ask lecturers questions when topics are unclear."
        ]
    elif prediction == "Moderate":
        st.warning("**Use a structured improvement plan.**")
        advice = [
            "Set aside focused study blocks for your most difficult courses.",
            "Meet a lecturer, tutor, or course adviser to clarify challenging topics.",
            "Track assignment deadlines and review your progress every week."
        ]
    else:
        st.error("**Prioritise academic support.**")
        advice = [
            "Contact your academic adviser or department as soon as possible.",
            "Create a recovery timetable for failed or difficult courses.",
            "Improve attendance, use office hours, and seek tutoring or a study group."
        ]
    for item in advice:
        st.write(f"• {item}")
    st.caption(f"Model classification: {prediction}. This prediction is based on a small demonstration dataset and should not be treated as a validated assessment.")
