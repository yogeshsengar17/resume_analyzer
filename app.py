# app.py

import streamlit as st
from utils.score_calculator import predict_job_role, calculate_ats_score
from utils.resume_parser import extract_text

st.title("Resume Analyzer with JD Matching")

# Upload Resume
uploaded_file = st.file_uploader("Upload your resume (.pdf or .docx)", type=["pdf", "docx"])

# Job Description input
job_description_input = st.text_area("Paste Job Description here (optional for JD Match)", height=200)

if uploaded_file:
    resume_text = extract_text(uploaded_file)

    # Predict role + confidence
    predicted_role,confidence = predict_job_role(resume_text)
    st.success(f"Predicted Role: **{predicted_role}**")
    st.info(f"Confidence Score: **{confidence*100:.2f}%**")

    # If custom JD is given
    if job_description_input.strip():
        ats_score = calculate_ats_score(resume_text, job_description_input)
        st.warning(f"Custom JD Match Score: **{ats_score}%**")
        if ats_score >=20:
            st.success("This resume is a good match for the provided job description.")
        else:
            st.error("This resume may need improvements for this role.")
