
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model_filename = 'decision_tree_model.pkl'
# Ensure the model file is available in the deployment environment
# In a real deployment, you'd ensure this file is part of your repo or accessible.

try:
    with open(model_filename, 'rb') as file:
        model = joblib.load(file)
except FileNotFoundError:
    st.error(f"Model file '{model_filename}' not found. Please ensure it's in the same directory as app.py")
    st.stop()

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("🎓 Student Performance Predictor")

st.markdown("Enter the student's details below to predict if they will Pass or Fail.")

# Input fields
st.header("Student Information")
study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=5.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
prev_marks = st.number_input("Previous Marks (%) (e.g., in a prior exam)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
assignment_option = st.selectbox("Assignment Submitted?", ("Yes", "No"))

# Convert assignment input to numerical (1 for Yes, 0 for No)
assignment = 1 if assignment_option == "Yes" else 0

# Prediction button
if st.button("Predict Performance"):
    # Prepare the input for the model
    user_input_array = np.array([[study_hours, attendance, prev_marks, assignment]])
    user_input_df = pd.DataFrame(user_input_array, columns=['StudyHours', 'Attendance', 'PrevMarks', 'Assignment'])

    # Make prediction
    prediction = model.predict(user_input_df)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.success("🎉 The student is likely to PASS!")
    else:
        st.error("😞 The student is likely to FAIL.")
