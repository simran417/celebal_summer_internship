
import streamlit as st
import numpy as np
import pickle

# Load model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('🩺 Diabetes Progression Predictor')
st.write('Enter health data to predict diabetes progression.')

# Input sliders
age = st.slider('Age (standardized)', -0.1, 0.2, 0.05)
bmi = st.slider('BMI', 0.0, 0.2, 0.1)
bp = st.slider('Blood Pressure', 0.0, 0.2, 0.1)
s1 = st.slider('S1 (TC)', 0.0, 0.2, 0.1)
s5 = st.slider('S5 (Ldl)', 0.0, 0.4, 0.2)

# Prepare input
input_data = np.array([[age, bmi, bp, s1, s5] + [0]*5])

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.success(f'Predicted Value: {round(prediction[0], 2)}')
