
import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('iris_model.pkl', 'rb'))

# Set Streamlit page settings
st.set_page_config(page_title="Iris Classifier", layout="centered")

# Inject custom CSS for background and button
st.markdown("""
    <style>
    /* Yellow background */
    .main {
        background-color: #ffffcc;
        padding: 20px;
        border-radius: 10px;
    }

    /* Red button style */
    .stButton > button {
        background-color: red;
        color: white;
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5em 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("🌸 Iris Flower Classifier")
st.markdown("This app predicts the species of an Iris flower based on your input measurements.")

# Input sliders in two columns
st.subheader("📏 Enter Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 3.5)

with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

# Predict button
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.success(f"🌺 The predicted Iris species is: **{prediction[0]}**")
