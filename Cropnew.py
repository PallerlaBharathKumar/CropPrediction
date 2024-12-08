import pandas as pd
import joblib
import streamlit as st
import os
import base64
import time

# Set page layout
st.set_page_config(layout="wide")
st.sidebar.title("Crop Prediction")

# Sidebar navigation
Mode = st.sidebar.selectbox(
    'Choose mode',
    ['About Project', 'Crop Predictor', 'Dataset', 'Range of Crops']
)

# About Project Section
if Mode == 'About Project':
    st.markdown(
        """
        <h1 style='text-align: center; color: lightseagreen;'>Crop Recommendation</h1>
        <p style='text-align: center; font-style: italic; color: crimson;'>Maximize agricultural yield by recommending appropriate crops</p>
        <h2>About:</h2>
        <ul>
            <li>This project helps beginner farmers determine suitable crops based on soil parameters.</li>
            <li>Uses machine learning (Logistic Regression) for crop prediction.</li>
        </ul>
        <h2>Parameters Considered:</h2>
        <ul>
            <li>Nitrogen (N)</li>
            <li>Phosphorus (P)</li>
            <li>Potassium (K)</li>
            <li>Temperature</li>
            <li>Humidity</li>
            <li>PH</li>
            <li>Rainfall</li>
        </ul>
        """, unsafe_allow_html=True
    )

# Crop Predictor Section
if Mode == 'Crop Predictor':
    st.title("Crop Prediction using Machine Learning")
    st.image('./images/crop2.jpg', use_column_width=True)

    # Input fields for parameters
    box1 = st.number_input('Enter Nitrogen Value', step=1, value=0)
    box2 = st.number_input('Enter Phosphorous Value', step=1, value=0)
    box3 = st.number_input('Enter Potassium Value', step=1, value=0)
    box4 = st.number_input('Enter Temperature Value', step=1, value=0)
    box5 = st.number_input('Enter Humidity Value', step=1, value=0)
    box6 = st.number_input('Enter PH Value', step=1, value=0)
    box7 = st.number_input('Enter Rainfall Value', step=1, value=0)

    testing = [[box1, box2, box3, box4, box5, box6, box7]]

    try:
        # Load model
        model = joblib.load('./models/crop_model.pkl')  # Ensure the file path is correct
        prediction = model.predict(testing)
        var = int(prediction[0])

        if st.button("Predict"):
            crop_list = [
                "Apple", "Banana", "Blackgram", "Chickpea", "Coconut", "Coffee",
                "Cotton", "Grapes", "Jute", "Kidneybeans", "Lentil", "Maize",
                "Mango", "Mothbeans", "Mungbean", "Muskmelon", "Orange", "Papaya",
                "Pigeonpeas", "Pomegranate", "Rice", "Watermelon"
            ]
            st.success(f"{crop_list[var]} is suitable for this land.")
    except Exception as e:
        st.error(f"Error loading the model: {e}")

# Dataset Section
if Mode == 'Dataset':
    st.title("Crop Recommendation Dataset")
    try:
        dataset = pd.read_csv("Crop_recommendation.csv")
        st.dataframe(dataset)

        # CSV download functionality
        def csv_downloader(data):
            csvfile = data.to_csv(index=False)
            b64 = base64.b64encode(csvfile.encode()).decode()
            new_filename = f"crop_recommendation_{time.strftime('%d-%m-%Y')}.csv"
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download CSV</a>',
                        unsafe_allow_html=True)
        csv_downloader(dataset)
    except Exception as e:
        st.error(f"Error loading the dataset: {e}")

# Range of Crops Section
if Mode == 'Range of Crops':
    st.title("Range of Crop Parameters")
    try:
        dataset = pd.read_csv("Crop_recommendation.csv")
        crop_list = dataset['Crop'].unique()
        Crop = st.selectbox("Select Crop", crop_list)

        def calculate_range(data, parameter):
            values = data[data["Crop"] == Crop][parameter].tolist()
            return min(values), max(values)

        columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        for col in columns:
            min_val, max_val = calculate_range(dataset, col)
            st.success(f"Range of {col} for {Crop}: {min_val} - {max_val}")

    except Exception as e:
        st.error(f"Error: {e}")
