import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('reference.tsv', delimiter='|')  # Sesuaikan delimiter jika perlu

# Load the trained model
model_path = 'divorce_prediction.h5'  # Sesuaikan dengan nama file model Anda
model = load_model(model_path)

# Function to preprocess input data
def preprocess_data(data):
    # Example: Scaling the input data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# Function to predict divorce probability
def predict_divorce(answers):
    # Prepare input data for prediction
    input_data = np.array([answers])  # answers should be a list of 54 values
    processed_data = preprocess_data(input_data)
    
    # Predict using the loaded model
    prediction = model.predict(processed_data)
    
    return prediction[0][0]  # Assuming single output prediction

# Streamlit UI
st.title('Divorce Prediction App')

# Display the questions and collect answers
answers = []
for i, row in df.iterrows():
    attribute_id = row['atribute_id']
    description = row['description']
    answer = st.slider(f'{description}', 1, 5, 3, key=f'question_{attribute_id}')
    answers.append(answer)

# Prediction button
if st.button('Predict'):
    prediction = predict_divorce(answers)
    st.write(f'Probability of Divorce: {prediction:.2f}')

# Optionally, display the dataset or summary statistics
if st.checkbox('Show Dataset'):
    st.write(df)

# Optionally, add more Streamlit components as needed

