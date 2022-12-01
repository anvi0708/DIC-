import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import xgboost as xgb
import numpy as np
st.header("Claim Prediction app")
st.text_input("Enter your Name: ", key="name")
data = pd.read_csv("dic.csv")
#load label encoder
encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)

# load model
best_xgboost_model = xgb.XGBRegressor()
best_xgboost_model.load_model("best_model.json")

if st.checkbox('Show Training Dataframe'):
    data
    
st.subheader("Please select relevant features of your Agency")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency = st.radio(
        'Name of the Agency',
        np.unique(data['Agency']))   
    
st.subheader("Please select relevant features of your Agency Type")
left_column, right_column = st.columns(2)
with left_column:
    inp_Agency_Type = st.radio(
        'Name of the Agency Type',
        np.unique(data['Agency_Type']))   
    
st.subheader("Please select relevant features of your Dist Channel")
left_column, right_column = st.columns(2)
with left_column:
    inp_Dist_Channel = st.radio(
        'Name of the Dist Channel',
        np.unique(data['Dist_Channel']))   
    
    
st.subheader("Please select relevant features of your Product")
left_column, right_column = st.columns(2)
with left_column:
    inp_Prod_Name = st.radio(
        'Name of the Product',
        np.unique(data['Prod_Name']))



input_Duration = st.slider('Enter Duration', 0, max(data["Duration"]), 200)

st.subheader("Please select relevant features of your Destination")
left_column, right_column = st.columns(2)
with left_column:
    inp_Destination = st.radio(
        'Name of the Destination',
        np.unique(data['Destination']))

input_Net_Sales = st.slider('Enter Net Sales', 0.0, max(data["Net_Sales"]), 100.0)
input_Commission = st.slider('Enter Commission Value', 0.0, max(data["Commission"]), 100.0)
input_Age = st.slider('Enter Age', 0, max(data["Age"]), 100)

if st.button('Make Prediction'):
    input_Agency = encoder.transform(np.expand_dims(inp_Agency, -1))
    input_Agency_Type = encoder.transform(np.expand_dims(inp_Agency_Type, -1))
    input_Dist_Channel = encoder.transform(np.expand_dims(inp_Dist_Channel, -1))
    input_Prod_Name = encoder.transform(np.expand_dims(inp_Prod_Name, -1))
    input_Destination = encoder.transform(np.expand_dims(inp_Destination, -1))
    inputs = np.expand_dims(
        [int(input_Agency),int(input_Agency_Type), int(input_Dist_Channel), int(input_Prod_Name), input_Duration, int(input_Destination), input_Net_Sales, input_Commission, input_Age], 0)
    prediction = best_xgboost_model.predict(inputs)
    print("final pred", np.squeeze(claim, -1))
    st.write(f"Your fish weight is: {np.squeeze(claim, -1):.2f}g")

    st.write(f"Thank you {st.session_state.name}! I hope you liked it.")
    st.write(f"If you want to see more advanced applications you can follow me on [medium](https://medium.com/@gkeretchashvili)")



