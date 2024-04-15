import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, sys
from sklearn import set_config
set_config(transform_output='pandas')
# Load the filepaths
FILEPATHS_FILE = 'config/filepaths.json'
with open(FILEPATHS_FILE) as f:
    FPATHS = json.load(f)
    
# Define the load raw eda data function with caching
@st.cache_data
def load_data(fpath):
    df = pd.read_csv(fpath)
    return df
    
# Define the load train or test data function with caching
@st.cache_data
def load_Xy_data(fpath):
    return joblib.load(fpath)
    
@st.cache_resource
def load_model_ml(fpath):
    return joblib.load(fpath)
    
### Start of App
st.title('House Price Predictor')


# Load & cache dataframe
#df = load_data(fpath = FPATHS['data']['ml']['train'])
# Load training data
X_train, y_train = load_Xy_data(fpath=FPATHS['data']['ml']['train'])
# Load testing data
X_test, y_test = load_Xy_data(fpath=FPATHS['data']['ml']['test'])
# Load model
linreg = load_model_ml(fpath = FPATHS['models']['linear_regression'])


# Add text for entering features
st.subheader("Select values using the sidebar on the left.\n Then check the box below to predict the price.")
st.sidebar.subheader("Enter House Features For Prediction")
# # Create widgets for each feature
# Living Area Sqft
selected_sqft = st.sidebar.number_input('Living Area Sqft (100-6000 sqft)', min_value=100, max_value=6000, step = 100)
# Bedroom
selected_bedrooms = st.sidebar.slider('Bedroom', min_value=0, max_value=10)

# Total Full Baths
selected_full_baths = st.sidebar.slider("Total Full Baths", min_value=0, max_value=6, step = 1)


# Define function to convert widget values to dataframe
def get_X_to_predict():
    X_to_predict = pd.DataFrame({'Living Area Sqft': selected_sqft,
                                 'Bedroom': selected_bedrooms,
                                 'Total Full Baths': selected_full_baths},
                                index=['id'])
    return X_to_predict



def get_prediction(model,X_to_predict):
    return  model.predict(X_to_predict)[0]
    
if st.checkbox("Predict"):
    
    X_to_pred = get_X_to_predict()
    new_pred = get_prediction(linreg, X_to_pred)
    
    st.markdown(f"> #### Model Predicted Price = ${new_pred:,.0f}")
    
else:
    st.empty()
