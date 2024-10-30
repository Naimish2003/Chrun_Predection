import pandas as pd
import pickle
import streamlit as st

# Load the data and model
df_1 = pd.read_csv("tel_churn.csv")
model = pickle.load(open("/artifacts/model.pkl", "rb"))

# Define input fields
st.title("Customer Churn Prediction")
st.write("Please provide the following details to predict customer churn:")

# Collecting inputs
inputQuery1 = st.selectbox("SeniorCitizen", [0, 1])  # Assuming binary input
inputQuery2 = st.number_input("MonthlyCharges", min_value=0.0)
inputQuery3 = st.number_input("TotalCharges", min_value=0.0)
inputQuery4 = st.selectbox("Gender", ["Male", "Female"])
inputQuery5 = st.selectbox("Partner", ["Yes", "No"])
inputQuery6 = st.selectbox("Dependents", ["Yes", "No"])
inputQuery7 = st.selectbox("PhoneService", ["Yes", "No"])
inputQuery8 = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])
inputQuery9 = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
inputQuery10 = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
inputQuery11 = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
inputQuery12 = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
inputQuery13 = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
inputQuery14 = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
inputQuery15 = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
inputQuery16 = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
inputQuery17 = st.selectbox("PaperlessBilling", ["Yes", "No"])
inputQuery18 = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
inputQuery19 = st.number_input("Tenure", min_value=1)

# Process the input data
data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
         inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
         inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
new_df = pd.DataFrame(data, columns = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                       'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                       'PaymentMethod', 'tenure'])

# Get dummy variables for categorical data
new_df_dummies = pd.get_dummies(new_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                         'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                         'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                         'Contract', 'PaperlessBilling', 'PaymentMethod']], drop_first=True)

# Add the numerical features back to the dummies DataFrame
new_df_final = pd.concat([new_df_dummies, new_df[['MonthlyCharges', 'TotalCharges', 'tenure']]], axis=1)

# Ensure the order of features matches the model's training features
feature_names = model.feature_names_in_  # Get feature names used during training
new_df_final = new_df_final.reindex(columns=feature_names, fill_value=0)  # Reindex to ensure order and fill missing with 0

# Prediction
if st.button("Predict"):
    single = model.predict(new_df_final)

    if single == 1:
        st.write("This customer is likely to churn!")
    else:
        st.write("This customer is likely to stay.")
