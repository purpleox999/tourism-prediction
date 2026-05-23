import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
from huggingface_hub import hf_hub_download
import os

# Define the repo_id where the model and preprocessor are stored
repo_id_model = "excelsiorR/tourism-package-prediction-model"

# Download and load the XGBoost model (saved as .json)
@st.cache_resource
def load_model():
    model_path_hf = hf_hub_download(repo_id=repo_id_model, filename="xgboost_model.json")
    loaded_model = xgb.XGBClassifier()
    loaded_model.load_model(model_path_hf)
    return loaded_model

# Download and load the preprocessor (saved as .joblib)
@st.cache_resource
def load_preprocessor():
    preprocessor_path_hf = hf_hub_download(repo_id=repo_id_model, filename="preprocessor.joblib")
    loaded_preprocessor = joblib.load(preprocessor_path_hf)
    return loaded_preprocessor

loaded_model = load_model()
loaded_preprocessor = load_preprocessor()

def predict_purchase(
    Age: float,
    TypeofContact: str,
    CityTier: int,
    DurationOfPitch: float,
    Occupation: str,
    Gender: str,
    NumberOfPersonVisiting: int,
    NumberOfFollowups: float,
    ProductPitched: str,
    PreferredPropertyStar: float,
    MaritalStatus: str,
    NumberOfTrips: float,
    Passport: int,
    PitchSatisfactionScore: int,
    OwnCar: int,
    NumberOfChildrenVisiting: float,
    Designation: str,
    MonthlyIncome: float
):
    input_data = {
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': Passport,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': OwnCar,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }

    input_df = pd.DataFrame([input_data])
    processed_input = loaded_preprocessor.transform(input_df)
    prediction = loaded_model.predict(processed_input)[0]
    prediction_proba = loaded_model.predict_proba(processed_input)[0, 1]

    result = "Will Purchase" if prediction == 1 else "Will Not Purchase"
    return f"Prediction: {result}, Probability: {prediction_proba:.2f}"

# Streamlit Interface
st.set_page_config(page_title="Tourism Package Purchase Predictor")
st.title("Tourism Package Purchase Predictor")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18.0, max_value=90.0, value=35.0)
        typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
        citytier = st.number_input("City Tier", min_value=1, max_value=3, value=1)
        durationofpitch = st.number_input("Duration of Pitch", min_value=1.0, max_value=100.0, value=10.0)
        occupation = st.selectbox("Occupation", ['Salaried', 'Free Lancer', 'Small Business', 'Large Business'])
        gender = st.selectbox("Gender", ['Female', 'Male'])
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        numberoffollowups = st.number_input("NumberOfFollowups", min_value=0.0, max_value=20.0, value=3.0)
        productpitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
    with col2:
        preferredpropertystar = st.number_input("Preferred Property Star", min_value=1.0, max_value=5.0, value=4.0)
        maritalstatus = st.selectbox("Marital Status", ['Single', 'Divorced', 'Married', 'Unmarried'])
        numberoftrips = st.number_input("NumberOfTrips", min_value=0.0, max_value=50.0, value=2.0)
        passport = st.number_input("Passport (0=No, 1=Yes)", min_value=0, max_value=1, value=1)
        pitchsatisfactionsocre = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=4)
        owncar = st.number_input("Own Car (0=No, 1=Yes)", min_value=0, max_value=1, value=1)
        numberofchildrenvisiting = st.number_input("NumberOfChildrenVisiting", min_value=0.0, max_value=10.0, value=1.0)
        designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, max_value=1000000.0, value=50000.0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        prediction_text = predict_purchase(
            Age=age,
            TypeofContact=typeofcontact,
            CityTier=citytier,
            DurationOfPitch=durationofpitch,
            Occupation=occupation,
            Gender=gender,
            NumberOfPersonVisiting=numberofpersonvisiting,
            NumberOfFollowups=numberoffollowups,
            ProductPitched=productpitched,
            PreferredPropertyStar=preferredpropertystar,
            MaritalStatus=maritalstatus,
            NumberOfTrips=numberoftrips,
            Passport=passport,
            PitchSatisfactionScore=pitchsatisfactionsocre,
            OwnCar=owncar,
            NumberOfChildrenVisiting=numberofchildrenvisiting,
            Designation=designation,
            MonthlyIncome=monthlyincome
        )
        st.success(prediction_text)
