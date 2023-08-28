import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

# Load your trained XGBoost model (assuming you have a model named 'xgb_model.pkl')
model = xgb.Booster()
model.load_model('acdf_xgb.json')

def predict(Age, Ethnicity, Gender, Hypertension, Diabetes, Smoking, Alcohol, COPD, IMD_Bin):
    # Convert categorical variables to numeric
    ethnicity_dict = {'White British': 0, 'Other white': 1, 'Asian': 2, 'Black/African': 3, 'Other Background': 4, 'Mixed Background': 5}
    Ethnicity = ethnicity_dict[Ethnicity]
    Gender = 1 if Gender == 'Male' else 0
    Hypertension = 1 if Hypertension == 'Yes' else 0
    Diabetes = 1 if Diabetes == 'Yes' else 0
    Smoking = 1 if Smoking == 'Yes' else 0
    Alcohol = 1 if Alcohol == 'Yes' else 0
    COPD = 1 if COPD == 'Yes' else 0
    IMD_Bin = 1 if IMD_Bin > 5 else 0
    data = pd.DataFrame([[Age, Ethnicity, Gender, Hypertension, Diabetes, Smoking, Alcohol, COPD, IMD_Bin]], columns=['Age', 'Ethnicity', 'Gender', 'Hypertension', 'Diabetes', 'Smoking', 'Alcohol', 'COPD', 'IMD_Bin'])
    data_dmatrix = xgb.DMatrix(data)
    prediction = model.predict(data_dmatrix)
    return prediction

st.sidebar.title('Navigation')
selected_page = st.sidebar.selectbox("Choose a page", ["Information", "Prediction App"])

if selected_page == "Information":
    st.title("Information Page")
    st.write("""
    ## Project Overview

    The primary goal of this project is to predict the postoperative length of stay (LOS) for patients who have undergone Anterior Cervical Discectomy and Fusion (ACDF) surgery. Traditionally, predicting LOS for surgical patients has relied on complex pre-operative, peri-operative and postoperative clinical data, often making the process of such predictions labour intensive, prone to subjectivity and time-consuming.

    With this project, for the first time, we shift our attention towards leveraging non-clinical, baseline patient data including:
    
    - Basic demographics such as Age, Sex and Ethnicity.
    - Patient's comorbidities like Hypertension, Diabetes, Smoking habits, Alcohol consumption and Chronic Obstructive Pulmonary Disease (COPD).
    - Socio-economic deprivation indices such as the Index of Multiple Deprivations.
    
    By focusing on these parameters, our aim is to make LOS predictions more proactive and highlight parameters that can affect outcomes even before the surgery takes place. Nevertheless, this is one piece of the puzzle and we hope to expand our model to include both pre- and peri-operative clinical data in the future.

    ## Model Performance

    Our trained predictive model has exhibited promising results with a pooled F1 score of 73%. The model has been trained on a dataset of 1,533 patients and tested on 500 unseen patietns who underwent ACDF surgery at a tertiary centre in the UK between 2012 and 2022. The optimal ML model was an XGBoost model, outperforming KNN and RF models on all metrics.

    ### LOS Outcome Categories

    The predicted LOS is categorized into three distinct outcomes:
    
    - **Short LOS:** Less than 2 days (Labelled as 0)
    - **Long LOS:** Between 2 to 3 days (Labelled as 1)
    - **Extended LOS:** More than 3 days (Labelled as 2)

    ## Importance

    By predicting LOS using non-clinical data, we move towards a proactive healthcare model that optimizes patient care and preempts challenges, all by analyzing simple pre-operative baseline chasractersitics of a patient.
    
    ## XGBoost Model
    Utilize the sidebar to navigate to the prediction app and try out the model yourself!

    ## Not for Clinical use. For Research use only. 
    """)

elif selected_page == "Prediction App":
    st.title('XGBoost Model Prediction')

    # Collect input variables
    #Age = st.number_input('Age', min_value=16, max_value=100, step=1)
    Age = st.slider('Age', 16, 100, 50)
    Ethnicity = st.selectbox('Ethnicity', ['White British', 'Other white', 'Asian', 'Black/African', 'Other Background', 'Mixed Background'], help="Asian patients have the longest LOS compared to all other ethnic groups.")
    Gender = st.selectbox('Sex', ['Male', 'Female'], help="Male patients have a shorter LOS than Female patients.") # Assuming 'Yes' means Male and 'No' means Female
    Hypertension = st.selectbox('Hypertension', ['Yes', 'No'], help =   "Patients without Hypertension have a shorter LOS than patients with Hypertension.")
    Diabetes = st.selectbox('Diabetes', ['Yes', 'No'], help="Patients without Diabetes have a shorter LOS than patients with Diabetes.")
    Smoking = st.selectbox('Smoking', ['Yes', 'No'], help="Patients who smoke have a shorter LOS than patients who do not smoke.")
    Alcohol = st.selectbox('Alcohol', ['Yes', 'No'], help="Patients who consume alcohol have a shorter LOS than patients who do not consume alcohol.")
    COPD = st.selectbox('COPD', ['Yes', 'No'], help="Patients with COPD have a longer LOS than patients without COPD.")
    IMD_Bin = st.number_input('Index of Multiple Deprivation (IMD) Decile', min_value=1, max_value=10, step=1, help="Patients from more deprived areas have a longer LOS than patients from less deprived areas.")

    # Predict
    if st.button('Predict'):
        #dmatrix = xgb.DMatrix(features)
        #prediction = model.predict(dmatrix)
        price = predict(Age, Ethnicity, Gender, Hypertension, Diabetes, Smoking, Alcohol, COPD, IMD_Bin)
        prediction_label = int(price[0])  # Assuming model returns a class label as output

        # Map the label to the corresponding string
        label_map = {
            0.0: "Short LOS (<2 days)",
            1.0: "Long LOS (2-3 days)",
            2.0: "Extended LOS (>3 days)"
            }

        st.write('Predicted Length of Stay:', label_map[prediction_label])
