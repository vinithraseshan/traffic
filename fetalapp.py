import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import warnings
warnings.filterwarnings("ignore")

fetal_health = pd.read_csv('fetal_health.csv').drop(columns = 'fetal_health')

with open('adaboost.pickle', 'rb') as ada_pickle:
    ada_model = pickle.load(ada_pickle)

with open('decision_tree.pickle', 'rb') as dt_pickle:
    dt_model = pickle.load(dt_pickle)

with open('random_forest.pickle', 'rb') as rf_pickle:
    rf_model = pickle.load(rf_pickle)

with open('voting.pickle', 'rb') as voting_pickle:
    voting_model = pickle.load(voting_pickle)

st.sidebar.header("Fetal Health Feature Input")
userinput = st.sidebar.file_uploader("Choose a CSV file", type='csv')
st.sidebar.success("Ensure your uploaded file has the same column names and data types as shown above")

model = st.sidebar.radio(
    "Select model type",
    ["Random Forest", "Decision Tree", "AdaBoost", "Soft Voting"],
    index=None,
)

st.sidebar.success(f"You selected: {model}")

st.sidebar.markdown("**Sample Data Format for Upload**")
st.sidebar.dataframe(fetal_health.head(5))

st.header("Fetal Health Classification: A Machine Learning App")
st.image('fetal_health_image.gif')
st.write("Utilize our advanced Machine Learning application to predict fetal health classifications.")

if userinput is not None:
    st.success("CSV file uploaded successfully")
    st.write(f"**Predicting Fetal Health Class Using '{model}' Model**")
    user_data = pd.read_csv(userinput)

    #used chat gpt to make function
    def highlight_prediction(row):
        styles = [''] * len(row) 
        if row['Predicted Class'] == 1:
            styles[row.index.get_loc('Predicted Class')] = 'background-color: lime'
        elif row['Predicted Class'] == 2:
            styles[row.index.get_loc('Predicted Class')] = 'background-color: yellow'
        elif row['Predicted Class'] == 3:
            styles[row.index.get_loc('Predicted Class')] = 'background-color: orange'
        return styles

    if model == 'Random Forest':
        features = user_data.drop(columns=['Prediction Class'], errors='ignore')  # Or specify explicitly
        user_data['Predicted Class'] = rf_model.predict(features)
        user_data['Prediction Probabilities'] = rf_model.predict_proba(features).max(axis=1)
        st.write("Predicted Data")
        st.dataframe(user_data.style.apply(highlight_prediction, axis=1))
    elif model == 'Decision Tree':
        features = user_data.drop(columns=['Prediction Class'], errors='ignore')  # Or specify explicitly
        user_data['Predicted Class'] = dt_model.predict(features)
        user_data['Prediction Probabilities'] = dt_model.predict_proba(features).max(axis=1)
        st.write("Predicted Data")
        st.dataframe(user_data.style.apply(highlight_prediction, axis=1))
    elif model == 'AdaBoost':
        features = user_data.drop(columns=['Prediction Class'], errors='ignore')  # Or specify explicitly
        user_data['Predicted Class'] = ada_model.predict(features)
        user_data['Prediction Probabilities'] = ada_model.predict_proba(features).max(axis=1) 
        st.write("Predicted Data")
        st.dataframe(user_data.style.apply(highlight_prediction, axis=1))
    elif model == 'Soft Voting':
        features = user_data.drop(columns=['Prediction Class'], errors='ignore')  # Or specify explicitly
        user_data['Predicted Class'] = voting_model.predict(features)
        user_data['Prediction Probabilities'] = voting_model.predict_proba(features).max(axis=1)
        st.write("Predicted Data")
        st.dataframe(user_data.style.apply(highlight_prediction, axis=1))
else:
    st.error('Please upload a file.')

tab1, tab2, tab3 = st.tabs(["Feature Importance", "Classification Report", "Confusion Matrix"])

if model == 'Decision Tree':
    with tab1:
        st.write("### Feature Importance")
        st.image('dtimp.svg')
        
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Classification Report")
        st.dataframe(pd.read_csv('df_report.csv'))
    with tab3:
        st.write("Confusion Matrix")
        st.image('dtcm.svg')
        
elif model == 'Random Forest':
    with tab1:
        st.write("### Feature Importance")
        st.image('rfimp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Classification Report")
        st.dataframe(pd.read_csv('rf_report.csv'))
    with tab3:
        st.write("Confusion Matrix")
        st.image('rfcm.svg')

elif model == 'AdaBoost':
    with tab1:
        st.write("### Feature Importance")
        st.image('adaimp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Classification Report")
        st.dataframe(pd.read_csv('ada_report.csv'))
    with tab3:
        st.write("Confusion Matrix")
        st.image('adacm.svg')

elif model == 'Soft Voting':
    with tab1:
        st.write("### Feature Importance")
        st.image('votingimp.svg')
        st.caption("Relative importance of features in prediction.")
    with tab2:
        st.write("### Classification Report")
        st.dataframe(pd.read_csv('voting_report.csv'))
    with tab3:
        st.write("Confusion Matrix")
        st.image('votingcm.svg')
