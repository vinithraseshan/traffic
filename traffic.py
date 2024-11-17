import streamlit as st
import pandas as pd
import pickle

with open("training_columns.pkl", "rb") as f:
    train_columns = pickle.load(f)

traffic = pd.read_csv('Traffic_Volume.csv')
traffic['date_time'] = pd.to_datetime(traffic['date_time'])
traffic['month'] = traffic['date_time'].dt.month
traffic['weekday'] = traffic['date_time'].dt.weekday
traffic['hour'] = traffic['date_time'].dt.hour

traffic.drop(columns=['date_time'], inplace=True)


mapie_pickle = open('mapie.pickle', 'rb') 
mapie_model = pickle.load(mapie_pickle) 
mapie_pickle.close()

st.header("Traffic Volume Predictor ")
st.write('Predict traffic volume using our app.')
st.image('traffic_image.gif')

alpha = st.slider("Select alpha value for prediction intervals", min_value=0.01, max_value=0.50, value=0.1)
confidence = 1-alpha

st.header("Predicting Traffic Volume..")

st.sidebar.image('traffic_sidebar.jpg')
st.sidebar.markdown("**Input Features**")
st.sidebar.write("You can either upload your data file or manually enter features")


with st.sidebar.expander("Option 1: Upload CSV file"):
    st.text("Upload a CSV file containing the traffic details")
    userinput = st.file_uploader("Choose a CSV file", type='csv')
    st.markdown("**Sample Data Format for Upload**")
    st.dataframe(traffic.head(5).drop(columns = ['traffic_volume']))
    st.write("Ensure your uploaded file has the same column names and data types as shown above")


with st.sidebar.expander("Option 2: Fill out Form"):
    with st.form("user_input"):
        holiday = st.selectbox(label = 'Holiday', options = traffic['holiday'].unique())
        temp = st.number_input('Temperature', min_value = traffic['temp'].min(), max_value = traffic['temp'].max(),step = 0.01)
        rain = st.number_input('Rain', min_value = traffic['rain_1h'].min(), max_value = traffic['rain_1h'].max(),step = 0.1)
        snow = st.number_input('Snow', min_value = traffic['snow_1h'].min(), max_value = traffic['snow_1h'].max(),step = 0.1)
        clouds = st.number_input('Clouds', min_value = traffic['clouds_all'].min(), max_value = traffic['clouds_all'].max(),step = 1)
        weather = st.selectbox(label = 'Weather', options = traffic['weather_main'].unique())
        month = st.selectbox(label = 'Month', options = traffic['month'].unique())
        weekday = st.selectbox(label = 'Weekday', options = traffic['weekday'].unique())
        hour = st.selectbox(label = 'Hour', options = traffic['hour'].unique())
        button = st.form_submit_button('Submit Form Data')


if userinput is not None:  
    st.success("CSV file successfully uploaded")
    user_data = pd.read_csv(userinput)
    features = user_data.copy()
    features = pd.get_dummies(features, drop_first=True) 
    missing_cols = set(train_columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0  
    features = features[train_columns]
    st.write(f"**Prediction result with {confidence*100}% Prediction Interval**")
    y_pred, intervals = mapie_model.predict(features, alpha=alpha)
    user_data['Predicted Traffic Volume'] = y_pred
    user_data['Lower Volume Limit'] = [max(0, interval[0]) for interval in intervals]  
    user_data['Upper Volume Limit'] = [interval[1] for interval in intervals] 
    st.write(user_data)

elif button:  
    st.success("Form submitted successfully")
    encode_df = pd.DataFrame({
        'holiday': [holiday],
        'temp': [temp],
        'rain_1h': [rain],
        'snow_1h': [snow],
        'clouds_all': [clouds],
        'weather_main': [weather],
        'month': [month],
        'weekday': [weekday],
        'hour': [hour]
    })

    encode_df = pd.get_dummies(encode_df, drop_first=True)
    missing_cols = set(train_columns) - set(encode_df.columns)
    for col in missing_cols:
        encode_df[col] = 0  
    features = encode_df[train_columns]
    
    prediction, intervals = mapie_model.predict(features, alpha=alpha)
    pred_value = prediction[0] 
    lower_limit = max(0,intervals[0][0])  
    upper_limit = intervals[0][1]  
    st.metric(label="Predicted Volume", value=f"{pred_value:.2f}")
    st.write(f"**{confidence*100}% Confidence Interval**: [{lower_limit}, {upper_limit}]")
    
else:
    st.error("Please upload a file or fill out the form to make a prediction.")


st.write("### Model Insights")

tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", 
                            "Histogram of Residuals", 
                            "Predicted Vs. Actual", 
                            "Coverage Plot"])
with tab1:
    st.write("### Feature Importance")
    st.image('feature_imp.svg')
    st.caption("Relative importance of features in prediction.")
with tab2:
    st.write("### Histogram of Residuals")
    st.image('residual_plot.svg')
    st.caption("Distribution of residuals to evaluate prediction quality.")
with tab3:
    st.write("### Plot of Predicted Vs. Actual")
    st.image('pred_vs_actual.svg')
    st.caption("Visual comparison of predicted and actual values.")
with tab4:
    st.write("### Coverage Plot")
    st.image('coverage.svg')
    st.caption("Range of predictions with confidence intervals.")
