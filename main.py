import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



header = st.beta_container()
dataset = st.beta_container()
feature = st.beta_container()
modeltrain = st.beta_container()

with header:
    st.title("Welcome To My First Streamlit Project")
    st.text('In this project I look into the transaction of taxies in NYC')


with dataset:
    st.header('NYC taxi dataset.')
    st.text('I found this dataset on Kaggle')

    taxi_data =pd.read_csv('data/nyctaxi.csv')
    st.write(taxi_data.head(5))


    st.subheader('Pickup Location distribution on NYC Data set')
    pickloc_dist = pd.DataFrame(taxi_data['start_station_id'].value_counts()).head(20)
    st.bar_chart(pickloc_dist)




with feature:
    st.header('The features I have created.')
    st.markdown('* **First Feature: **')
    st.markdown('* **Second Feature: ** ')





with modeltrain:
    st.header(' Time to train the Model')
    st.text('Here you get to choose the hyperparameter of the model and see how to perform changes.')
    sel_col, disp_col = st.beta_columns(2)

    max_depth = sel_col.slider('What should be the max depth of the column', min_value=5, max_value=100, value=20,step=10)
    n_estimators = sel_col.selectbox('HOw many trees should be there?', options= [100,200,300,'No Limit'], index = 0)
    input_feature = sel_col.text_input('Which feature should be used as the input feature?','start_station_id')


    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x = taxi_data[[input_feature]]
    y = taxi_data[['tripduration']]

    regr.fit(x, y)
    prediction = regr.predict(y)


    disp_col.subheader('Mean absolute error of the model is: ')
    disp_col.write(mean_absolute_error(y, prediction))
    disp_col.subheader('Mean square error of the model is: ')
    disp_col.write(mean_squared_error(y, prediction))
    disp_col.subheader('R squared scoe of the model is: ')
    disp_col.write(r2_score(y, prediction))
