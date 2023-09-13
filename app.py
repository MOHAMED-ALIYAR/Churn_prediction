import streamlit as st
import pandas as pd
import numpy as np
import pickle


#title
st.markdown("<h1 style='text-align: center; color: black;'>CHURN PREDICTION </h1>", unsafe_allow_html=True)

#load the pickle file

#scale = pickle.load(open('scale.pkl','rb'))
model=tf.keras.models.load_model('keras2')




gender_option= [0,1]
geo_option=[0,1,2]
card_option=[0,1]
member_option=[0,1]

col1,col2,col3=st.columns([5,2,5])
with col1:
    Cedit_score=st.number_input('Enter the Credit Score')

    Geography=st.selectbox('Country(France:0  Germany:1 Spain:2)',geo_option,key=1)

    Gender=st.selectbox('Gender(male:0 Female:1)',gender_option,key=2)

    Age=st.number_input('Enter the Age')

    Tenure=st.number_input('Enter the Tenure')
with col3:
    Balance=st.number_input('Enter the Balance')

    Number_product=st.number_input('Enter the No.of Product ')

    Card=st.selectbox('Card(Yes:1  No:0)',card_option,key=3)

    Member=st.selectbox('Active Member Status(Yes:1  No:0)',member_option,key=4)

    Salary=st.number_input('Enter the Salary')



button=st.button('PREDICT')

if button:
    a=(np.array([[Cedit_score,Geography,Gender,Age,Tenure,Balance,Number_product,Card,Member,Salary]]).reshape(1,-1))

    b=model.predict(a)
    if b>0.5:
        st.markdown("<h1 style='text-align: center; color: black;'>EXIT </h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 style='text-align: center; color: black;'>NOT EXIT</h1>", unsafe_allow_html=True)
