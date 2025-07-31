import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf 
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder

model =tf.keras.models.load_model('model.h5')

with open('label_gender.pkl','rb') as file:
    label_gender=pickle.load(file)

with open('geo_encoder.pkl','rb') as file:
    geo_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

st.title('Customer Churn Prediction')

geography=st.select_slider('Geography',geo_encoder.categories_[0])
gender=st.select_slider('Gender',label_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
estimated_salary=st.number_input('Estimated Salary')
credit_score=st.number_input('Credit Score')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Num of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is active Member',[0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = geo_encoder.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=geo_encoder.get_feature_names_out(['Geography']))
input_df=pd.concat([input_data.reset_index(drop=True),geo_df],axis=1)
input_scaled=scaler.transform(input_df)

pred=model.predict(input_scaled)
st.write(pred[0][0])

if pred[0][0]>0.5:
    st.write('The person is likely to churn')
else:
    st.write("the person is not likely to churn")
