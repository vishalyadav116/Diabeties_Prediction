#!/usr/bin/env python
# coding: utf-8
pip install python
pip install pandas
pip install numpy
pip install sklearn
pip install matplotlib
pip install seaborn

#import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data = pd.read_csv('diabetes.csv')
# =============================================================================
# data.head()
# data.shape
# data.info()
# data.describe()
# data.isnull().sum()
# =============================================================================

# EDA
# data.Outcome.value_counts()

# Target Ratio
# data.Outcome.value_counts().plot(kind='bar')
# plt.show()

# View Columns
# data.columns

st.title('Diabetes Checkup')
st.header('To predict whether you are Diabetic or not, follow the points :')
st.text('1. Click on top left arrow to open the sidebar.\n2. Slide the values on the lines.\n3. Look for the result.')


x=data.drop('Outcome',axis=1)
y=data.Outcome

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0 ,stratify=y)
# print('Shape of x_train : ',x_train.shape)
# print('Shape of x_test : ',x_test.shape)
# print('Shape of y_train : ',y_train.shape)
# print('Shape of y_test : ',y_test.shape)

# print(y_train.value_counts())
# print(y_test.value_counts())

# Model Building
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
predicted_val = rf.predict(x_test)
acc_score=accuracy_score(y_test,predicted_val)


def user_report():
    pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    glucose = st.sidebar.slider('Glucose',0,200,100)
    bp = st.sidebar.slider('Blood Pressure',0,122,70)
    skinthickness = st.sidebar.slider('Skin Thickness',0,100,20)
    insulin = st.sidebar.slider('Insulin',0,846,79)
    bmi = st.sidebar.slider('BMI',0,67,20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function',0.0,2.4,0.47)
    age = st.sidebar.slider('Age',21,88,33)
    
    user_report = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age }
    report_data = pd.DataFrame(user_report,index=[0])
    return report_data

user_data = user_report()

user_result = rf.predict(user_data)

st.subheader('Your Result : ')
output=''
if user_result[0]==0:
    output='You are non-diabetic'
else:
    output='You are diabetic'
st.write(output)

st.subheader('Prediction Accuracy : ')
st.write(acc_score)

if st.checkbox('View population data'):
    st.write(data.head(100))
    st.text('Outcome value "1" depicts you are not healthy and "0" depicts you are healthy.')


if st.checkbox('View Categories affecting Diabeties'):
    for ele in data.columns:
        st.text(ele)

fig=plt.figure(figsize=[20,16])
for i in range(len(data.columns)):
    plt.subplot(3,3,i+1)
    sns.distplot(data[data.columns[i]],color='green')
    plt.title(data.columns[i],fontdict={'fontsize':17,'color': 'red'})
    plt.tight_layout()
if st.button('Click to see graphs of all the Categories'):
    st.pyplot(fig)
    
