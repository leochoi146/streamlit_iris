import streamlit as st
import plotly.express as px
import numpy as np
import pickle
import load_data
from load_data import load_iris
import time


st.title('My Awesome Flower Predictor')
st.header('We predict Iris types')
st.subheader('No joke')

# with st.spinner(text='Loading'):
#     time.sleep(5)
#     st.success('Done')

#load data
df_iris = load_iris()

file = st.file_uploader('File uploader')

st.color_picker('pick a color')

st.plotly_chart(px.scatter(df_iris, x='sepal_width', y='sepal_length'))

show_df = st.checkbox('Do you want to see the data?')

if show_df:
    df_iris

#get user flower input

s_l = st.number_input('Input the Sepal Length')
s_w = st.number_input('Input the Sepal Width')
p_l = st.number_input('Input the Petal Length')
p_w = st.number_input('Input the Petal Width')

user_values = np.array([s_l, s_w, p_l, p_w])

with open('saved-iris-model-2.pkl', 'rb') as f:
    model = pickle.load(f)
    
with st.echo():
    # this is my code
    prediction = model.predict(user_values.reshape(1, -1))

t = type(prediction)

st.write(t)

st.subheader(f'The model predicts: {prediction[0]}')

st.balloons()

st.beta_container()


col1, col2, col3 = st.beta_columns(3)
col1.subheader('Columnisation')
col2.subheader('Columnisation')
col3.subheader('Columnisation')
with col1:
    'I am printing things'
with col2:
    df_iris
with col3:
    st.subheader('cool stuff')
    

st.beta_expander('Expander')
with st.beta_expander('Expander'):
    st.write('Juicy deets')