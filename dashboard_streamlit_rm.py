import streamlit as st
import pandas as pd
import numpy as np

siteHeader = st.container()
with siteHeader:
    st.title('Modelo de evaluación de ingresos')
    st.markdown(""" En este modelo buscamos encontrar cuáles son las variables más relevantes
    para determinar el ingreso de una persona. """) 

dataExploration = st.container()
with dataExploration:
    st.header('Dataset: Ingresos')
    st.text(""" Este dataset corresponde a una transformación del set de datos oficiales 
    provenientes de la siguiente fuente: 
    https://raw.githubusercontent.com/fridaruh/pipeline_dt/master/in/income.csv
    """)

df = pd.read_csv('https://raw.githubusercontent.com/fridaruh/pipeline_dt/master/in/income.csv')

st.text("")

num_reg = st.slider('¿Cuántos registros desea ver?', min_value=0, max_value=20, value=10, step=2)

st.write(df.drop(['Unnamed: 0', 'fnlwgt'], axis=1).head(num_reg))

marital_status = pd.DataFrame(df['marital-status'].value_counts())

st.text("")
st.bar_chart(marital_status)

st.markdown(""" <style>
    .main {
    background-color: #AF9EC;
}
</style> """, unsafe_allow_html=True)
