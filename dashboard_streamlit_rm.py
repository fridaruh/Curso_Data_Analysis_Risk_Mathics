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

df = df.drop(['Unnamed: 0', 'fnlwgt','native-country'], axis=1)

marital_status = pd.DataFrame(df['marital-status'].value_counts())

st.text("")

st.markdown('**Marital Status**')

st.bar_chart(marital_status)

#Convierto a variables dummies las variables categóricas de df
df_2 = pd.get_dummies(df, columns=['workclass','education','marital-status','occupation','relationship','race','sex'])

#Asigno a Y nuestra variable a predecir

Y = df_2['income_bi']

#Asigno a X nuestras variables predictoras

modelTraining = st.container()
with modelTraining:
    st.header('Entrenamiento del modelo')
    st.text('En esta sección podrás hacer una selección de los parámetros del modelo')

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import roc_auc_score

semillas = [0, 55, 99]

semilla_aleatoria = st.radio('¿Qué semilla desea utilizar?',semillas)

X = df_2.drop(['income','income_bi'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=semilla_aleatoria)

profundidad = st.slider('¿Cuántas ramas desea utilizar?', min_value=1, max_value=10, value=3, step=1)

t = tree.DecisionTreeClassifier(max_depth=profundidad)

model = t.fit(x_train, y_train)

prediction = model.predict(x_test)
score_test = model.score(x_test, y_test)
score_train = model.score(x_train, y_train)

st.header('Resultados del entrenamiento')
st.text('Score de entrenamiento: ' + str(score_train))
st.text('Score de test: ' + str(score_test))



st.markdown(""" <style>
    .main {
    background-color: #AF9EC;
}
</style> """, unsafe_allow_html=True)
