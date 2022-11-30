import streamlit as st
import pickle
import pandas as pd
import numpy as np

#Importamos  los modelos
pipe = pickle.load(open('../models/model.pkl','rb'))
df = pickle.load(open('../models/df.pkl','rb'))

st.title("Predictor de Migración de Clientes")

edad = st.number_input('Edad del Cliente')

ant_cliente = st.number_input("Antiguedad del Cliente (Meses)")

n_lineas = st.number_input("Número de Lineas del Cliente")

gamma_equipo = st.selectbox('Gamma del Equipo', df['vgamma_c'].unique())

seguro = st.selectbox('Tiene seguro el equipo', ['si','no'])

seguro=1 if seguro=='si' else 0

cambios = st.number_input("Cantidad de cambios de plan")

cons_KB = st.number_input("KBs Consumidos")

cortes = st.number_input("Cantidad de cortes por deuda")

if st.button("Calcular"):
    query = np.array([edad, ant_cliente, n_lineas, gamma_equipo,seguro, cambios, cons_KB, cortes])
    prediction = pipe.predict(pd.DataFrame(columns=df.drop(columns='vtarge_c').columns, data=query.reshape(1,-1)))
    if prediction[0] == 1:
        st.title("Es probable que el cliente se quede con nosotros")
    else:
        st.title("Es probable que el cliente migre a otra operadora")
