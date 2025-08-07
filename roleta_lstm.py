import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Inicializar variáveis na sessão
if "resultados" not in st.session_state:
    st.session_state.resultados = []
if "acertos" not in st.session_state:
    st.session_state.acertos = []
if "perdas" not in st.session_state:
    st.session_state.perdas = []

st.title("IA Preditiva para Roleta - Análise Probabilística")

# Inserção de novo número
with st.form(key="formulario"):
    novo_resultado = st.number_input("Insira o novo número sorteado (0 a 36):", min_value=0, max_value=36, step=1, key="input")
    submit_button = st.form_submit_button(label="Registrar")

if submit_button:
    st.session_state.resultados.append(novo_resultado)
    st.session_state.input = 0  # Limpa o campo de entrada
    st.experimental_rerun()

# Visualizar resultados
st.subheader("Histórico de Resultados")
st.write(st.session_state.resultados)

# Função para gerar dados para o modelo
WINDOW_SIZE = 5

def preparar_dados(resultados):
    scaler = MinMaxScaler()
    resultados_np = np.array(resultados).reshape(-1, 1)
    resultados_normalizados = scaler.fit_transform(resultados_np)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(resultados_normalizados)):
        X.append(resultados_normalizados[i-WINDOW_SIZE:i, 0])
        y.append(resultados_normalizados[i, 0])
    return np.array(X), np.array(y), scaler

# Treinar modelo se houver dados suficientes
if len(st.session_state.resultados) > WINDOW_SIZE + 1:
    X, y, scaler = preparar_dados(st.session_state.resultados)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=1, verbose=0)

    ultimos_resultados = st.session_state.resultados[-WINDOW_SIZE:]
    entrada = scaler.transform(np.array(ultimos_resultados).reshape(-1, 1)).reshape(1, WINDOW_SIZE, 1)
    previsao_normalizada = model.predict(entrada)
    previsao = scaler.inverse_transform(previsao_normalizada)[0][0]

    numero_previsto = int(round(previsao)) % 37

    # Vizinhança para avaliar acerto
    vizinhos = 1  # Pode aumentar para 2 ou 3
    numeros_vizinhos = [(numero_previsto + i) % 37 for i in range(-vizinhos, vizinhos+1)]

    st.subheader("Previsão da IA")
    st.write(f"Próximo número previsto: **{numero_previsto}**")
    st.write(f"Considerando vizinhos: {numeros_vizinhos}")

    if len(st.session_state.resultados) > WINDOW_SIZE + 2:
        ultimo_real = st.session_state.resultados[-1]
        if ultimo_real in numeros_vizinhos:
            st.session_state.acertos.append(1)
            st.session_state.perdas.append(0)
            st.success("Acerto!")
        else:
            st.session_state.acertos.append(0)
            st.session_state.perdas.append(1)
            st.error("Erro na previsão")

    # Gráfico de desempenho
    st.subheader("Desempenho da IA")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=np.cumsum(st.session_state.acertos), mode='lines', name='Acertos'))
    fig.add_trace(go.Scatter(y=np.cumsum(st.session_state.perdas), mode='lines', name='Erros'))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Insira pelo menos 6 números para iniciar as previsões.")
