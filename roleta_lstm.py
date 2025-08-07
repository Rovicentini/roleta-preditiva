# Roleta IA Robusta com LSTM, TensorFlow, Análise de Tendência, e Visualização
# Autor: Rodrigo + ChatGPT
# Requisitos: pip install streamlit tensorflow scikit-learn pandas matplotlib numpy

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide")
st.title("🎯 IA Avançada para Roleta Europeia")

# --- VARIÁVEIS GLOBAIS ---
NUM_TOTAL = 37  # Números da Roleta Europeia: 0 a 36
SEQUENCIA_ENTRADA = 10  # Quantos números analisar por entrada

if 'historico' not in st.session_state:
    st.session_state.historico = []

if 'resultados' not in st.session_state:
    st.session_state.resultados = []

if 'vizinhanca' not in st.session_state:
    st.session_state.vizinhanca = 0

if 'modelo_treinado' not in st.session_state:
    st.session_state.modelo_treinado = False

# --- FUNÇÕES ---
def adicionar_numero(numero):
    try:
        n = int(numero)
        if 0 <= n < NUM_TOTAL:
            st.session_state.historico.append(n)
    except:
        pass

def get_dados_normalizados():
    dados = np.array(st.session_state.historico).reshape(-1, 1)
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(dados)
    return dados_norm, scaler

def criar_dados_sequenciais(dados_norm):
    X, y = [], []
    for i in range(len(dados_norm) - SEQUENCIA_ENTRADA):
        X.append(dados_norm[i:i+SEQUENCIA_ENTRADA])
        y.append(dados_norm[i+SEQUENCIA_ENTRADA])
    return np.array(X), np.array(y)

def treinar_modelo():
    dados_norm, scaler = get_dados_normalizados()
    X, y = criar_dados_sequenciais(dados_norm)
    if len(X) < 5:
        return None, None

    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(SEQUENCIA_ENTRADA, 1)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    st.session_state.modelo_treinado = True
    return model, scaler

def prever_proximo(modelo, scaler):
    if not modelo:
        return []
    ultimos = st.session_state.historico[-SEQUENCIA_ENTRADA:]
    if len(ultimos) < SEQUENCIA_ENTRADA:
        ultimos = [0] * (SEQUENCIA_ENTRADA - len(ultimos)) + ultimos
    entrada = np.array(ultimos).reshape(-1, 1)
    entrada_norm = scaler.transform(entrada)
    entrada_norm = entrada_norm.reshape(1, SEQUENCIA_ENTRADA, 1)
    pred_norm = modelo.predict(entrada_norm, verbose=0)
    pred = scaler.inverse_transform(pred_norm)
    valor = int(np.round(pred[0][0]))

    sugestoes = [(valor + i) % NUM_TOTAL for i in range(-st.session_state.vizinhanca, st.session_state.vizinhanca + 1)]
    sugestoes = sorted(list(set([n % NUM_TOTAL for n in sugestoes])))
    return sugestoes

def calcular_performance():
    acertos = 0
    total = len(st.session_state.resultados)
    for res in st.session_state.resultados:
        if res['acerto']:
            acertos += 1
    return acertos, total - acertos

def exibir_grafico_performance():
    acertos, erros = calcular_performance()
    fig, ax = plt.subplots()
    ax.bar(['Acertos', 'Erros'], [acertos, erros])
    st.pyplot(fig)

# --- SIDEBAR ---
st.sidebar.header("🎛️ Configurações")
viz = st.sidebar.slider("Número de vizinhos (acertos)", 0, 5, st.session_state.vizinhanca)
st.session_state.vizinhanca = viz

if st.sidebar.button("🔁 Reiniciar Tudo"):
    st.session_state.historico = []
    st.session_state.resultados = []
    st.session_state.modelo_treinado = False

# --- INTERFACE PRINCIPAL ---
st.subheader("🎰 Inserir Número da Roleta")
def adicionar_numero_callback():
    numero = st.session_state.entrada_numero
    if numero != "" and st.session_state.get("ultima_entrada") != numero:
        adicionar_numero(numero)
        st.session_state.ultima_entrada = numero
        st.session_state.entrada_numero = ""  # limpa o campo

st.text_input("Digite o número sorteado (0 a 36):", key="entrada_numero", on_change=adicionar_numero_callback)

# --- EXIBIR HISTÓRICO ---
st.subheader("📜 Histórico")
if st.session_state.historico:
    st.markdown(" ".join([f"**{num}**" for num in st.session_state.historico[::-1]]))
else:
    st.info("Nenhum número inserido ainda.")

# --- TREINAR E PREVER ---
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 1:
    model, scaler = treinar_modelo()
    sugestoes = prever_proximo(model, scaler)

    st.subheader("📈 Sugestão de Apostas da IA")
    st.write("**Sugestão de números:**", sugestoes)

    # Comparar com último número
    if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 2:
        ultimo_real = st.session_state.historico[-1]
        acerto = ultimo_real in sugestoes
        st.session_state.resultados.append({
            'real': ultimo_real,
            'previsto': sugestoes,
            'acerto': acerto
        })

        st.write(f"**Último número real:** {ultimo_real} | **Acertou?** {'✅' if acerto else '❌'}")
        acertos, erros = calcular_performance()
st.sidebar.markdown(f"✅ Acertos: {acertos} | ❌ Erros: {erros} | Total: {acertos + erros}")
else:
    st.info("Insira ao menos 11 números para iniciar a previsão com IA.")



