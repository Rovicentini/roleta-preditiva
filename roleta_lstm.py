import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px

# Inicializar variáveis de sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

if "modelo" not in st.session_state:
    st.session_state.modelo = None

if "input_numero" not in st.session_state:
    st.session_state.input_numero = ""

# Função para treinar o modelo
def treinar_modelo():
    if len(st.session_state.historico) < 10:
        return  # Treina só com pelo menos 10 números
    dados = np.array(st.session_state.historico[-20:])  # pegar últimos 20 números
    dados = dados.reshape((1, len(dados), 1))
    labels = dados[:, 1:, :]
    dados = dados[:, :-1, :]

    modelo = Sequential()
    modelo.add(LSTM(50, activation='relu', input_shape=(dados.shape[1], dados.shape[2])))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mse')

    modelo.fit(dados, labels, epochs=100, verbose=0)
    st.session_state.modelo = modelo

# Função para prever próximo número
def prever_proximo():
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None
    entrada = np.array(st.session_state.historico[-19:]).reshape((1, 19, 1))
    pred = st.session_state.modelo.predict(entrada, verbose=0)
    pred_num = int(np.round(pred[0, 0]))
    if pred_num < 0:
        pred_num = 0
    if pred_num > 36:
        pred_num = 36
    return pred_num

# Função para processar submissão
def adicionar_numero():
    try:
        numero = int(st.session_state.input_numero)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
            treinar_modelo()
            st.session_state.input_numero = ""  # limpa o campo
        else:
            st.error("Número inválido! Insira um número entre 0 e 36.")
    except ValueError:
        st.error("Por favor, insira um número válido.")

# Layout moderno com CSS customizado
st.markdown(
    """
    <style>
    /* Fundo degradê suave */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #e0e0e0;
        font-family: 'Roboto', sans-serif;
    }
    /* Título centralizado e estilizado */
    .titulo {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        color: #fff;
        letter-spacing: 0.1em;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
    }
    /* Container do input */
    .input-container {
        max-width: 320px;
        margin: 0 auto 2rem auto;
    }
    /* Campo de texto com bordas arredondadas e padding */
    div.stTextInput > label > div > input {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        font-size: 1.2rem;
        border: none;
        outline: none;
        box-shadow: 0 0 8px #fff5;
        background-color: #fff2;
        color: #333;
        transition: box-shadow 0.3s ease;
    }
    div.stTextInput > label > div > input:focus {
        box-shadow: 0 0 14px #fff;
    }
    /* Botão estilizado */
    div.stButton > button {
        background-color: #ff6f61;
        color: white;
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        font-size: 1.1rem;
        transition: background-color 0.3s ease;
        margin-top: 0.5rem;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #ff4a39;
    }
    /* Container histórico */
    .historico-container {
        max-width: 600px;
        margin: 0 auto;
        background-color: #ffffff22;
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 0 15px #fff3;
        font-size: 1.2rem;
        color: #ddd;
    }
    /* Texto da previsão */
    .previsao {
        text-align: center;
        font-size: 1.6rem;
        margin-top: 2rem;
        font-weight: 700;
        color: #ffe;
        text-shadow: 1px 1px 6px #0008;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="titulo">Roleta Preditiva</div>', unsafe_allow_html=True)

with st.form(key="input_form"):
    with st.container():
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        numero_input = st.text_input(
            "Digite o número sorteado (0-36):",
            key="input_numero",
            max_chars=2,
            placeholder="Exemplo: 17",
            label_visibility="visible",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    submit_btn = st.form_submit_button("Registrar")

if submit_btn:
    adicionar_numero()

# Mostrar histórico
st.markdown('<div class="historico-container">', unsafe_allow_html=True)
st.subheader("Histórico dos números inseridos")
if len(st.session_state.historico) == 0:
    st.write("Nenhum número inserido ainda.")
else:
    st.write(st.session_state.historico)
st.markdown("</div>", unsafe_allow_html=True)

# Mostrar previsão
proximo = prever_proximo()
if proximo is not None:
    st.markdown(f'<div class="previsao">Próximo número previsto pela IA: <strong>{proximo}</strong></div>', unsafe_allow_html=True)
else:
    st.info("Insira pelo menos 10 números para começar a prever.")
