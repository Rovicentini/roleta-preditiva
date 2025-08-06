import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px

# ================================
# ESTADO INICIAL
# ================================
if "historico" not in st.session_state:
    st.session_state.historico = []
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "input_numero" not in st.session_state:
    st.session_state.input_numero = ""

# ================================
# FUNÇÕES
# ================================
def treinar_modelo():
    if len(st.session_state.historico) < 10:
        return
    dados = np.array(st.session_state.historico[-20:])
    dados = dados.reshape((1, len(dados), 1))
    labels = dados[:, 1:, :]
    dados = dados[:, :-1, :]

    modelo = Sequential()
    modelo.add(LSTM(50, activation='relu', input_shape=(dados.shape[1], dados.shape[2])))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mse')
    modelo.fit(dados, labels, epochs=100, verbose=0)
    st.session_state.modelo = modelo

def prever_proximo():
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None
    entrada = np.array(st.session_state.historico[-19:]).reshape((1, 19, 1))
    pred = st.session_state.modelo.predict(entrada, verbose=0)
    return int(np.clip(np.round(pred[0, 0]), 0, 36))

def adicionar_numero():
    try:
        numero = int(st.session_state.input_numero)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
            treinar_modelo()
            # Limpa input de forma segura usando st.text_input com key diferente
            st.session_state.input_numero = "" 
        else:
            st.error("Número inválido! Insira entre 0 e 36.")
    except ValueError:
        st.error("Por favor, insira um número válido.")

# ================================
# ESTILO PERSONALIZADO
# ================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Roboto', sans-serif;
        color: #f2f2f2;
    }
    .titulo {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        color: #fff;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
    }
    /* Input estilizado */
    div[data-testid="stTextInput"] input {
        border-radius: 12px;
        border: none;
        background: rgba(255, 255, 255, 0.2);
        color: #fff;
        font-size: 1.2rem;
        padding: 10px;
        text-align: center;
    }
    div[data-testid="stTextInput"] input:focus {
        background: rgba(255, 255, 255, 0.3);
        outline: none;
    }
    /* Botão */
    div.stButton > button {
        background: linear-gradient(90deg, #ff512f, #dd2476);
        color: white;
        font-weight: bold;
        padding: 0.7rem 1.2rem;
        border: none;
        border-radius: 10px;
        width: 100%;
        transition: transform 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
    }
    /* Histórico em cards */
    .historico-card {
        display: inline-block;
        background: rgba(255,255,255,0.1);
        margin: 5px;
        padding: 10px 15px;
        border-radius: 10px;
        font-weight: bold;
        color: #fff;
        font-size: 1.2rem;
        box-shadow: 0 0 8px rgba(0,0,0,0.4);
    }
    .previsao {
        text-align: center;
        font-size: 1.8rem;
        margin-top: 2rem;
        font-weight: bold;
        color: #ffdf5d;
        text-shadow: 1px 1px 6px #000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================================
# INTERFACE
# ================================
st.markdown('<div class="titulo">🎯 Roleta Preditiva</div>', unsafe_allow_html=True)

with st.form(key="input_form", clear_on_submit=True):
    numero_input = st.text_input(
        "Digite o número sorteado (0-36):",
        key="input_numero",
        max_chars=2,
        placeholder="Ex: 17",
    )
    submit_btn = st.form_submit_button("Registrar", on_click=adicionar_numero)

# Histórico
st.subheader("📜 Histórico de números")
if len(st.session_state.historico) == 0:
    st.info("Nenhum número inserido ainda.")
else:
    st.markdown("".join([f"<span class='historico-card'>{n}</span>" for n in st.session_state.historico]), unsafe_allow_html=True)

# Previsão
proximo = prever_proximo()
if proximo is not None:
    st.markdown(f"<div class='previsao'>🔮 Próximo número previsto: <strong>{proximo}</strong></div>", unsafe_allow_html=True)
else:
    st.info("Insira pelo menos 10 números para ativar a previsão.")
