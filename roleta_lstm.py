import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==============================
# 🎨 Layout Ultra Moderno
# ==============================
st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0b0f19;
        color: #fafafa;
    }
    .stApp {
        background: linear-gradient(135deg, #050d1a, #101d34, #1b2b4a);
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #00ffc6;
    }
    .stTextInput input, .stTextArea textarea {
        background: #121212;
        color: #00ffc6;
        border: 2px solid #00ffc6;
        border-radius: 12px;
        font-size: 22px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ffc6, #00b894);
        color: #000;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.05);
    }
    .prediction-box {
        text-align: center;
        font-size: 46px;
        font-weight: bold;
        color: #00ffc6;
        background: rgba(0, 255, 198, 0.08);
        padding: 25px;
        border-radius: 16px;
        border: 2px solid #00ffc6;
        margin-top: 20px;
        box-shadow: 0px 0px 20px rgba(0, 255, 198, 0.4);
    }
    .historico-bolhas {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .bolha {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        font-weight: bold;
        border-radius: 50%;
        font-size: 18px;
        color: #fff;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.4);
    }
    .vermelho { background: #d72638; }
    .preto { background: #1e1e1e; }
    .verde { background: #21c55d; color: #fff; }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🔧 Estados
# ==============================
if "historico" not in st.session_state:
    st.session_state.historico = []
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "contador_treinamento" not in st.session_state:
    st.session_state.contador_treinamento = 0

# ==============================
# 🎨 Função Cores da Roleta
# ==============================
def cor_roleta(numero):
    vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    pretos = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
    if numero == 0:
        return "verde"
    return "vermelho" if numero in vermelhos else "preto"

# ==============================
# 🧠 Função de Treinamento
# ==============================
def treinar_modelo():
    dados = np.array(st.session_state.historico)
    X, y = [], []
    for i in range(len(dados)-19):
        X.append(dados[i:i+19])
        y.append(dados[i+19])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    modelo = Sequential()
    modelo.add(LSTM(50, activation='relu', input_shape=(19, 1)))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mse')
    modelo.fit(X, y, epochs=20, verbose=0)
    return modelo

# ==============================
# 🔮 Função de Previsão
# ==============================
def prever_proximo():
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None
    janela = min(19, len(st.session_state.historico))
    entrada = np.array(st.session_state.historico[-janela:]).reshape((1, janela, 1))
    if janela < 19:
        zeros_pad = np.zeros((1, 19 - janela, 1))
        entrada = np.concatenate([zeros_pad, entrada], axis=1)
    pred = st.session_state.modelo.predict(entrada, verbose=0)
    return int(np.clip(np.round(pred[0, 0]), 0, 36))

# ==============================
# ➕ Inserir Números
# ==============================
def inserir_numero_unico():
    numero_str = st.session_state.input_numero.strip()
    if numero_str.isdigit():
        numero = int(numero_str)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
            st.session_state.contador_treinamento += 1
            if len(st.session_state.historico) >= 20 and st.session_state.contador_treinamento >= 5:
                st.session_state.modelo = treinar_modelo()
                st.session_state.contador_treinamento = 0
    st.session_state.input_numero = ""  # limpa rápido

def inserir_varios_numeros():
    numeros_str = st.session_state.input_varios.strip()
    numeros = [n.strip() for n in numeros_str.replace(";", ",").split(",") if n.strip().isdigit()]
    for n in numeros:
        numero = int(n)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
    if len(st.session_state.historico) >= 20:
        st.session_state.modelo = treinar_modelo()
    st.session_state.input_varios = ""  # limpa rápido

# ==============================
# 🖥️ Interface
# ==============================
st.title("🎰 Roleta Preditiva Inteligente")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("➕ Inserir Número")
    st.text_input("Digite um número (0 a 36):", key="input_numero", on_change=inserir_numero_unico)
    st.subheader("📥 Inserir Vários")
    st.text_area("Cole aqui (ex: 12, 5, 8, 19, 0, 32):", key="input_varios")
    st.button("Adicionar em Lote", on_click=inserir_varios_numeros)

    if st.session_state.historico:
        st.subheader("📜 Histórico (Últimos 30)")
        bolhas_html = "<div class='historico-bolhas'>"
        for num in reversed(st.session_state.historico[-30:]):  # Último à esquerda
            cor = cor_roleta(num)
            bolhas_html += f"<div class='bolha {cor}'>{num}</div>"
        bolhas_html += "</div>"
        st.markdown(bolhas_html, unsafe_allow_html=True)

with col2:
    st.subheader("🔮 Previsão")
    proximo = prever_proximo()
    if proximo is not None:
        st.markdown(f"<div class='prediction-box'>🎯 {proximo}</div>", unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 10 números para iniciar as previsões.")
