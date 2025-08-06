import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==============================
# 🎨 Layout Moderno
# ==============================
st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0d0f1a;
        color: #fafafa;
    }
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput input {
        background: #1c1c1e;
        color: #00ffc6;
        border: 2px solid #00ffc6;
        border-radius: 10px;
        font-size: 22px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ffc6, #00b894);
        color: #000;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 18px;
    }
    .prediction-box {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #00ffc6;
        background: rgba(0, 255, 198, 0.08);
        padding: 25px;
        border-radius: 12px;
        border: 2px solid #00ffc6;
        margin-top: 20px;
    }
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
    modelo.fit(X, y, epochs=20, verbose=0)  # 🔥 Menos épocas = mais rápido
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
# ➕ Adicionar Número (Rápido)
# ==============================
def adicionar_numero():
    numero_str = st.session_state.input_numero.strip()
    if numero_str.isdigit():
        numero = int(numero_str)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
            st.session_state.contador_treinamento += 1

            # Só treina a cada 5 inserções (reduz lag)
            if len(st.session_state.historico) >= 20 and st.session_state.contador_treinamento >= 5:
                st.session_state.modelo = treinar_modelo()
                st.session_state.contador_treinamento = 0

    st.session_state.input_numero = ""  # Limpa instantaneamente

# ==============================
# 🖥️ Interface
# ==============================
st.title("🎰 Roleta Preditiva Inteligente")
st.markdown("Insira os números e veja as previsões!")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("➕ Inserir Número")
    st.text_input("Digite o número (0 a 36):", key="input_numero", on_change=adicionar_numero)

    if st.session_state.historico:
        st.subheader("📜 Histórico (Últimos 20)")
        st.write(", ".join(map(str, st.session_state.historico[-20:])))

with col2:
    st.subheader("🔮 Previsão")
    proximo = prever_proximo()
    if proximo is not None:
        st.markdown(f"<div class='prediction-box'>🎯 {proximo}</div>", unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 10 números para iniciar as previsões.")
