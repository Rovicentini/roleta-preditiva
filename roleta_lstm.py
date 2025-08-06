import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px

# ==============================
# 🎨 Configuração do Layout
# ==============================
st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

# Estilo CSS personalizado
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput input {
        background: #222;
        color: #fff;
        border: 2px solid #4CAF50;
        border-radius: 8px;
        font-size: 20px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50, #2ecc71);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #43a047, #27ae60);
    }
    .prediction-box {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #00ffcc;
        background: rgba(0, 255, 204, 0.1);
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 🔧 Inicialização do Estado
# ==============================
if "historico" not in st.session_state:
    st.session_state.historico = []
if "modelo" not in st.session_state:
    st.session_state.modelo = None

# ==============================
# 🧠 Função para treinar o modelo
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
    modelo.fit(X, y, epochs=50, verbose=0)
    return modelo

# ==============================
# 🔮 Função de previsão corrigida
# ==============================
def prever_proximo():
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None

    janela = min(19, len(st.session_state.historico))
    entrada = np.array(st.session_state.historico[-janela:]).reshape((1, janela, 1))

    # Padding se tiver menos de 19 números
    if janela < 19:
        zeros_pad = np.zeros((1, 19 - janela, 1))
        entrada = np.concatenate([zeros_pad, entrada], axis=1)

    pred = st.session_state.modelo.predict(entrada, verbose=0)
    return int(np.clip(np.round(pred[0, 0]), 0, 36))

# ==============================
# ➕ Função para adicionar número
# ==============================
def adicionar_numero():
    if st.session_state.input_numero.strip().isdigit():
        numero = int(st.session_state.input_numero)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
            if len(st.session_state.historico) >= 20:
                st.session_state.modelo = treinar_modelo()
        st.session_state.input_numero = ""  # Limpa campo

# ==============================
# 🖥️ Interface
# ==============================
st.title("🎰 Roleta Preditiva Inteligente")
st.markdown("Insira os números que saíram na roleta e veja a previsão do próximo!")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("➕ Inserir Número")
    st.text_input("Digite o número (0 a 36):", key="input_numero", on_change=adicionar_numero)

    if st.session_state.historico:
        st.subheader("📜 Histórico de Números")
        st.write(", ".join(map(str, st.session_state.historico[-20:])))

with col2:
    st.subheader("🔮 Próxima Previsão")
    proximo = prever_proximo()
    if proximo is not None:
        st.markdown(f"<div class='prediction-box'>🎯 {proximo}</div>", unsafe_allow_html=True)
    else:
        st.info("Insira ao menos 10 números para gerar previsões.")

# ==============================
# 📊 Gráfico do Histórico
# ==============================
if st.session_state.historico:
    fig = px.line(y=st.session_state.historico, markers=True, title="Histórico dos Últimos Números")
    fig.update_layout(
        plot_bgcolor="#1e1e2f",
        paper_bgcolor="#1e1e2f",
        font=dict(color="#ffffff"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    st.plotly_chart(fig, use_container_width=True)
