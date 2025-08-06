import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

# ==============================
# 🌌 Estilo Lovable Futurista
# ==============================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }

    /* Fundo animado */
    body {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientMove 12s ease infinite;
        color: #fff;
    }
    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Cartões estilo vidro */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 198, 0.2);
    }

    /* Inputs e botões futuristas */
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.07);
        color: #fff;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
        font-size: 18px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ffc6, #00b8ff);
        color: #000;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 22px;
        font-size: 16px;
        transition: 0.2s;
        box-shadow: 0 0 12px rgba(0,255,198,0.5);
    }
    .stButton>button:hover { transform: scale(1.05); }

    /* Histórico em bolhas neon */
    .historico-bolhas { display: flex; flex-wrap: wrap; gap: 8px; }
    .bolha {
        width: 50px; height: 50px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 18px; color: #fff;
        text-shadow: 0 0 6px rgba(0,0,0,0.8);
    }
    .vermelho { background: #d72638; box-shadow: 0 0 12px rgba(255,0,0,0.6); }
    .preto { background: #1e1e1e; box-shadow: 0 0 12px rgba(255,255,255,0.3); }
    .verde { background: #21c55d; box-shadow: 0 0 12px rgba(0,255,0,0.6); }

    /* Caixa de previsão com glow pulsante */
    .prediction-box {
        font-size: 48px; font-weight: bold; text-align: center;
        color: #00ffc6; padding: 20px;
        background: rgba(0,255,198,0.05);
        border: 2px solid rgba(0,255,198,0.3);
        border-radius: 16px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 25px rgba(0,255,198,0.4);
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(0,255,198,0.2); }
        50% { box-shadow: 0 0 25px rgba(0,255,198,0.7); }
        100% { box-shadow: 0 0 10px rgba(0,255,198,0.2); }
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Estado e Funções
# ==============================
if "historico" not in st.session_state: st.session_state.historico = []
if "modelo" not in st.session_state: st.session_state.modelo = None
if "contador_treinamento" not in st.session_state: st.session_state.contador_treinamento = 0

def cor_roleta(num):
    vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    pretos = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
    return "verde" if num==0 else "vermelho" if num in vermelhos else "preto"

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

def prever_proximo():
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None
    janela = min(19, len(st.session_state.historico))
    entrada = np.array(st.session_state.historico[-janela:]).reshape((1, janela, 1))
    if janela < 19:
        entrada = np.concatenate([np.zeros((1, 19-janela, 1)), entrada], axis=1)
    pred = st.session_state.modelo.predict(entrada, verbose=0)
    return int(np.clip(np.round(pred[0, 0]), 0, 36))

def inserir_numero():
    if st.session_state.input.strip().isdigit():
        n = int(st.session_state.input.strip())
        if 0 <= n <= 36:
            st.session_state.historico.append(n)
            st.session_state.contador_treinamento += 1
            if len(st.session_state.historico) >= 20 and st.session_state.contador_treinamento >= 5:
                st.session_state.modelo = treinar_modelo()
                st.session_state.contador_treinamento = 0
    st.session_state.input = ""

def inserir_em_massa():
    texto = st.session_state.massa.strip().replace(",", " ").replace(";", " ")
    numeros = [n for n in texto.split() if n.isdigit()]
    for n in numeros:
        n = int(n)
        if 0 <= n <= 36:
            st.session_state.historico.append(n)
    st.session_state.massa = ""
    if len(st.session_state.historico) >= 20:
        st.session_state.modelo = treinar_modelo()

# ==============================
# UI Principal
# ==============================
st.markdown("<h1 style='text-align:center;'>🎰 Roleta Preditiva Inteligente</h1>", unsafe_allow_html=True)

col_input, col_hist, col_prev = st.columns([1.5, 2, 1])

# 🔢 BLOCO DE INSERÇÃO
with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("➕ Inserir Número Único")
    st.text_input("Digite um número (0 a 36):", key="input", on_change=inserir_numero)

    st.subheader("📥 Inserir em Massa")
    st.text_area("Cole vários números separados por espaço, vírgula ou ponto e vírgula:", key="massa")
    st.button("Adicionar Números em Massa", on_click=inserir_em_massa)
    st.markdown("</div>", unsafe_allow_html=True)

# 📜 BLOCO HISTÓRICO
with col_hist:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📜 Histórico (Últimos 30)")
    if st.session_state.historico:
        bolhas = "<div class='historico-bolhas'>"
        for num in reversed(st.session_state.historico[-30:]):
            bolhas += f"<div class='bolha {cor_roleta(num)}'>{num}</div>"
        bolhas += "</div>"
        st.markdown(bolhas, unsafe_allow_html=True)
    else:
        st.info("Nenhum número inserido ainda.")
    st.markdown("</div>", unsafe_allow_html=True)

# 🔮 BLOCO PREVISÃO
with col_prev:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔮 Próxima Previsão")
    prox = prever_proximo()
    if prox is not None:
        st.markdown(f"<div class='prediction-box'>🎯 {prox}</div>", unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 10 números para prever.")
    st.markdown("</div>", unsafe_allow_html=True)
