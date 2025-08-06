import streamlit as st
import numpy as np
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =============================
# CONFIGURAÇÃO DE ESTILO MODERNO
# =============================
st.set_page_config(page_title="Roleta Preditiva", layout="wide")

st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #fff;
            font-family: 'Segoe UI', sans-serif;
        }
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        }
        .title {
            font-size: 38px;
            font-weight: bold;
            text-align: center;
            color: #ffffff;
            margin-bottom: 10px;
        }
        .prediction-box {
            background: rgba(255,255,255,0.08);
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .prediction-number {
            font-size: 60px;
            font-weight: bold;
            color: #4fff9f;
        }
        .neighbors {
            font-size: 20px;
            color: #ffd166;
        }
        .confidence {
            font-size: 18px;
            color: #06d6a0;
            margin-top: 10px;
        }
        .history {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .ball {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 16px;
            color: white;
        }
        .ball-red { background: #d90429; }
        .ball-black { background: #222; }
        .ball-green { background: #1b9c85; }
        .block {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================
# FUNÇÕES PRINCIPAIS
# =============================

def criar_modelo():
    model = Sequential([
        LSTM(32, input_shape=(19, 1)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def prever_proximo():
    if len(st.session_state.historico) < 19:
        return None, [], "Aguardando mais dados"
    
    # Prepara dados
    entrada = np.array(st.session_state.historico[-19:]).reshape((1, 19, 1))
    pred = st.session_state.modelo.predict(entrada, verbose=0)
    numero_previsto = int(np.round(pred[0][0])) % 37

    # Ajuste com padrões históricos
    frequencia = {n: st.session_state.historico.count(n) for n in set(st.session_state.historico)}
    if numero_previsto in frequencia and frequencia[numero_previsto] > 3:
        numero_previsto = (numero_previsto + 5) % 37

    # Definir vizinhos dinamicamente
    ultimos = st.session_state.historico[-10:]
    dispersao = len(set(ultimos))
    vizinhos = 1 if dispersao > 7 else 2 if dispersao > 4 else 3

    numeros_vizinhos = [(numero_previsto + i) % 37 for i in range(-vizinhos, vizinhos+1)]

    # Definir confiança
    confianca = "ALTA" if vizinhos == 1 else "MÉDIA" if vizinhos == 2 else "BAIXA"

    return numero_previsto, numeros_vizinhos, confianca

def cor_numero(numero):
    if numero == 0:
        return "green"
    vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    return "red" if numero in vermelhos else "black"

# =============================
# ESTADO INICIAL
# =============================
if "historico" not in st.session_state:
    st.session_state.historico = []
if "modelo" not in st.session_state:
    st.session_state.modelo = criar_modelo()

# =============================
# LAYOUT PRINCIPAL
# =============================
st.markdown("<div class='title'>🎰 Roleta Preditiva Inteligente</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    with st.container():
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("🎯 Inserir Números")
        numero = st.number_input("Informe o número (0 a 36):", min_value=0, max_value=36, step=1, key="input_numero")
        if st.button("Adicionar Número"):
            st.session_state.historico.insert(0, int(numero))  # Último número à esquerda
            if len(st.session_state.historico) > 200:
                st.session_state.historico = st.session_state.historico[:200]
            st.session_state.input_numero = 0
        st.text_area("Inserir números em massa (separados por vírgula):", key="massa")
        if st.button("Adicionar em Massa"):
            nums = [int(x.strip()) for x in st.session_state.massa.split(",") if x.strip().isdigit()]
            for n in nums[::-1]:  # Mantém ordem correta
                st.session_state.historico.insert(0, n)
            if len(st.session_state.historico) > 200:
                st.session_state.historico = st.session_state.historico[:200]
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("<div class='block'>", unsafe_allow_html=True)
        st.subheader("📜 Histórico")
        st.markdown("<div class='history'>", unsafe_allow_html=True)
        for n in st.session_state.historico:
            cor = cor_numero(n)
            st.markdown(f"<div class='ball ball-{cor}'>{n}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='block prediction-box'>", unsafe_allow_html=True)
    st.subheader("🔮 Predição")
    numero_previsto, vizinhos, confianca = prever_proximo()
    if numero_previsto is not None:
        st.markdown(f"<div class='prediction-number'>{numero_previsto}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='neighbors'>Aposte também nos vizinhos: {', '.join(map(str, vizinhos))}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='confidence'>Confiança: {confianca}</div>", unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 19 números para iniciar a predição.")
    st.markdown("</div>", unsafe_allow_html=True)
