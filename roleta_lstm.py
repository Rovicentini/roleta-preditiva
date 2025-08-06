import streamlit as st
import numpy as np
import time
from collections import Counter, defaultdict

st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

# ==============================
# 🌌 Estilo Lovable Futurista (INALTERADO)
# ==============================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600&display=swap');
    * { font-family: 'Space Grotesk', sans-serif; }

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

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 198, 0.2);
    }

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

    .historico-bolhas {
        display: flex; 
        flex-wrap: nowrap;
        gap: 8px; 
        overflow-x: auto;
        padding-bottom: 8px;
    }
    .bolha {
        min-width: 50px; height: 50px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 18px; color: #fff;
        text-shadow: 0 0 6px rgba(0,0,0,0.8);
        flex-shrink: 0;
    }
    .vermelho { background: #d72638; box-shadow: 0 0 12px rgba(255,0,0,0.6); }
    .preto { background: #1e1e1e; box-shadow: 0 0 12px rgba(255,255,255,0.3); }
    .verde { background: #21c55d; box-shadow: 0 0 12px rgba(0,255,0,0.6); }

    .prediction-box {
        font-size: 32px; font-weight: bold; text-align: center;
        color: #00ffc6; padding: 15px;
        background: rgba(0,255,198,0.05);
        border: 2px solid rgba(0,255,198,0.3);
        border-radius: 16px;
        animation: pulse 2s infinite;
        box-shadow: 0 0 25px rgba(0,255,198,0.4);
        margin-bottom: 10px;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px rgba(0,255,198,0.2); }
        50% { box-shadow: 0 0 25px rgba(0,255,198,0.7); }
        100% { box-shadow: 0 0 10px rgba(0,255,198,0.2); }
    }

    .roulette-wheel {
        width: 180px;
        height: 180px;
        margin: auto;
        border: 10px solid rgba(255,255,255,0.2);
        border-radius: 50%;
        background: conic-gradient(#d72638 0deg 9.7deg, #1e1e1e 9.7deg 19.4deg, 
                                  #d72638 19.4deg 29.1deg, #1e1e1e 29.1deg 38.8deg,
                                  #d72638 38.8deg 48.5deg, #1e1e1e 48.5deg 58.2deg,
                                  #21c55d 58.2deg 68deg, #d72638 68deg 77.7deg,
                                  #1e1e1e 77.7deg 87.4deg, #d72638 87.4deg 97.1deg,
                                  #1e1e1e 97.1deg 106.8deg, #d72638 106.8deg 116.5deg,
                                  #1e1e1e 116.5deg 126.2deg, #d72638 126.2deg 135.9deg,
                                  #1e1e1e 135.9deg 145.6deg, #d72638 145.6deg 155.3deg,
                                  #1e1e1e 155.3deg 165deg, #d72638 165deg 174.7deg,
                                  #1e1e1e 174.7deg 184.4deg, #d72638 184.4deg 194.1deg,
                                  #1e1e1e 194.1deg 203.8deg, #d72638 203.8deg 213.5deg,
                                  #1e1e1e 213.5deg 223.2deg, #d72638 223.2deg 232.9deg,
                                  #1e1e1e 232.9deg 242.6deg, #d72638 242.6deg 252.3deg,
                                  #1e1e1e 252.3deg 262deg, #d72638 262deg 271.7deg,
                                  #1e1e1e 271.7deg 281.4deg, #d72638 281.4deg 291.1deg,
                                  #1e1e1e 291.1deg 300.8deg, #d72638 300.8deg 310.5deg,
                                  #1e1e1e 310.5deg 320.2deg, #d72638 320.2deg 329.9deg,
                                  #1e1e1e 329.9deg 339.6deg, #d72638 339.6deg 349.3deg,
                                  #1e1e1e 349.3deg 360deg);
        animation: spin 1s linear infinite;
        box-shadow: 0 0 25px rgba(255,255,255,0.3);
    }
    @keyframes spin { 100% { transform: rotate(360deg); } }
    </style>
""", unsafe_allow_html=True)

# ==============================
# Estado e Funções
# ==============================
if "historico" not in st.session_state: st.session_state.historico = []
if "input" not in st.session_state: st.session_state.input = ""

def cor_roleta(num):
    vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    pretos = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
    return "verde" if num==0 else "vermelho" if num in vermelhos else "preto"

# ==============================
# 🔮 Função de Previsão Inteligente
# ==============================
def prever_com_historico():
    hist = st.session_state.historico
    if len(hist) < 15:
        return None

    freq_recente = Counter(hist[-30:])
    freq_global = Counter(hist)
    correlacao = defaultdict(Counter)
    for i in range(len(hist)-1):
        correlacao[hist[i]][hist[i+1]] += 1

    scores = Counter()
    for n in range(37):
        scores[n] = freq_recente[n]*2 + freq_global[n]*1

    ultimos = hist[-3:]
    for u in ultimos:
        for num, count in correlacao[u].items():
            scores[num] += count * 2

    for n in ultimos:
        scores[n] *= 0.6

    ordenados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = [num for num, sc in ordenados if sc > 0]

    melhores = [top[0]]
    for i in range(1, len(top)):
        if len(melhores) >= 5: break
        if scores[top[i]] >= scores[melhores[0]] * 0.65:
            melhores.append(top[i])
        else:
            break
    return melhores

# ==============================
# Inserção
# ==============================
def inserir_numero():
    if st.session_state.input.strip().isdigit():
        n = int(st.session_state.input.strip())
        if 0 <= n <= 36:
            st.session_state.historico.append(n)
    st.session_state.input = ""

def inserir_em_massa():
    texto = st.session_state.massa.strip().replace(",", " ").replace(";", " ")
    numeros = [n for n in texto.split() if n.isdigit()]
    for n in numeros:
        n = int(n)
        if 0 <= n <= 36:
            st.session_state.historico.append(n)
    st.session_state.massa = ""

# ==============================
# UI Principal (Layout Mantido)
# ==============================
st.markdown("<h1 style='text-align:center;'>🎰 Roleta Preditiva Inteligente</h1>", unsafe_allow_html=True)
col_input, col_hist, col_prev = st.columns([1.5, 2, 1])

with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("➕ Inserir Número Único")
    st.text_input("Digite um número (0 a 36):", key="input", on_change=inserir_numero)
    st.subheader("📥 Inserir em Massa")
    st.text_area("Cole vários números separados por espaço, vírgula ou ponto e vírgula:", key="massa")
    st.button("Adicionar Números em Massa", on_click=inserir_em_massa)
    st.markdown("</div>", unsafe_allow_html=True)

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

with col_prev:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔮 Números Mais Prováveis")
    previsoes = prever_com_historico()
    if previsoes:
        placeholder = st.empty()
        with placeholder:
            st.markdown("<div class='roulette-wheel'></div>", unsafe_allow_html=True)
        time.sleep(2)
        for p in previsoes:
            st.markdown(f"<div class='prediction-box'>🎯 {p}</div>", unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 15 números para prever.")
    st.markdown("</div>", unsafe_allow_html=True)
