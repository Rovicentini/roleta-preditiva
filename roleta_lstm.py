import streamlit as st
import numpy as np
import time
from collections import Counter, defaultdict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

# ==============================
# 🌌 Estilo Lovable Futurista (Mantido)
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
    @keyframes gradientMove { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
    .glass-card {
        background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(12px);
        border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px; margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 198, 0.2);
    }
    .stTextInput input, .stTextArea textarea {
        background: rgba(255,255,255,0.07); color: #fff; border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px; font-size: 18px; text-align: center;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ffc6, #00b8ff); color: #000;
        font-weight: bold; border-radius: 12px; padding: 10px 22px;
        font-size: 16px; transition: 0.2s;
        box-shadow: 0 0 12px rgba(0,255,198,0.5);
    }
    .stButton>button:hover { transform: scale(1.05); }
    .historico-bolhas { display: flex; flex-wrap: nowrap; gap: 8px; overflow-x: auto; padding-bottom: 8px;}
    .bolha { min-width: 50px; height: 50px; border-radius: 50%; display: flex;
             align-items: center; justify-content: center; font-weight: bold; font-size: 18px; color: #fff;
             text-shadow: 0 0 6px rgba(0,0,0,0.8); flex-shrink: 0;}
    .vermelho { background: #d72638; box-shadow: 0 0 12px rgba(255,0,0,0.6);}
    .preto { background: #1e1e1e; box-shadow: 0 0 12px rgba(255,255,255,0.3);}
    .verde { background: #21c55d; box-shadow: 0 0 12px rgba(0,255,0,0.6);}
    .prediction-box { font-size: 32px; font-weight: bold; text-align: center; color: #00ffc6;
                      padding: 20px; background: rgba(0,255,198,0.05);
                      border: 2px solid rgba(0,255,198,0.3); border-radius: 16px;
                      animation: pulse 2s infinite; box-shadow: 0 0 25px rgba(0,255,198,0.4);}
    @keyframes pulse {0%{box-shadow:0 0 10px rgba(0,255,198,0.2);}50%{box-shadow:0 0 25px rgba(0,255,198,0.7);}100%{box-shadow:0 0 10px rgba(0,255,198,0.2);}}
    .roulette-wheel { width: 180px; height: 180px; margin: auto; border: 10px solid rgba(255,255,255,0.2);
                      border-radius: 50%; animation: spin 1s linear infinite;
                      background: conic-gradient(#d72638 0deg 9.7deg, #1e1e1e 9.7deg 19.4deg, 
                      #d72638 19.4deg 29.1deg, #1e1e1e 29.1deg 38.8deg,#d72638 38.8deg 48.5deg,
                      #1e1e1e 48.5deg 58.2deg,#21c55d 58.2deg 68deg,#d72638 68deg 77.7deg,
                      #1e1e1e 77.7deg 87.4deg,#d72638 87.4deg 97.1deg,#1e1e1e 97.1deg 106.8deg,
                      #d72638 106.8deg 116.5deg,#1e1e1e 116.5deg 126.2deg,#d72638 126.2deg 135.9deg,
                      #1e1e1e 135.9deg 145.6deg,#d72638 145.6deg 155.3deg,#1e1e1e 155.3deg 165deg,
                      #d72638 165deg 174.7deg,#1e1e1e 174.7deg 184.4deg,#d72638 184.4deg 194.1deg,
                      #1e1e1e 194.1deg 203.8deg,#d72638 203.8deg 213.5deg,#1e1e1e 213.5deg 223.2deg,
                      #d72638 223.2deg 232.9deg,#1e1e1e 232.9deg 242.6deg,#d72638 242.6deg 252.3deg,
                      #1e1e1e 252.3deg 262deg,#d72638 262deg 271.7deg,#1e1e1e 271.7deg 281.4deg,
                      #d72638 281.4deg 291.1deg,#1e1e1e 291.1deg 300.8deg,#d72638 300.8deg 310.5deg,
                      #1e1e1e 310.5deg 320.2deg,#d72638 320.2deg 329.9deg,#1e1e1e 329.9deg 339.6deg,
                      #d72638 339.6deg 349.3deg,#1e1e1e 349.3deg 360deg);}
    @keyframes spin {100% {transform: rotate(360deg);}}
    </style>
""", unsafe_allow_html=True)

# ==============================
# Estado e Funções
# ==============================
if "historico" not in st.session_state: st.session_state.historico = []
if "input" not in st.session_state: st.session_state.input = ""

# 🔴 Cores da Roleta
def cor_roleta(num):
    vermelhos = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}
    pretos = {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}
    return "verde" if num == 0 else "vermelho" if num in vermelhos else "preto"

# 🧠 Função Inteligente de Previsão Multi-Fator
def prever_multifator():
    hist = st.session_state.historico
    if len(hist) < 15:
        return []

    pontos = defaultdict(float)
    total = len(hist)

    # Frequência global e recente
    freq_global = Counter(hist)
    freq_recente = Counter(hist[-20:])

    # Cadeia de Markov (pares e trios)
    pares, trios = defaultdict(Counter), defaultdict(Counter)
    for i in range(len(hist)-1):
        pares[hist[i]][hist[i+1]] += 1
    for i in range(len(hist)-2):
        trios[(hist[i], hist[i+1])][hist[i+2]] += 1

    ult1, ult2 = hist[-1], hist[-2] if len(hist) > 1 else None

    # Vizinhos físicos da roleta
    vizinhos = {
        0:[26,32,15,3,35], 32:[0,26,35,12,3], 15:[0,32,19,4,21], 19:[15,4,21,36,2], 
        4:[15,19,21,2,25], 21:[4,2,25,17,34], 2:[19,21,25,17,28], 25:[4,21,17,34,6],
        17:[21,25,34,6,27], 34:[21,25,17,6,13], 6:[25,17,34,13,27], 27:[17,6,13,36,33],
        13:[6,34,27,33,36], 36:[19,27,13,33,11], 11:[36,33,30,8,23], 30:[33,11,8,23,5],
        8:[11,30,23,10,24], 23:[30,8,10,24,5], 10:[8,23,24,16,33], 5:[23,30,10,24,16],
        24:[8,10,16,33,20], 16:[10,24,33,20,14], 33:[24,16,14,31,9], 20:[16,33,14,31,1],
        14:[16,33,20,31,9], 31:[33,14,9,22,18], 9:[14,31,22,18,29], 22:[31,9,18,29,7],
        18:[31,22,29,7,28], 29:[22,18,7,28,12], 7:[18,29,28,12,35], 28:[18,7,29,12,2],
        12:[29,7,35,3,26], 35:[7,12,3,26,0], 3:[12,35,26,0,32], 26:[35,3,0,32,15], 1:[20,33,14,31,9]
    }

    # Pontuação
    for n in range(37):
        pontos[n] += freq_global[n] * 0.2
        pontos[n] += freq_recente[n] * 0.4
        pontos[n] += (total - hist[::-1].index(n)) * 0.1 if n in hist else 0  # atraso longo

        if ult1 in pares and n in pares[ult1]:
            pontos[n] += pares[ult1][n] * 0.5
        if ult2 and (ult2, ult1) in trios and n in trios[(ult2, ult1)]:
            pontos[n] += trios[(ult2, ult1)][n] * 0.7

        if n in vizinhos.get(ult1, []): pontos[n] += 0.3
        if ult2 and n in vizinhos.get(ult2, []): pontos[n] += 0.2

    return sorted(pontos.items(), key=lambda x: x[1], reverse=True)[:5]

# Inserir número
def inserir_numero():
    if st.session_state.input.strip().isdigit():
        n = int(st.session_state.input.strip())
        if 0 <= n <= 36: st.session_state.historico.append(n)
    st.session_state.input = ""

# Inserir em massa
def inserir_em_massa():
    texto = st.session_state.massa.strip().replace(",", " ").replace(";", " ")
    nums = [int(n) for n in texto.split() if n.isdigit() and 0 <= int(n) <= 36]
    st.session_state.historico.extend(nums)
    st.session_state.massa = ""

# ==============================
# UI Principal
# ==============================
st.markdown("<h1 style='text-align:center;'>🎰 Roleta Preditiva Inteligente</h1>", unsafe_allow_html=True)
col_input, col_hist, col_prev = st.columns([1.5, 2, 1])

# Entrada de dados
with col_input:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("➕ Inserir Número Único")
    st.text_input("Digite um número (0 a 36):", key="input", on_change=inserir_numero)
    st.subheader("📥 Inserir em Massa")
    st.text_area("Cole vários números:", key="massa")
    st.button("Adicionar Números em Massa", on_click=inserir_em_massa)
    st.markdown("</div>", unsafe_allow_html=True)

# Histórico
with col_hist:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📜 Histórico")
    if st.session_state.historico:
        bolhas = "<div class='historico-bolhas'>"
        for num in reversed(st.session_state.historico[-50:]):
            bolhas += f"<div class='bolha {cor_roleta(num)}'>{num}</div>"
        bolhas += "</div>"
        st.markdown(bolhas, unsafe_allow_html=True)
    else:
        st.info("Nenhum número inserido ainda.")
    st.markdown("</div>", unsafe_allow_html=True)

# Previsão
with col_prev:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔮 Previsão Inteligente")
    if len(st.session_state.historico) >= 15:
        placeholder = st.empty()
        with placeholder:
            st.markdown("<div class='roulette-wheel'></div>", unsafe_allow_html=True)
        time.sleep(2)
        preds = prever_multifator()
        html_preds = "<div class='prediction-box'>🎯 Números Prováveis:<br>" + " | ".join([f"<b>{p[0]}</b>" for p in preds]) + "</div>"
        placeholder.markdown(html_preds, unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 15 números para prever.")
    st.markdown("</div>", unsafe_allow_html=True)
