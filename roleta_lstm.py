import streamlit as st
import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Roleta Preditiva", page_icon="🎰", layout="wide")

# ==============================
# 🌌 Estilo Lovable Futurista (sem alteração)
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
    .historico-bolhas {
        display: flex; 
        flex-wrap: nowrap;  /* sem quebra de linha, rolar horizontal */
        gap: 8px; 
        overflow-x: auto; /* permite scroll horizontal */
        padding-bottom: 8px;
    }
    .bolha {
        min-width: 50px; height: 50px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: bold; font-size: 18px; color: #fff;
        text-shadow: 0 0 6px rgba(0,0,0,0.8);
        flex-shrink: 0; /* evita encolher */
    }
    .vermelho { background: #d72638; box-shadow: 0 0 12px rgba(255,0,0,0.6); }
    .preto { background: #1e1e1e; box-shadow: 0 0 12px rgba(255,255,255,0.3); }
    .verde { background: #21c55d; box-shadow: 0 0 12px rgba(0,255,0,0.6); }

    /* Caixa de previsão */
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

    /* Animação da roleta */
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
if "modelo" not in st.session_state: st.session_state.modelo = None
if "contador_treinamento" not in st.session_state: st.session_state.contador_treinamento = 0
if "input" not in st.session_state: st.session_state.input = ""

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
    modelo.add(Dense(37, activation='softmax'))  # Saída para 37 classes (0 a 36)
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    modelo.fit(X, y, epochs=40, verbose=0)
    return modelo

def prever_proximo(top_n=3):
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None
    janela = min(19, len(st.session_state.historico))
    entrada = np.array(st.session_state.historico[-janela:]).reshape((1, janela, 1))
    if janela < 19:
        entrada = np.concatenate([np.zeros((1, 19-janela, 1)), entrada], axis=1)
    
    prob = st.session_state.modelo.predict(entrada, verbose=0)[0]  # vetor de probabilidade (37,)
    
    top_indices = prob.argsort()[-top_n:][::-1]  # índices dos top n números com maior probabilidade
    top_probs = prob[top_indices]
    
    return list(zip(top_indices, top_probs))  # lista de (numero, probabilidade)

def inserir_numero():
    if st.session_state.input.strip().isdigit():
        n = int(st.session_state.input.strip())
        if 0 <= n <= 36:
            st.session_state.historico.append(n)
            st.session_state.contador_treinamento += 1
            if len(st.session_state.historico) >= 20 and st.session_state.contador_treinamento >= 5:
                st.session_state.modelo = treinar_modelo()
                st.session_state.contador_treinamento = 0
    st.session_state.input = ""  # limpa o input após ENTER

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

# 🔮 BLOCO PREVISÃO COM ROLETA
with col_prev:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔮 Próxima Previsão")

    previsoes = prever_proximo()
    if previsoes is not None:
        placeholder = st.empty()
        with placeholder:
            st.markdown("<div class='roulette-wheel'></div>", unsafe_allow_html=True)
        time.sleep(2.5)  # Tempo de rotação da roleta

        # Monta texto com os top números + probabilidades
        previsoes_texto = "<div style='text-align:center;'>"
        for num, prob in previsoes:
            cor = cor_roleta(num)
            color_bg = "#21c55d" if cor == "verde" else "#d72638" if cor == "vermelho" else "#1e1e1e"
            previsoes_texto += (
                f"<span style='font-size:28px; font-weight:bold; color:#00ffc6; margin:8px;'>"
                f"<span style='color:#fff; text-shadow:0 0 6px rgba(0,0,0,0.8); "
                f"background-color:{color_bg}; border-radius:50%; padding:10px 16px; display:inline-block;'>"
                f"{num}</span>"
                f" <small style='font-weight:normal; color:#0ff;'>{prob*100:.1f}%</small></span>&nbsp;&nbsp;"
            )
        previsoes_texto += "</div>"

        placeholder.markdown(previsoes_texto, unsafe_allow_html=True)
    else:
        st.info("Insira pelo menos 10 números para prever.")

    st.markdown("</div>", unsafe_allow_html=True)
