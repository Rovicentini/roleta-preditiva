import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# --- Configuração da página ---
st.set_page_config(page_title="Roleta Preditiva", layout="centered")

# --- Estilos personalizados ---
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 22px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        font-size: 18px;
        margin-top: 10px;
    }
    .chips {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        margin-top: 10px;
    }
    .chip {
        background-color: #4CAF50;
        color: white;
        padding: 8px 14px;
        border-radius: 20px;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sessão de histórico ---
if "historico" not in st.session_state:
    st.session_state.historico = []

# --- Função para resetar o campo após Enter ---
def salvar_numero():
    numero = st.session_state["num_input"]
    if numero.strip() != "":
        try:
            num = int(numero)
            if 0 <= num <= 36:
                st.session_state.historico.append(num)
            else:
                st.warning("Digite apenas números entre 0 e 36.")
        except:
            st.warning("Digite um número válido.")
    st.session_state["num_input"] = ""  # limpa o campo

# --- Título ---
st.title("🎰 Roleta Preditiva")
st.write("Insira os números que saíram na roleta para identificar padrões e gerar previsões.")

# --- Campo para inserir número (Enter já salva e limpa) ---
st.text_input(
    "Digite o número (0 a 36) e pressione Enter:",
    key="num_input",
    on_change=salvar_numero,
    placeholder="Ex: 17"
)

# --- Mostrar histórico ---
if st.session_state.historico:
    st.subheader("📜 Histórico de números inseridos")
    hist_str = " → ".join(map(str, st.session_state.historico[::-1]))
    st.markdown(f"**Últimos números (mais recente à esquerda):**  \n{hist_str}")

    # Botão para limpar histórico
    if st.button("🗑️ Limpar histórico"):
        st.session_state.historico.clear()

# --- Exibir previsões simuladas ---
st.subheader("🔮 Próxima previsão")
if len(st.session_state.historico) >= 6:
    # Aqui futuramente entra a IA LSTM real
    previsoes_falsas = np.random.choice(range(37), 3, replace=False)  # simulação
    st.markdown("**Possíveis próximos números:**")
    chips_html = "<div class='chips'>" + "".join([f"<div class='chip'>{p}</div>" for p in previsoes_falsas]) + "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)
else:
    st.warning("Insira pelo menos 6 números para gerar previsões.")
