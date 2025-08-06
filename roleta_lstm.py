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
        font-size: 20px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sessão de histórico ---
if "historico" not in st.session_state:
    st.session_state.historico = []

# --- Título ---
st.title("🎰 Roleta Preditiva")
st.write("Insira os números que saíram na roleta para identificar padrões e gerar previsões.")

# --- Campo para inserir número ---
numero = st.text_input("Digite o número (0 a 36) e pressione Enter:", key="num_input")

# --- Ao pressionar Enter ---
if numero:
    try:
        num = int(numero)
        if 0 <= num <= 36:
            st.session_state.historico.append(num)
            st.session_state.num_input = ""  # limpa campo automaticamente
        else:
            st.warning("Digite apenas números entre 0 e 36.")
    except:
        st.warning("Digite um número válido.")

# --- Mostrar histórico ---
if st.session_state.historico:
    st.subheader("📜 Histórico de números inseridos")
    hist_str = " → ".join(map(str, st.session_state.historico[::-1]))
    st.markdown(f"**Últimos números (mais recente à esquerda):**  \n{hist_str}")

    # Botão para limpar histórico
    if st.button("🗑️ Limpar histórico"):
        st.session_state.historico.clear()

# --- Placeholder de previsão ---
st.subheader("🔮 Próxima previsão")
if len(st.session_state.historico) >= 6:
    st.info("Modelo em execução... (Previsões reais serão implementadas na próxima etapa)")
else:
    st.warning("Insira pelo menos 6 números para gerar previsões.")
