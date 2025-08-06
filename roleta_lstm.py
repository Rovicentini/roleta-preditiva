import streamlit as st
import numpy as np
import plotly.express as px

# Inicializa histórico se não existir
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Função para limpar o input e adicionar o número ao histórico
def inserir_numero():
    if st.session_state.input_numero not in st.session_state.historico:
        st.session_state.historico.insert(0, st.session_state.input_numero)
    st.session_state.input_numero = 0  # limpa campo de input

# Função para prever próximo número (exemplo simples, ajuste conforme seu modelo)
def prever_proximo():
    if len(st.session_state.historico) < 19:
        return None
    entrada = np.array(st.session_state.historico[:19]).reshape((1, 19, 1))
    # Aqui sua lógica de predição, placeholder:
    proximo_numero = (st.session_state.historico[0] + 1) % 37
    return proximo_numero

# Layout moderno e limpo
st.markdown("""
<style>
    .main-container {
        background: linear-gradient(135deg, #1f1c2c, #928dab);
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
        max-width: 700px;
        margin: 30px auto;
    }
    .header {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-align: center;
        letter-spacing: 2px;
    }
    .input-section {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }
    .historic-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: center;
        margin-bottom: 20px;
    }
    .number-box {
        width: 45px;
        height: 45px;
        border-radius: 8px;
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: 700;
        font-size: 1.1rem;
        color: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        user-select: none;
    }
    .number-red { background-color: #d32f2f; }
    .number-black { background-color: #212121; }
    .number-green { background-color: #388e3c; }
    .prediction {
        text-align: center;
        font-size: 1.3rem;
        margin-top: 10px;
        font-weight: 600;
        color: #ffe066;
    }
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<div class="header">Roleta Preditiva - Histórico e Previsão</div>', unsafe_allow_html=True)

    # Input número com on_change para limpar e adicionar ao histórico
    st.number_input(
        "Informe o número sorteado (0 a 36) e pressione ENTER:",
        min_value=0, max_value=36, step=1,
        key="input_numero",
        on_change=inserir_numero
    )

    # Histórico - último número à esquerda
    st.markdown('<div class="historic-container">', unsafe_allow_html=True)
    for num in st.session_state.historico:
        # Definir cores conforme roleta europeia
        if num == 0:
            color_class = "number-green"
        elif num in [1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36]:
            color_class = "number-red"
        else:
            color_class = "number-black"
        st.markdown(f'<div class="number-box {color_class}">{num}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Previsão
    proximo = prever_proximo()
    if proximo is not None:
        st.markdown(f'<div class="prediction">Próximo número previsto: <strong>{proximo}</strong></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction">Insira pelo menos 19 números para iniciar a previsão.</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
