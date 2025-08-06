import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

# --- Configurações da página ---
st.set_page_config(
    page_title="Roleta Preditiva",
    page_icon="🎰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilos customizados (CSS) ---
st.markdown(
    """
    <style>
    /* Fundo claro e fonte agradável */
    .main {
        background-color: #f9fafb;
        color: #1a202c;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTextInput>div>div>input {
        height: 45px;
        font-size: 18px;
        border-radius: 8px;
        border: 1.5px solid #ddd;
        padding-left: 15px;
    }
    .stButton>button {
        background-color: #0072f5;
        color: white;
        font-weight: bold;
        height: 45px;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005bb5;
        cursor: pointer;
    }
    .sidebar .sidebar-content {
        background-color: #e1e7f5;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Funções e variáveis globais ---

# Lista de números inseridos pelo usuário (inicial vazia)
if 'historico' not in st.session_state:
    st.session_state.historico = []

# Modelo LSTM simples (exemplo)
def criar_modelo():
    model = Sequential()
    model.add(LSTM(32, input_shape=(5, 1), return_sequences=False))
    model.add(Dense(37, activation='softmax'))  # Roleta tem 0-36 = 37 números
    model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy')
    return model

if 'modelo' not in st.session_state:
    st.session_state.modelo = criar_modelo()

# Função para preparar dados (sequência para input LSTM)
def preparar_dados(historico):
    if len(historico) < 6:
        return None, None
    X, y = [], []
    for i in range(len(historico) - 5):
        seq_in = historico[i:i+5]
        seq_out = historico[i+5]
        X.append(seq_in)
        y.append(seq_out)
    X = np.array(X).reshape(-1, 5, 1) / 36  # Normaliza
    y = np.array(y)
    return X, y

# Treina o modelo com histórico atual
def treinar_modelo():
    X, y = preparar_dados(st.session_state.historico)
    if X is not None:
        st.session_state.modelo.fit(X, y, epochs=15, verbose=0)

# Faz a previsão dos próximos 5 números mais prováveis
def prever_proximos():
    if len(st.session_state.historico) < 5:
        return [], []
    seq_input = np.array(st.session_state.historico[-5:]).reshape(1, 5, 1) / 36
    preds = st.session_state.modelo.predict(seq_input, verbose=0)[0]
    top5_idx = preds.argsort()[-5:][::-1]
    top5_probs = preds[top5_idx]
    return top5_idx, top5_probs

# --- Layout da aplicação ---

st.title("🎰 Roleta Preditiva Inteligente")

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("## Insira o número sorteado na roleta (0 a 36)")
    numero_input = st.text_input(
        "Número sorteado:",
        key="input_numero",
        max_chars=2,
        placeholder="Exemplo: 17",
        label_visibility="collapsed",
    )

    # Botão para adicionar o número
    if st.button("Adicionar número") or (st.session_state.get("input_numero") and st.session_state.get("input_numero") != ''):
        try:
            numero = int(numero_input)
            if 0 <= numero <= 36:
                st.session_state.historico.append(numero)
                treinar_modelo()
                st.session_state.input_numero = ""  # limpa input
                st.experimental_rerun()  # para dar foco e atualizar interface
            else:
                st.error("Número inválido! Insira um número entre 0 e 36.")
        except ValueError:
            st.error("Entrada inválida! Digite apenas números entre 0 e 36.")

    st.markdown("---")
    st.markdown("## Previsão dos próximos números mais prováveis")
    top5, probs = prever_proximos()
    if top5 != []:
        for num, prob in zip(top5, probs):
            st.write(f"**Número {num}** com probabilidade {prob:.2%}")

with col2:
    st.markdown("## Histórico de números sorteados")
    if len(st.session_state.historico) == 0:
        st.info("Nenhum número inserido ainda.")
    else:
        hist_df = pd.DataFrame(st.session_state.historico, columns=["Número"])
        st.dataframe(hist_df, height=300)

    st.markdown("---")
    st.markdown("## Frequência dos números")
    if len(st.session_state.historico) > 0:
        freq = pd.Series(st.session_state.historico).value_counts().sort_index()
        freq_df = pd.DataFrame({'Número': freq.index, 'Frequência': freq.values})
        fig = px.bar(freq_df, x='Número', y='Frequência', title="Números mais frequentes", 
                     labels={"Número": "Número", "Frequência": "Quantidade"})
        st.plotly_chart(fig, use_container_width=True)

# --- Finalização ---
st.markdown("---")
st.markdown("Desenvolvido para uso pessoal e aprendizado, com inteligência LSTM simples.")

