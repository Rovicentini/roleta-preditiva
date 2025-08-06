import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.express as px

# Inicializar variáveis de sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

if "modelo" not in st.session_state:
    st.session_state.modelo = None

if "limpar_input" not in st.session_state:
    st.session_state.limpar_input = False

if "input_numero" not in st.session_state:
    st.session_state.input_numero = ""

# Função para treinar o modelo
def treinar_modelo():
    if len(st.session_state.historico) < 10:
        return  # Treina só com pelo menos 10 números
    dados = np.array(st.session_state.historico[-20:])  # pegar últimos 20 números
    dados = dados.reshape((1, len(dados), 1))
    labels = dados[:, 1:, :]
    dados = dados[:, :-1, :]

    modelo = Sequential()
    modelo.add(LSTM(50, activation='relu', input_shape=(dados.shape[1], dados.shape[2])))
    modelo.add(Dense(1))
    modelo.compile(optimizer='adam', loss='mse')

    modelo.fit(dados, labels, epochs=100, verbose=0)
    st.session_state.modelo = modelo

# Função para prever próximo número
def prever_proximo():
    if st.session_state.modelo is None or len(st.session_state.historico) < 10:
        return None
    entrada = np.array(st.session_state.historico[-19:]).reshape((1, 19, 1))
    pred = st.session_state.modelo.predict(entrada, verbose=0)
    pred_num = int(np.round(pred[0, 0]))
    # Ajustar para intervalo válido da roleta (0-36)
    if pred_num < 0:
        pred_num = 0
    if pred_num > 36:
        pred_num = 36
    return pred_num

# Layout melhorado com cores
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .titulo {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #ffffff;
        text-shadow: 2px 2px 4px #6a3e1a;
    }
    .input-area {
        max-width: 300px;
        margin: 0 auto 1rem auto;
    }
    .historico {
        max-width: 500px;
        margin: 0 auto;
        background-color: #ffffffaa;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="titulo">Roleta Preditiva</div>', unsafe_allow_html=True)

with st.form(key='input_form', clear_on_submit=False):
    input_valor = "" if st.session_state.limpar_input else st.session_state.input_numero
    numero_input = st.text_input(
        "Digite o número sorteado (0-36):",
        key="input_numero",
        value=input_valor,
        max_chars=2,
        placeholder="Exemplo: 17",
        label_visibility="visible"
    )
    submit_btn = st.form_submit_button("Registrar")

if submit_btn:
    try:
        numero = int(numero_input)
        if 0 <= numero <= 36:
            st.session_state.historico.append(numero)
            treinar_modelo()
            st.session_state.limpar_input = True
            st.experimental_rerun()
        else:
            st.error("Número inválido! Insira um número entre 0 e 36.")
    except ValueError:
        st.error("Por favor, insira um número válido.")

# Mostrar histórico
st.markdown('<div class="historico">', unsafe_allow_html=True)
st.subheader("Histórico dos números inseridos")
if len(st.session_state.historico) == 0:
    st.write("Nenhum número inserido ainda.")
else:
    st.write(st.session_state.historico)
st.markdown('</div>', unsafe_allow_html=True)

# Mostrar previsão
proximo = prever_proximo()
if proximo is not None:
    st.success(f"Próximo número previsto pela IA: **{proximo}**")
else:
    st.info("Insira pelo menos 10 números para começar a prever.")

