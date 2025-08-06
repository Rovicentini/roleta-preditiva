import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# === Arquivos locais ===
history_path = "roleta_historico.csv"
model_path = "roleta_model_lstm.h5"
scaler_path = "scaler.pkl"

# === Funções base ===
def load_history():
    if os.path.exists(history_path):
        return pd.read_csv(history_path)
    else:
        return pd.DataFrame(columns=["Rodada", "Número"])

def save_history(df):
    df.to_csv(history_path, index=False)

def preprocess_data(sequence_length=10):
    """Prepara os dados para treino da rede LSTM"""
    numeros = historico["Número"].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    numeros_scaled = scaler.fit_transform(numeros)

    X, y = [], []
    for i in range(sequence_length, len(numeros_scaled)):
        X.append(numeros_scaled[i-sequence_length:i, 0])
        y.append(numeros_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    joblib.dump(scaler, scaler_path)
    return X, y, scaler

def criar_modelo(sequence_length=10):
    """Cria a rede neural LSTM"""
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(sequence_length, 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def treinar_modelo():
    if len(historico) < 20:
        return None  # poucos dados para treinar
    X, y, scaler = preprocess_data()
    model = criar_modelo()
    model.fit(X, y, epochs=100, batch_size=8, verbose=0)
    model.save(model_path)
    return model

def prever_proximos(qtd=5):
    if len(historico) < 20:
        return [], []
    scaler = joblib.load(scaler_path)
    numeros = historico["Número"].values.reshape(-1, 1)
    numeros_scaled = scaler.transform(numeros)

    ultima_seq = numeros_scaled[-10:].reshape((1, 10, 1))
    preds = []

    modelo = criar_modelo()
    modelo.load_weights(model_path)

    for _ in range(qtd):
        pred_scaled = modelo.predict(ultima_seq, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        pred_round = int(np.clip(round(pred), 0, 36))
        preds.append(pred_round)

        # atualiza sequência com previsão simulada
        nova_seq = np.append(ultima_seq[:, 1:, :], [[[pred_scaled[0][0]]]], axis=1)
        ultima_seq = nova_seq

    return preds, None

# === Interface Streamlit ===
st.set_page_config(page_title="🎰 Painel Inteligente da Roleta (LSTM)", layout="centered")

historico = load_history()
st.title("🎰 Painel Inteligente da Roleta com IA LSTM")

numero = st.number_input("Digite o número sorteado (0 a 36):", min_value=0, max_value=36, step=1)
if st.button("Adicionar Número"):
    nova_rodada = len(historico) + 1
    historico = pd.concat([historico, pd.DataFrame({"Rodada": [nova_rodada], "Número": [numero]})], ignore_index=True)
    save_history(historico)

    if len(historico) >= 20:
        st.info("Treinando modelo com base nos últimos resultados...")
        treinar_modelo()

    st.success(f"Número {int(numero)} adicionado com sucesso!")

    preds, _ = prever_proximos()
    if preds:
        st.subheader("🔮 Próximos números prováveis (IA LSTM)")
        st.write("➡️ " + ", ".join([str(p) for p in preds]))
    else:
        st.warning("Poucos dados para prever. Insira pelo menos 20 rodadas.")

    # Mostrar gráfico de frequência
    freq = historico["Número"].value_counts().reindex(range(37), fill_value=0)
    df = pd.DataFrame({"Número": range(37), "Frequência": freq})
    fig = px.bar(df, x="Número", y="Frequência", title="Heatmap de Frequência da Roleta")
    st.plotly_chart(fig)
