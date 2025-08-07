import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")
st.title("🔮 Previsor de Roleta Inteligente")

st.sidebar.header("🎰 Configurações")
num_vizinhos = st.sidebar.slider("Quantidade de vizinhos (cada lado):", 0, 5, 2)
total_vizinhos = num_vizinhos * 2 + 1

numero_roleta = [26, 3, 35, 12, 28, 7, 29, 18, 22, 9, 31, 14,
                 20, 1, 33, 16, 24, 5, 10, 23, 8, 30, 11, 36,
                 13, 27, 6, 34, 17, 25, 2, 21, 4, 19, 15, 32, 0]

historico = st.session_state.get("historico", [])
resultados = st.session_state.get("resultados", [])
acertos_erros = st.session_state.get("acertos_erros", [])

novo_numero = st.text_input("Digite o número sorteado:", key="input_numero")

# Lógica de vizinhos
def get_vizinhos(numero, vizinhos=2):
    if numero not in numero_roleta:
        return []
    idx = numero_roleta.index(numero)
    indices = [(idx + i) % len(numero_roleta) for i in range(-vizinhos, vizinhos + 1)]
    return [numero_roleta[i] for i in indices]

# Inicialização de rede neural
scaler = MinMaxScaler(feature_range=(0, 1))
model = st.session_state.get("model")
X_buffer = st.session_state.get("X_buffer", deque(maxlen=1000))
y_buffer = st.session_state.get("y_buffer", deque(maxlen=1000))

if model is None:
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    st.session_state.model = model

# Entrada do usuário
if st.button("Adicionar Número") and novo_numero != "":
    try:
        numero = int(novo_numero)
        if numero not in numero_roleta:
            st.warning("Número inválido. Digite um número de 0 a 36.")
        else:
            historico.append(numero)
            st.session_state.historico = historico

            # Previsão com LSTM
            if len(historico) >= 21:
                dados = np.array(historico[-21:]).reshape(-1, 1)
                dados = scaler.fit_transform(dados)
                X = dados[:-1].reshape(1, 20, 1)
                y = dados[-1].reshape(1, 1)

                # Atualiza o buffer
                X_buffer.append(X[0])
                y_buffer.append(y[0])

                # Treina o modelo continuamente
                if len(X_buffer) >= 20:
                    X_train = np.array(X_buffer)
                    y_train = np.array(y_buffer)
                    model.fit(X_train, y_train, epochs=1, batch_size=4, verbose=0)

                pred_scaled = model.predict(X, verbose=0)
                pred = scaler.inverse_transform(pred_scaled)[0][0]
                pred_mais_proximo = min(numero_roleta, key=lambda x: abs(x - pred))
                vizinhos = get_vizinhos(pred_mais_proximo, num_vizinhos)
                resultados.append({"previsto": pred_mais_proximo, "vizinhos": vizinhos, "real": numero})
                st.session_state.resultados = resultados

                # Avaliação do acerto
                acerto = int(numero in vizinhos)
                acertos_erros.append(acerto)
                st.session_state.acertos_erros = acertos_erros

    except ValueError:
        st.warning("Digite um número válido inteiro.")

# Exibição do histórico
st.subheader("📜 Histórico de Números")
st.write(historico[::-1])

# Última previsão
if resultados:
    st.subheader("🎯 Última Previsão da IA")
    ultima = resultados[-1]
    st.markdown(f"**Número Previsto:** {ultima['previsto']}")
    st.markdown(f"**Vizinhos Considerados ({num_vizinhos} de cada lado):** {ultima['vizinhos']}")
    st.markdown(f"**Número Real Sorteado:** {ultima['real']}")

# Visualização de desempenho
if acertos_erros:
    st.subheader("📊 Desempenho da IA")
    acertos = sum(acertos_erros)
    total = len(acertos_erros)
    taxa_acerto = (acertos / total) * 100
    st.markdown(f"**Taxa de Acerto:** {taxa_acerto:.2f}% ({acertos}/{total})")

    fig, ax = plt.subplots()
    ax.plot(acertos_erros, marker='o', linestyle='-', label="Acerto (1) / Erro (0)")
    ax.set_title("Evolução dos Acertos por Rodada")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Resultado")
    st.pyplot(fig)

    # Tabela detalhada
    st.markdown("### 📋 Detalhes por Rodada")
    df = pd.DataFrame(resultados)
    df['Acertou'] = ["✅" if x else "❌" for x in acertos_erros]
    st.dataframe(df[::-1], use_container_width=True)

# Probabilidade por número
if historico:
    st.subheader("📈 Análise Probabilística (Frequência dos Últimos 100)")
    df_freq = pd.DataFrame(historico[-100:], columns=["Número"])
    freq = df_freq.value_counts().reset_index()
    freq.columns = ["Número", "Frequência"]
    st.bar_chart(freq.set_index("Número"))

    mais_frequentes = freq.head(5)
    st.markdown("**Top 5 Números Mais Frequentes:**")
    st.write(mais_frequentes)
