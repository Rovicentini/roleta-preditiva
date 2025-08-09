# Roleta IA Robusta com LSTM, TensorFlow, Análise de Tendência, e Visualização
# Autor: Rodrigo Vicentini
# Requisitos: pip install streamlit tensorflow scikit-learn pandas matplotlib numpy
# Configuração do TensorFlow para evitar warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')


import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide")
st.title("🎯 IA Avançada para Roleta Europeia")

# --- VARIÁVEIS GLOBAIS ---
NUM_TOTAL = 37  # Números da Roleta Europeia: 0 a 36
SEQUENCIA_ENTRADA = 10  # Quantos números analisar por entrada

if 'historico' not in st.session_state:
    st.session_state.historico = []

if 'resultados' not in st.session_state:
    st.session_state.resultados = []

if 'vizinhanca_map' not in st.session_state:
    st.session_state.vizinhanca_map = {
        0: [26, 32, 3],
        1: [20, 33, 14],
        2: [17, 25, 21],
        3: [35, 26, 0],
        4: [19, 21, 2],
        5: [10, 24, 16],
        6: [27, 13, 34],
        7: [28, 12, 29],
        8: [30, 11, 23],
        9: [22, 18, 31],
        10: [5, 23, 24],
        11: [8, 30, 36],
        12: [7, 36, 3],
        13: [6, 34, 27],
        14: [1, 33, 20],
        15: [19, 32, 19],
        16: [5, 24, 33],
        17: [2, 25, 34],
        18: [9, 22, 29],
        19: [4, 15, 21],
        20: [1, 14, 33],
        21: [4, 2, 19],
        22: [9, 18, 31],
        23: [10, 8, 30],
        24: [10, 5, 16],
        25: [2, 17, 34],
        26: [3, 0, 32],
        27: [6, 13, 36],
        28: [7, 29, 12],
        29: [7, 28, 18],
        30: [8, 11, 23],
        31: [9, 22, 18],
        32: [26, 0, 15],
        33: [1, 14, 20],
        34: [6, 13, 25],
        35: [3, 12, 6],
        36: [11, 12, 27],
    }


if 'modelo_treinado' not in st.session_state:
    st.session_state.modelo_treinado = False


def preparar_dados(historico, sequencia=SEQUENCIA_ENTRADA):
    X, y = [], []
    for i in range(len(historico) - sequencia):
        seq_in = historico[i:i + sequencia]
        seq_out = historico[i + sequencia]
        X.append(seq_in)
        y.append(seq_out)
    if not X or not y:
        return None, None
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))  # necessário para LSTM
    y = to_categorical(y, num_classes=NUM_TOTAL)  # one-hot
    return X, y
def aumentar_dados(historico):
    sequencia_roleta = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
    dados_aumentados = []
    for i in range(len(historico)-1):
        num = historico[i]
        idx = sequencia_roleta.index(num)
        vizinhos = [
            sequencia_roleta[(idx-1) % 37],
            sequencia_roleta[(idx+1) % 37]
        ]
        dados_aumentados.extend([num] + vizinhos)
    return dados_aumentados[-500:]  # Limita ao último 500 itens

numeros_selecionados = []
probs = []
sugestoes_regressao = []
sugestoes_softmax = []


def treinar_modelo_lstm(historico, sequencia=SEQUENCIA_ENTRADA):
    dados_enriquecidos = aumentar_dados(historico)  # Usa a nova função de aumento
    X, y = preparar_dados(dados_enriquecidos, sequencia)
    if X is None or y is None:
        return None
        
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(NUM_TOTAL, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    model.fit(X, y, epochs=50, batch_size=16, verbose=0)
    return model




# --- FUNÇÕES AUXILIARES ---

# Ordem dos números na roleta europeia no sentido horário
sequencia_roleta_europeia = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30,
    8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7,
    28, 12, 35, 3, 26
]

def obter_vizinhos_roleta(numero, quantidade_vizinhos=3):
    numero = int(numero)
    if numero not in sequencia_roleta_europeia:
        return []

    idx = sequencia_roleta_europeia.index(numero)
    vizinhos = []

    for i in range(1, quantidade_vizinhos + 1):
        vizinhos.append(sequencia_roleta_europeia[(idx - i) % len(sequencia_roleta_europeia)])
        vizinhos.append(sequencia_roleta_europeia[(idx + i) % len(sequencia_roleta_europeia)])

    return sorted(set(vizinhos))  # Elimina duplicatas, se houver

def calcular_vizinhos(prob):
    """Retorna menos vizinhos quando a confiança não é muito alta"""
    if prob > 0.3:  # Se probabilidade > 30%
        return 2  # Máximo 2 vizinhos
    elif prob > 0.2:
        return 1  # Máximo 1 vizinho
    else:
        return 0  # Sem vizinhos


# --- FUNÇÕES ---
def adicionar_numero(numero):
    try:
        n = int(numero)
        if 0 <= n < NUM_TOTAL:
            st.session_state.historico.append(n)
    except:
        pass

def get_dados_normalizados():
    dados = np.array(st.session_state.historico).reshape(-1, 1)
    scaler = MinMaxScaler()
    dados_norm = scaler.fit_transform(dados)
    return dados_norm, scaler

def criar_dados_sequenciais(dados_norm):
    X, y = [], []
    for i in range(len(dados_norm) - SEQUENCIA_ENTRADA):
        X.append(dados_norm[i:i+SEQUENCIA_ENTRADA])
        y.append(dados_norm[i+SEQUENCIA_ENTRADA])
    return np.array(X), np.array(y)

def treinar_modelo():
    # Verifica se há dados suficientes
    if len(st.session_state.historico) < SEQUENCIA_ENTRADA + 5:  # Buffer adicional
        st.warning(f"Necessário ao menos {SEQUENCIA_ENTRADA + 5} números para treinar")
        return None, None
    
    try:
        dados_norm, scaler = get_dados_normalizados()
        X, y = criar_dados_sequenciais(dados_norm)
        
        if len(X) == 0 or len(y) == 0:
            return None, None

        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(SEQUENCIA_ENTRADA, 1)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        
        # Adicionando callback para evitar erros
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        model.fit(X, y, epochs=100, verbose=0, callbacks=[early_stop])
        
        st.session_state.modelo_treinado = True
        return model, scaler
        
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None, None


def prever_proximo(modelo, scaler):
    if not modelo or not scaler or len(st.session_state.historico) < SEQUENCIA_ENTRADA:
        return []

    try:
        ultimos = st.session_state.historico[-SEQUENCIA_ENTRADA:]
        entrada = np.array(ultimos).reshape(-1, 1)
        entrada_norm = scaler.transform(entrada)
        entrada_norm = entrada_norm.reshape(1, SEQUENCIA_ENTRADA, 1)
        pred_norm = modelo.predict(entrada_norm, verbose=0)
        pred = scaler.inverse_transform(pred_norm)
        return [int(np.round(pred[0][0]))]
    except:
        return []

    
    sugestoes.extend(vizinhos)

    sugestoes = sorted(set(sugestoes))  # Remove duplicatas e ordena

    return sugestoes


def calcular_performance():
    if not hasattr(st.session_state, 'resultados') or not st.session_state.resultados:
        return 0, 0
    
    resultados = st.session_state.resultados[-50:]  # Pega os últimos 50 resultados
    
    acertos = sum(1 for r in resultados if isinstance(r, dict) and r.get('acerto', False))
    total = len(resultados)
    
    return acertos, total - acertos

# --- SIDEBAR ---

if st.sidebar.button("🔁 Reiniciar Tudo"):
    st.session_state.historico = []
    st.session_state.resultados = []
    st.session_state.modelo_treinado = False

# --- INTERFACE PRINCIPAL ---
st.subheader("🎰 Inserir Número da Roleta")
def adicionar_numero_callback():
    numero = st.session_state.entrada_numero
    if numero != "":  # ✅ Agora aceita repetições!
        adicionar_numero(numero)
        st.session_state.ultima_entrada = numero  # Opcional: mantém registro do último número
        st.session_state.entrada_numero = ""  # limpa o campo

st.text_input("Digite o número sorteado (0 a 36):", key="entrada_numero", on_change=adicionar_numero_callback)

# --- EXIBIR HISTÓRICO ---
st.subheader("📜 Histórico")
if st.session_state.historico:
    st.markdown(" ".join([f"**{num}**" for num in st.session_state.historico[::-1]]))
else:
    st.info("Nenhum número inserido ainda.")

# --- TREINAR E PREVER ---
    sugestoes_regressao = []
    sugestoes_softmax = []

# Apenas se houver dados suficientes
# --- TREINAR E PREVER ---
# ✅ CÓDIGO CORRIGIDO ✅
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 5:  # Novo limite seguro
    model_regressao, scaler = treinar_modelo()
    if model_regressao and scaler:  # Verifica se o modelo é válido
        sugestoes_regressao = prever_proximo(model_regressao, scaler)
else:
    st.warning(f"Aguarde até ter {SEQUENCIA_ENTRADA + 5} números no histórico")

  # CLASSIFICAÇÃO
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 1:
    model_classificacao = treinar_modelo_lstm(st.session_state.historico)
    if model_classificacao:
        entrada = np.array(st.session_state.historico[-SEQUENCIA_ENTRADA:]).reshape(1, SEQUENCIA_ENTRADA, 1)
        predicao_softmax = model_classificacao.predict(entrada, verbose=0)
        probs = predicao_softmax[0]

        # TOP 3 números com maior probabilidade:
        top_n = 3  # Você pode ajustar para 2 ou 1 se quiser menos sugestões
        indices_ordenados = np.argsort(probs)[-top_n:]  # Pega os top_n mais prováveis
        numeros_selecionados = [i for i in indices_ordenados if probs[i] > np.mean(probs)]
    else:
        st.warning("Modelo de classificação não foi treinado por falta de dados.")
else:  # ✅ CORRETO - else alinhado com o primeiro if
    st.info(f"ℹ️ Insira ao menos {SEQUENCIA_ENTRADA + 1} números para iniciar a previsão com IA.")




# Sugestão de número + quantidade de vizinhos recomendada pela IA
# ------ INÍCIO DA MODIFICAÇÃO (ITENS 3 E 4) ------
# ------ NOVO BLOCO (ITENS 3 E 4) ------
def filtrar_sugestoes(probs, min_conf=0.2):  # Item 3 - Filtro inteligente
    top5_idx = np.argsort(probs)[-5:]  # Pega os 5 mais prováveis
    sugestoes = []
    for idx in top5_idx:
        prob = probs[idx]
        if prob >= min_conf:  # Filtra por confiança mínima
            qtd_vizinhos = 2 if prob > 0.3 else (1 if prob > 0.2 else 0)
            sugestoes.append((idx, qtd_vizinhos, prob))  # Item 4 - Adiciona confiança
    return sorted(sugestoes, key=lambda x: x[2], reverse=True)  # Ordena por confiança

# Aplicação:
sugestoes_com_vizinhos = filtrar_sugestoes(predicao_softmax[0]) if 'predicao_softmax' in locals() else []
# -------------------------------------

# ------ FIM DA MODIFICAÇÃO ------

# Ordenar por probabilidade decrescente
sugestoes_com_vizinhos = sorted(sugestoes_com_vizinhos, key=lambda x: probs[x[0]], reverse=True)


    # --- EXIBIR SUGESTÕES ---
# --- EXIBIR SUGESTÕES ---
st.subheader("📈 Sugestão de Apostas da IA")
st.write("🔢 **Sugestão de números (Regressão):**", sugestoes_regressao)

st.subheader("🎯 Sugestões Inteligentes (Foco Qualidade)")
if sugestoes_com_vizinhos:
    for num, qtd_viz, prob in sugestoes_com_vizinhos[:3]:  # Mostra até 3 sugestões
        st.markdown(
            f"- **Número {num}** (Confiança: {prob:.1%})" + 
            (f" + {qtd_viz} vizinho(s)" if qtd_viz > 0 else "")
        )
        
    # Validação em tempo real
    if st.session_state.historico:
        ultimo_numero = st.session_state.historico[-1]
        acerto = any(num == ultimo_numero for num, _, _ in sugestoes_com_vizinhos)
        st.session_state.resultados.append(acerto)
        
        if len(st.session_state.resultados) > 10:
            taxa_acerto = np.mean(st.session_state.resultados[-20:])
            st.metric("Taxa de Acerto (Últimos 20)", f"{taxa_acerto:.1%}")
else:
    st.warning("Nenhuma sugestão com confiança suficiente hoje.")



# --- AVALIAÇÃO DE DESEMPENHO ---
# --- AVALIAÇÃO DE DESEMPENHO ---
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 2:
    # Garante que todas as variáveis existam
    ultimo_numero = st.session_state.historico[-1]
    acerto_classificacao = False
    numeros_sugeridos = []
    
    # Verifica sugestões de classificação
    if 'sugestoes_com_vizinhos' in locals() and sugestoes_com_vizinhos:
        numeros_sugeridos = [num for num, _, _ in sugestoes_com_vizinhos]
        acerto_classificacao = ultimo_numero in numeros_sugeridos
    
    # Verifica sugestões de regressão
    acerto_regressao = ultimo_numero in sugestoes_regressao if sugestoes_regressao else False
    
    # Armazena resultado de forma segura
    resultado = {
        'real': ultimo_numero,
        'previsto_class': numeros_sugeridos,
        'previsto_reg': sugestoes_regressao,
        'acerto': acerto_classificacao
    }
    st.session_state.resultados.append(resultado)
    
    # Exibe resultados
    st.write(f"🔮 Último número: {ultimo_numero}")
    st.write(f"🧠 IA Classificou: {numeros_sugeridos} → {'✅' if acerto_classificacao else '❌'}")
    st.write(f"📈 IA Regrediu: {sugestoes_regressao} → {'✅' if acerto_regressao else '❌'}")
    
    # Atualiza estatísticas
    acertos, erros = calcular_performance()
    st.sidebar.markdown(f"""
        **Estatísticas:**
        - Acertos: {acertos}
        - Erros: {erros}
        - Precisão: {acertos/(acertos+erros)*100:.1f}%
    """)

elif not st.session_state.historico:
    st.info("⏳ Histórico vazio. Insira números para começar.")
else:
    st.info(f"📥 Insira mais {SEQUENCIA_ENTRADA + 2 - len(st.session_state.historico)} números para análise.")
        # Atualiza estatísticas
        acertos, erros = calcular_performance()
        st.sidebar.markdown(f"""
            **Performance (Últimos 50):**  
            ✅ Acertos: {acertos}  
            ❌ Erros: {erros}  
            🎯 Precisão: {acertos/(acertos+erros)*100:.1f}%
        """)
        
    except Exception as e:
        st.error(f"Erro na avaliação: {str(e)}")

elif not st.session_state.historico:
    st.info("ℹ️ Histórico vazio. Insira números para começar.")
else:
    st.info(f"ℹ️ Insira mais {SEQUENCIA_ENTRADA + 2 - len(st.session_state.historico)} números para análise.")





























































