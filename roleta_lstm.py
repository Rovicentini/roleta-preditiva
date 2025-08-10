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
from tensorflow.keras.layers import Attention, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping


# Configurações do TensorFlow para melhor performance
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Reduza a verbosidade
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
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
    X_seq, X_feat, y = [], [], []
    sequencia_roleta = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
    
    for i in range(len(historico) - sequencia - 1):
        seq = historico[i:i+sequencia]
        target = historico[i+sequencia]
        
        # Features sequenciais
        X_seq.append(seq)
        
        # Features adicionais
features = [
    float(np.mean(seq)),
    float(np.std(seq)),
    historico[-100:].count(target)/100 if len(historico) > 100 else 0,
    sequencia_roleta.index(seq[-1]) if seq[-1] in sequencia_roleta else 0  # Adicionado
]
        X_feat.append(features)
        y.append(target)
    
    if not X_seq:
        return None, None, None
    
    X_seq = np.array(X_seq).reshape((-1, sequencia, 1))
    X_feat = np.array(X_feat)
    y = to_categorical(y, num_classes=NUM_TOTAL)
    
    return [X_seq, X_feat], y

numeros_selecionados = []
probs = []
sugestoes_regressao = []
sugestoes_softmax = []


def treinar_modelo_lstm(historico):
    dados_enriquecidos = aumentar_dados(historico[-100:]) if len(historico) > 100 else aumentar_dados(historico)
    
    @st.cache_resource(ttl=3600)
    def train_cached_model(_historico):
        X, y = preparar_dados(_historico)  # Adicionado esta linha
        if X is None:
            return None
            
        # Modelo completo
        input_seq = Input(shape=(SEQUENCIA_ENTRADA, 1))  # Adicionado
        input_feat = Input(shape=(4,))  # Adicionado
        
        lstm1 = LSTM(64, return_sequences=True)(input_seq)
        attention = Attention()([lstm1, lstm1])  # Adicionado
        lstm2 = LSTM(32)(attention)
        dense1 = Dense(16, activation='relu')(input_feat)
        
        combined = Concatenate()([lstm2, dense1])  # Adicionado
        output = Dense(NUM_TOTAL, activation='softmax')(combined)  # Adicionado
        
        model = Model(inputs=[input_seq, input_feat], outputs=output)  # Adicionado
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy')  # Adicionado
        
        history = model.fit(
            [X[0], X[1]], y,  # Corrigido para usar X e y
            epochs=20,
            batch_size=32,
            callbacks=[EarlyStopping(patience=3)]
        )
        return model  # Adicionado retorno
    
    return train_cached_model(dados_enriquecidos)


# --- FUNÇÕES AUXILIARES ---

# Ordem dos números na roleta europeia no sentido horário
sequencia_roleta_europeia = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30,
    8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7,
    28, 12, 35, 3, 26
]

# --- ANTES (função não existia) ---
# Adicionar após sequencia_roleta_europeia

# --- DEPOIS ---
def aumentar_dados(historico):
    """Versão otimizada para roleta"""
    if len(historico) < 5:
        return historico
    
    # Apenas gera pequenas variações dos dados originais
    aumentados = list(historico)
    
    # Adiciona versão invertida
    aumentados.extend(historico[::-1])
    
    # Adiciona ruído controlado (±1)
    for n in historico:
        novo_num = n + np.random.choice([-1, 0, 1])
        aumentados.append(max(0, min(36, novo_num)))
    
    return aumentados
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

def pos_processamento(predictions, historico, n=3):
    """ITEM 4 - Função de pós-processamento inteligente"""
    # Peso pelos números quentes (mais frequentes nos últimos 100 jogos)
    freq = Counter(historico[-100:])
    max_freq = max(freq.values()) if freq else 1
    weighted_probs = []
    
    for i, prob in enumerate(predictions):
        # Peso pela frequência
        freq_weight = 0.5 + (freq.get(i, 0) / max_freq)
        
        # Peso pela posição física na roleta
        if historico:
            last_pos = sequencia_roleta_europeia.index(historico[-1])
            curr_pos = sequencia_roleta_europeia.index(i)
            dist = min(abs(curr_pos - last_pos), 37 - abs(curr_pos - last_pos))
            pos_weight = 1.5 - (dist / 18)  # Diminui o peso conforme a distância
            
        weighted_probs.append(prob * freq_weight * pos_weight)
    
    # Normaliza
    weighted_probs = np.array(weighted_probs)
    weighted_probs /= weighted_probs.sum()
    
    # Retorna os top N números
    top_indices = np.argsort(weighted_probs)[-n:]
    return [(i, weighted_probs[i]) for i in top_indices]

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

    
    
def calcular_performance():
    if not hasattr(st.session_state, 'resultados') or not st.session_state.resultados:
        return 0, 0
    
    # Considera apenas os últimos 50 resultados para cálculo
    recentes = st.session_state.resultados[-50:]
    acertos = sum(1 for r in recentes if r.get('acerto', False))
    return acertos, len(recentes) - acertos


@st.cache_data(ttl=300)  # Cache de 5 minutos
def fazer_previsao(_model, _seq, _feat):
    if _model is None:
        return np.zeros(NUM_TOTAL)
    return _model.predict([_seq, _feat], verbose=0, batch_size=1)[0]  # Batch_size otimizado

# --- SIDEBAR ---

if st.sidebar.button("🔁 Reiniciar Tudo"):
    st.session_state.historico = []
    st.session_state.resultados = []
    st.session_state.modelo_treinado = False
    
with st.sidebar.expander("⚙️ Configurações Avançadas"):
    st.slider("Número de sugestões", 1, 5, 3, key='n_sugestoes')
    st.checkbox("Usar padrões físicos", True, key='usar_fisica')
    st.checkbox("Considerar frequência", True, key='usar_frequencia')

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
# CLASSIFICAÇÃO
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 1:
    with st.spinner('Analisando padrões...'):  # Feedback visual
        model_classificacao = treinar_modelo_lstm(st.session_state.historico)
    if model_classificacao:
        # Prepara os dados de entrada
        seq_data = np.array(st.session_state.historico[-SEQUENCIA_ENTRADA:]).reshape(1, SEQUENCIA_ENTRADA, 1)
        
        # Features adicionais
        feat_data = np.array([[
            np.mean(st.session_state.historico[-SEQUENCIA_ENTRADA:]),
            np.std(st.session_state.historico[-SEQUENCIA_ENTRADA:]),
            Counter(st.session_state.historico).most_common(1)[0][1]/len(st.session_state.historico),
            sequencia_roleta_europeia.index(st.session_state.historico[-1])
        ]])
        
        # Faz a predição
        predicao_softmax = fazer_previsao(model_classificacao, seq_data, feat_data)
        
        sugestoes_com_vizinhos = pos_processamento(predicao_softmax, st.session_state.historico)



# Sugestão de número + quantidade de vizinhos recomendada pela IA


# --- EXIBIR SUGESTÕES ---
st.subheader("📈 Sugestão de Apostas da IA")
st.write("🔢 **Sugestão de números (Regressão):**", sugestoes_regressao)

st.subheader("🎯 Sugestões Inteligentes (Foco Qualidade)")
if sugestoes_com_vizinhos:
    for num, prob in sorted(sugestoes_com_vizinhos, key=lambda x: x[1], reverse=True)[:st.session_state.n_sugestoes]:
        vizinhos = obter_vizinhos_roleta(num, 1) if st.session_state.usar_fisica else []
        st.markdown(
            f"- **Número {num}** (Confiança: {prob:.1%})" +
            (f" + vizinhos: {vizinhos}" if vizinhos else "")
        )
        
    # Validação em tempo real
    if st.session_state.historico:
        ultimo_numero = st.session_state.historico[-1]
        acerto = any(num == ultimo_numero for num, _ in sugestoes_com_vizinhos)
        st.session_state.resultados.append(acerto)
        
        # Corrigir para:
if sugestoes_com_vizinhos:
    for num, prob in sorted(sugestoes_com_vizinhos, key=lambda x: x[1], reverse=True)[:st.session_state.n_sugestoes]:
        # ... (código de exibição das sugestões)
    
    # Validação em tempo real (mesmo nível do for)
    if st.session_state.historico:
        ultimo_numero = st.session_state.historico[-1]
        acerto = any(num == ultimo_numero for num, _ in sugestoes_com_vizinhos)
        st.session_state.resultados.append(acerto)
        
        # Verificação de performance (agora corretamente indentada)
        if len(st.session_state.resultados) > 10:
            ultimos_resultados = [r for r in st.session_state.resultados[-20:] if isinstance(r, dict)]
            if ultimos_resultados:
                taxa_acerto = np.mean([r.get('acerto', False) for r in ultimos_resultados])
                st.metric("Taxa de Acerto (Últimos 20)", f"{taxa_acerto:.1%}")
            
else:
    st.warning("Nenhuma sugestão com confiança suficiente hoje.")



# --- AVALIAÇÃO DE DESEMPENHO ---
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA + 2:
    try:
        # Inicialização segura de todas as variáveis
        ultimo_numero = st.session_state.historico[-1]
        acerto_classificacao = False
        acerto_regressao = False
        numeros_sugeridos = []
        
        # Verificação segura das sugestões
        if 'sugestoes_com_vizinhos' in globals() and sugestoes_com_vizinhos:
            numeros_sugeridos = [num for num, _, _ in sugestoes_com_vizinhos]
            acerto_classificacao = ultimo_numero in numeros_sugeridos
        
        # Verificação segura da regressão
        if 'sugestoes_regressao' in globals():
            acerto_regressao = ultimo_numero in sugestoes_regressao if sugestoes_regressao else False
        
        # Armazenamento do resultado
        st.session_state.resultados.append({
            'real': ultimo_numero,
            'previsto_class': numeros_sugeridos,
            'previsto_reg': sugestoes_regressao if 'sugestoes_regressao' in globals() else [],
            'acerto': acerto_classificacao
        })
        
        # Exibição dos resultados
        st.write(f"🔮 Último número: {ultimo_numero}")
        if numeros_sugeridos:
            st.write(f"🧠 IA Classificou: {numeros_sugeridos} → {'✅' if acerto_classificacao else '❌'}")
        if 'sugestoes_regressao' in globals() and sugestoes_regressao:
            st.write(f"📈 IA Regrediu: {sugestoes_regressao} → {'✅' if acerto_regressao else '❌'}")
        
        # Cálculo das estatísticas
        if hasattr(st.session_state, 'resultados') and st.session_state.resultados:
            acertos = sum(1 for r in st.session_state.resultados if isinstance(r, dict) and r.get('acerto', False))
            total = len(st.session_state.resultados)
            if total > 0:
                st.sidebar.markdown(f"""
                    **Estatísticas:**
                    - ✅ Acertos: {acertos}
                    - ❌ Erros: {total - acertos}
                    - 🎯 Precisão: {acertos/total:.1%}
                """)

    except Exception as e:
        st.error(f"Erro ao processar resultados: {str(e)}")

elif not st.session_state.historico:
    st.info("⏳ Histórico vazio. Insira números para começar.")
else:
    faltam = max(0, SEQUENCIA_ENTRADA + 2 - len(st.session_state.historico))
    st.info(f"📥 Insira mais {faltam} número(s) para ativar as previsões")


































































