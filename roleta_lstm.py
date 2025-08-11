# Roleta IA Robusta com LSTM, TensorFlow, Análise de Tendência, e Visualização
# Autor: Rodrigo Vicentini
# Versão Corrigida e Otimizada por: [Seu Nome]
# Requisitos: pip install streamlit tensorflow scikit-learn pandas matplotlib numpy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Concatenate
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- CONFIGURAÇÃO INICIAL ---
st.set_page_config(layout="wide")
st.title("🎯 IA Avançada para Roleta Europeia")

# --- VARIÁVEIS GLOBAIS ---
NUM_TOTAL = 37  # Números da Roleta Europeia: 0 a 36
SEQUENCIA_ENTRADA = 10  # Quantos números analisar por entrada

# Ordem dos números na roleta europeia no sentido horário
SEQUENCIA_ROlETA_EUROPEIA = [
    0, 32, 15, 19, 4, 21, 2, 25, 17, 34, 6, 27, 13, 36, 11, 30,
    8, 23, 10, 5, 24, 16, 33, 1, 20, 14, 31, 9, 22, 18, 29, 7,
    28, 12, 35, 3, 26
]

# Inicialização do session_state
if 'historico' not in st.session_state:
    st.session_state.historico = []

if 'resultados' not in st.session_state:
    st.session_state.resultados = []

if 'modelo_treinado' not in st.session_state:
    st.session_state.modelo_treinado = False

# --- FUNÇÕES AUXILIARES ---
def obter_vizinhos_roleta(numero, quantidade_vizinhos=1):
    """Retorna os vizinhos físicos na roleta"""
    if numero not in SEQUENCIA_ROlETA_EUROPEIA:
        return []
    
    idx = SEQUENCIA_ROlETA_EUROPEIA.index(numero)
    vizinhos = []
    
    for i in range(1, quantidade_vizinhos + 1):
        vizinhos.append(SEQUENCIA_ROlETA_EUROPEIA[(idx - i) % len(SEQUENCIA_ROlETA_EUROPEIA)])
        vizinhos.append(SEQUENCIA_ROlETA_EUROPEIA[(idx + i) % len(SEQUENCIA_ROlETA_EUROPEIA)])
    
    return list(set(vizinhos))  # Remove duplicatas

def aumentar_dados(historico):
    """Aumenta os dados de treinamento de forma conservadora"""
    if len(historico) < 5:
        return historico
    
    aumentados = list(historico)
    
    # Adiciona pequenas variações com ruído controlado
    for n in historico:
        novo_num = n + np.random.choice([-1, 0, 1])
        novo_num = max(0, min(36, novo_num))
        aumentados.append(novo_num)
    
    return aumentados

def preparar_dados(historico, sequencia=SEQUENCIA_ENTRADA):
    """Prepara os dados para treinamento do modelo"""
    X_seq, X_feat, y = [], [], []
    
    if len(historico) < sequencia + 1:
        return None, None
    
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
            SEQUENCIA_ROlETA_EUROPEIA.index(seq[-1]) if seq[-1] in SEQUENCIA_ROlETA_EUROPEIA else 0
        ]
        X_feat.append(features)
        y.append(target)
    
    if not X_seq:
        return None, None
    
    X_seq = np.array(X_seq).reshape((-1, sequencia, 1))
    X_feat = np.array(X_feat)
    y = to_categorical(y, num_classes=NUM_TOTAL)
    
    return [X_seq, X_feat], y

def criar_modelo():
    """Cria um modelo mais simples e eficiente"""
    input_seq = Input(shape=(SEQUENCIA_ENTRADA, 1))
    input_feat = Input(shape=(4,))
    
    # Camada LSTM principal
    lstm_out = LSTM(32)(input_seq)
    
    # Camada para features adicionais
    dense_feat = Dense(8, activation='relu')(input_feat)
    
    # Combina as features
    combined = Concatenate()([lstm_out, dense_feat])
    output = Dense(NUM_TOTAL, activation='softmax')(combined)
    
    model = Model(inputs=[input_seq, input_feat], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def treinar_modelo(historico):
    """Treina o modelo com validação cruzada temporal"""
    if len(historico) < SEQUENCIA_ENTRADA * 2:
        st.warning(f"Necessário ao menos {SEQUENCIA_ENTRADA * 2} números para treinar")
        return None
    
    try:
        # Aumenta os dados de treinamento
        dados_aumentados = aumentar_dados(historico)
        
        # Prepara os dados
        X, y = preparar_dados(dados_aumentados)
        if X is None:
            return None
        
        # Cria e treina o modelo
        model = criar_modelo()
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        model.fit([X[0], X[1]], y, 
                 epochs=20, 
                 batch_size=16,
                 validation_split=0.2,
                 callbacks=[early_stop],
                 verbose=0)
        
        st.session_state.modelo_treinado = True
        return model
        
    except Exception as e:
        st.error(f"Erro ao treinar modelo: {str(e)}")
        return None

def pos_processamento(predictions, historico, n=3):
    """Aplica pós-processamento inteligente às predições"""
    # Peso pelos números quentes
    freq = Counter(historico[-100:]) if len(historico) > 10 else {}
    max_freq = max(freq.values()) if freq else 1
    
    weighted_probs = []
    for i, prob in enumerate(predictions):
        # Peso pela frequência
        freq_weight = 0.5 + (freq.get(i, 0) / max_freq)
        
        # Peso pela posição física na roleta
        if historico and historico[-1] in SEQUENCIA_ROlETA_EUROPEIA:
            last_pos = SEQUENCIA_ROlETA_EUROPEIA.index(historico[-1])
            curr_pos = SEQUENCIA_ROlETA_EUROPEIA.index(i) if i in SEQUENCIA_ROlETA_EUROPEIA else last_pos
            dist = min(abs(curr_pos - last_pos), 37 - abs(curr_pos - last_pos))
            pos_weight = 1.5 - (dist / 18)
        else:
            pos_weight = 1.0
        
        weighted_probs.append(prob * freq_weight * pos_weight)
    
    # Normaliza as probabilidades
    weighted_probs = np.array(weighted_probs)
    weighted_probs /= weighted_probs.sum()
    
    # Retorna os top N números
    top_indices = np.argsort(weighted_probs)[-n:]
    return [(i, weighted_probs[i]) for i in reversed(top_indices)]

def fazer_previsao(model, historico):
    """Faz a previsão usando o modelo treinado"""
    if model is None or len(historico) < SEQUENCIA_ENTRADA:
        return []
    
    # Prepara os dados de entrada
    seq_data = np.array(historico[-SEQUENCIA_ENTRADA:]).reshape(1, SEQUENCIA_ENTRADA, 1)
    
    # Features adicionais
    feat_data = np.array([[
        np.mean(historico[-SEQUENCIA_ENTRADA:]),
        np.std(historico[-SEQUENCIA_ENTRADA:]),
        Counter(historico).most_common(1)[0][1]/len(historico) if historico else 0,
        SEQUENCIA_ROlETA_EUROPEIA.index(historico[-1]) if historico[-1] in SEQUENCIA_ROlETA_EUROPEIA else 0
    ]])
    
    # Faz a predição
    predictions = model.predict([seq_data, feat_data], verbose=0)[0]
    return predictions

def avaliar_performance():
    """Calcula e exibe métricas de performance"""
    if not st.session_state.resultados:
        return 0.0
    
    acertos = sum(1 for r in st.session_state.resultados if r['acerto'])
    total = len(st.session_state.resultados)
    esperado = total * (st.session_state.n_sugestoes / 37)
    
    st.sidebar.markdown(f"""
    **📊 Estatísticas:**
    - ✅ Acertos: {acertos} (Esperado: {esperado:.1f})
    - ❌ Erros: {total - acertos}
    - 🎯 Precisão: {acertos/total:.1%} (Esperado: {st.session_state.n_sugestoes/37:.1%})
    """)
    
    return acertos / total if total > 0 else 0.0

# --- INTERFACE DO USUÁRIO ---

# SIDEBAR
st.sidebar.title("Configurações")

if st.sidebar.button("🔁 Reiniciar Tudo"):
    st.session_state.historico = []
    st.session_state.resultados = []
    st.session_state.modelo_treinado = False

with st.sidebar.expander("⚙️ Opções Avançadas"):
    st.slider("Número de sugestões", 1, 5, 3, key='n_sugestoes')
    st.checkbox("Usar padrões físicos", True, key='usar_fisica')
    st.checkbox("Considerar frequência", True, key='usar_frequencia')

# ENTRADA DE DADOS
st.subheader("🎰 Inserir Número da Roleta")

def adicionar_numero_callback():
    numero = st.session_state.entrada_numero
    if numero.isdigit():
        n = int(numero)
        if 0 <= n <= 36:
            st.session_state.historico.append(n)
            st.session_state.entrada_numero = ""
            
            # Atualiza resultados se tivermos previsões anteriores
            if hasattr(st.session_state, 'ultimas_sugestoes'):
                ultimo_numero = n
                acerto = any(num == ultimo_numero for num, _ in st.session_state.ultimas_sugestoes)
                st.session_state.resultados.append({'acerto': acerto, 'numero': ultimo_numero})

st.text_input("Digite o número sorteado (0 a 36):", 
             key="entrada_numero", 
             on_change=adicionar_numero_callback)

# EXIBIÇÃO DO HISTÓRICO
st.subheader("📜 Histórico de Números")
if st.session_state.historico:
    st.write(" ".join([f"**{num}**" for num in st.session_state.historico[-50:]]))
else:
    st.info("Nenhum número inserido ainda.")

# TREINAMENTO E PREVISÃO
if len(st.session_state.historico) >= SEQUENCIA_ENTRADA * 2:
    with st.spinner('🔍 Analisando padrões...'):
        modelo = treinar_modelo(st.session_state.historico)
        
        if modelo:
            predictions = fazer_previsao(modelo, st.session_state.historico)
            sugestoes = pos_processamento(predictions, st.session_state.historico, 
                                        n=st.session_state.n_sugestoes)
            
            if sugestoes:
                st.session_state.ultimas_sugestoes = sugestoes
                
                st.subheader("🎯 Sugestões de Apostas")
                for num, prob in sugestoes:
                    vizinhos = []
                    if st.session_state.usar_fisica and prob > 0.1:  # Só sugere vizinhos se confiança > 10%
                        vizinhos = obter_vizinhos_roleta(num, 1)
                    
                    st.markdown(f"- **Número {num}** (Confiança: {prob:.1%})" +
                               (f" + vizinhos: {vizinhos}" if vizinhos else ""))
else:
    st.warning(f"⏳ Insira mais {max(0, SEQUENCIA_ENTRADA*2 - len(st.session_state.historico))} números para ativar as previsões")

# AVALIAÇÃO DE PERFORMANCE
if st.session_state.resultados:
    avaliar_performance()
    
    # Exibe os últimos resultados
    st.subheader("📈 Últimos Resultados")
    cols = st.columns(5)
    for i, res in enumerate(st.session_state.resultados[-10:]):
        cols[i%5].metric(
            label=f"Número {res['numero']}",
            value="✅" if res['acerto'] else "❌"
        )

# NOTAS IMPORTANTES
st.markdown("---")
st.info("""
**📝 Notas Importantes:**
1. A roleta é um jogo de aleatoriedade pura - nenhum sistema pode garantir vitórias consistentes
2. Este sistema identifica padrões em sequências limitadas, mas não pode prever resultados com certeza
3. Use apenas para fins educacionais e de entretenimento
""")
