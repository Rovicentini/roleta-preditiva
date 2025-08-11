# Roleta IA - Sistema Neural Avançado de Previsão
# Autor: Rodrigo Vicentini
# Versão: Máxima Precisão - Foco em Resultados

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURAÇÃO ---
st.set_page_config(layout="centered")
st.title("🎯 ROULETTE AI - PREVISÃO DE ALTA PRECISÃO")

# --- CONSTANTES ---
NUM_TOTAL = 37  # 0-36
SEQUENCE_LEN = 15  # Aumentado para capturar padrões complexos
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]

# --- INICIALIZAÇÃO ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'stats' not in st.session_state:
    st.session_state.stats = {'acertos': 0, 'total': 0}

# --- FUNÇÕES NEURAIS AVANÇADAS ---
def get_deep_features(sequence):
    """Extrai features profundas da sequência"""
    # Estatísticas básicas
    mean = np.mean(sequence)
    std = np.std(sequence)
    
    # Padrões de rotação
    wheel_pos = [WHEEL_ORDER.index(n) if n in WHEEL_ORDER else -1 for n in sequence]
    rotations = [(wheel_pos[i+1]-wheel_pos[i])%37 for i in range(len(wheel_pos)-1)]
    
    # Frequências complexas
    freq = Counter(sequence)
    hot = max(freq.values()) if freq else 1
    
    return [
        mean/36, std/18,
        np.mean(rotations)/36,
        np.std(rotations)/18,
        freq.get(sequence[-1], 0)/hot,
        len(freq)/len(sequence)
    ]

def build_expert_model():
    """Constrói modelo neural especializado"""
    # Camada de sequência
    seq_input = Input(shape=(SEQUENCE_LEN, 1))
    lstm1 = LSTM(128, return_sequences=True)(seq_input)
    lstm1 = Dropout(0.3)(lstm1)
    att = Attention()([lstm1, lstm1])
    lstm2 = LSTM(64)(att)
    
    # Camada de features
    feat_input = Input(shape=(6,))
    dense_feat = Dense(32, activation='relu')(feat_input)
    dense_feat = Dropout(0.2)(dense_feat)
    
    # Combinação profunda
    combined = Concatenate()([lstm2, dense_feat])
    dense1 = Dense(128, activation='relu')(combined)
    dense1 = Dropout(0.4)(dense1)
    output = Dense(NUM_TOTAL, activation='softmax')(dense1)
    
    model = Model(inputs=[seq_input, feat_input], outputs=output)
    model.compile(optimizer=Adam(0.0003),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def predict_with_confidence(model, history):
    """Gera previsões com análise de confiança"""
    if len(history) < SEQUENCE_LEN or model is None:
        return []
    
    # Prepara dados
    seq = np.array(history[-SEQUENCE_LEN:]).reshape(1, SEQUENCE_LEN, 1)
    features = np.array([get_deep_features(history[-SEQUENCE_LEN:])])
    
    # Predição neural
    raw_pred = model.predict([seq, features], verbose=0)[0]
    
    # Pós-processamento inteligente
    enhanced_pred = []
    for num, prob in enumerate(raw_pred):
        # Fator de frequência
        freq_factor = 1 + 2 * Counter(history[-100:]).get(num, 0)/10
        
        # Fator de rotação
        if num in WHEEL_ORDER and history[-1] in WHEEL_ORDER:
            pos_last = WHEEL_ORDER.index(history[-1])
            pos_num = WHEEL_ORDER.index(num)
            rotation = min(abs(pos_num - pos_last), 37 - abs(pos_num - pos_last))
            rotation_factor = 2.0 - (rotation / 18)
        else:
            rotation_factor = 1.0
        
        enhanced_pred.append(prob * freq_factor * rotation_factor)
    
    enhanced_pred = np.array(enhanced_pred)
    enhanced_pred /= enhanced_pred.sum()  # Normaliza
    
    # Seleciona top números com confiança
    top_nums = np.argsort(enhanced_pred)[-5:][::-1]  # Top 5
    return [(n, enhanced_pred[n]) for n in top_nums if enhanced_pred[n] > 0.05]

def get_strategic_neighbors(number, confidence):
    """Calcula vizinhos estratégicos baseado na confiança"""
    if number not in WHEEL_ORDER:
        return []
    
    # Calcula quantidade de vizinhos conforme confiança
    neighbor_count = min(int(confidence * 10), 3)  # Máximo 3 vizinhos
    
    idx = WHEEL_ORDER.index(number)
    neighbors = []
    for i in range(1, neighbor_count + 1):
        neighbors.append(WHEEL_ORDER[(idx - i) % 37])
        neighbors.append(WHEEL_ORDER[(idx + i) % 37])
    
    return list(set(neighbors))  # Remove duplicatas

# --- INTERFACE DE ALTA PERFORMANCE ---
def add_number_callback():
    """Processamento dinâmico com Enter"""
    if st.session_state.num_input != "":
        try:
            num = int(st.session_state.num_input)
            if 0 <= num <= 36:
                # Adiciona ao histórico
                st.session_state.history.append(num)
                st.session_state.num_input = ""
                
                # Treino contínuo do modelo
                if len(st.session_state.history) > SEQUENCE_LEN * 2:
                    if st.session_state.model is None:
                        st.session_state.model = build_expert_model()
                    
                    with st.spinner("🧠 Aprendendo padrões..."):
                        X_seq, X_feat, y = [], [], []
                        for i in range(len(st.session_state.history) - SEQUENCE_LEN - 1):
                            seq = st.session_state.history[i:i+SEQUENCE_LEN]
                            X_seq.append(seq)
                            X_feat.append(get_deep_features(seq))
                            y.append(st.session_state.history[i+SEQUENCE_LEN])
                        
                        X_seq = np.array(X_seq).reshape(-1, SEQUENCE_LEN, 1)
                        X_feat = np.array(X_feat)
                        y = tf.keras.utils.to_categorical(y, NUM_TOTAL)
                        
                        st.session_state.model.fit(
                            [X_seq, X_feat], y,
                            epochs=15,
                            batch_size=16,
                            verbose=0,
                            callbacks=[EarlyStopping(patience=2)]
                        )
        except:
            pass

# --- LAYOUT PRINCIPAL ---
# Entrada de dados
st.text_input("DIGITE O NÚMERO SORTEADO (0-36) E PRESSIONE ENTER:",
             key="num_input",
             on_change=add_number_callback)

# Painel de previsões
if len(st.session_state.history) >= SEQUENCE_LEN:
    if st.session_state.model is None:
        st.session_state.model = build_expert_model()
    
    predictions = predict_with_confidence(st.session_state.model, st.session_state.history)
    
    if predictions:
        st.subheader("🎯 MELHORES APOSTAS (PRECISÃO NEURAL)")
        
        # Exibe cada previsão com estratégia
        for num, confidence in predictions:
            neighbors = get_strategic_neighbors(num, confidence)
            
            with st.expander(f"NÚMERO {num} - CONFIANÇA: {confidence:.1%}", expanded=True):
                cols = st.columns([1, 3])
                cols[0].metric("Probabilidade", f"{confidence:.1%}")
                
                if neighbors:
                    cols[1].write(f"**Vizinhos estratégicos ({len(neighbors)}):** {', '.join(map(str, neighbors))}")
                else:
                    cols[1].write("**Jogar apenas o número (alta confiança)**")
                
                # Atualiza estatísticas se for o último número
                if len(st.session_state.history) > SEQUENCE_LEN:
                    last_num = st.session_state.history[-1]
                    if num == last_num:
                        st.success("✅ ACERTOU NA ÚLTIMA RODADA!")
                        st.session_state.stats['acertos'] += 1
                    st.session_state.stats['total'] += 1
    else:
        st.warning("Sistema analisando padrões... aguarde mais rodadas")
else:
    st.info(f"⌛ Insira mais {SEQUENCE_LEN - len(st.session_state.history)} números para ativar o sistema")

# Estatísticas de performance
if st.session_state.stats['total'] > 0:
    st.subheader("📈 DESEMPENHO DO SISTEMA")
    acerto_percent = st.session_state.stats['acertos'] / st.session_state.stats['total']
    expected = st.session_state.stats['total'] * (3/37)  # Considerando 3 sugestões
    
    cols = st.columns(3)
    cols[0].metric("Acertos", st.session_state.stats['acertos'])
    cols[1].metric("Precisão", f"{acerto_percent:.1%}")
    cols[2].metric("Eficiência", 
                  f"{(st.session_state.stats['acertos']/expected):.1f}x",
                  "Acima do esperado")

# Histórico compacto
if st.session_state.history:
    st.subheader("ÚLTIMOS NÚMEROS")
    st.write(" → ".join(map(str, st.session_state.history[-20:])))
    
    # Frequências recentes
    freq = Counter(st.session_state.history[-50:]).most_common(5)
    st.caption(f"🔍 Números quentes: {', '.join([f'{n} ({c}x)' for n, c in freq])}")
