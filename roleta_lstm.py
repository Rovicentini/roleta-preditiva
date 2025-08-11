import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import streamlit as st

# Configurações globais
NUM_TOTAL = 37  # 0-36
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
SEQ_LEN = 20
BATCH_SIZE = 64

# Hiperparâmetros
LEARNING_RATE = 0.001
GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Inicialização do estado
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'memory' not in st.session_state:
    st.session_state.memory = deque(maxlen=2000)
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'balance': 1000,
        'bet_history': [],
        'wins': 0,
        'losses': 0,
        'epsilon': EPSILON
    }

class DQNAgent:
    def __init__(self):
        self.model = self._build_hybrid_model()
        self.target_model = self._build_hybrid_model()
        self.update_target_model()
        
    def _build_hybrid_model(self):
        """Modelo híbrido LSTM + Dense para Q-Learning"""
        # Input para sequência temporal
        seq_input = Input(shape=(SEQ_LEN, 1))
        lstm = LSTM(128, return_sequences=True)(seq_input)
        lstm = LSTM(64)(lstm)
        
        # Input para features adicionais
        feat_input = Input(shape=(6,))
        dense = Dense(64, activation='relu')(feat_input)
        
        # Combinação
        combined = Concatenate()([lstm, dense])
        x = Dense(128, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(NUM_TOTAL, activation='linear')(x)
        
        model = Model(inputs=[seq_input, feat_input], outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def get_features(self, history):
        """Extrai features avançadas do histórico"""
        if len(history) < 2:
            return [0]*6
        
        last = history[-1]
        mean = np.mean(history[-10:]) if len(history) >= 10 else np.mean(history)
        std = np.std(history[-10:]) if len(history) >= 10 else np.std(history)
        
        return [
            mean/36,
            std/18,
            history.count(last)/len(history),
            (max(history) - min(history))/36,
            len(set(history))/len(history),
            1 if last == history[-2] else 0
        ]
    
    def remember(self, state, action, reward, next_state, done):
        st.session_state.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= st.session_state.stats['epsilon']:
            return random.randrange(NUM_TOTAL)
        seq = np.array(state[0]).reshape(1, SEQ_LEN, 1)
        feat = np.array([state[1]])
        act_values = self.model.predict([seq, feat], verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        if len(st.session_state.memory) < batch_size:
            return
        
        minibatch = random.sample(st.session_state.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_seq = np.array(next_state[0]).reshape(1, SEQ_LEN, 1)
                next_feat = np.array([next_state[1]])
                target = reward + GAMMA * np.amax(self.target_model.predict([next_seq, next_feat], verbose=0)[0])
            
            seq = np.array(state[0]).reshape(1, SEQ_LEN, 1)
            feat = np.array([state[1]])
            target_f = self.model.predict([seq, feat], verbose=0)
            target_f[0][action] = target
            self.model.fit([seq, feat], target_f, epochs=1, verbose=0)

def clear_input():
    st.session_state.last_input = st.session_state.current_number
    st.session_state.current_number = ""

# Interface Streamlit
st.set_page_config(layout="wide")
st.title("🎰 Roleta AI Avançada - DQN")

# Controles
with st.sidebar:
    st.header("Controle")
    if st.button("Reiniciar Sistema"):
        st.session_state.history = []
        st.session_state.model = None
        st.session_state.memory = deque(maxlen=2000)
        st.session_state.stats = {
            'balance': 1000,
            'bet_history': [],
            'wins': 0,
            'losses': 0,
            'epsilon': EPSILON
        }

# Entrada de dados
with st.form("input_form", clear_on_submit=True):
    num = st.number_input("Número sorteado (0-36):", 
                         min_value=0, 
                         max_value=36, 
                         key="current_number",
                         on_change=clear_input)
    
    if st.form_submit_button("Registrar"):
        if 'last_input' in st.session_state and st.session_state.last_input is not None:
            num = int(st.session_state.last_input)
            st.session_state.history.append(num)
            
            # Inicializar agente se necessário
            if st.session_state.model is None:
                st.session_state.model = DQNAgent()
            
            # Processar aprendizado
            if len(st.session_state.history) > SEQ_LEN:
                agent = st.session_state.model
                state = (
                    st.session_state.history[-SEQ_LEN-1:-1],
                    agent.get_features(st.session_state.history[-SEQ_LEN-1:-1])
                )
                next_state = (
                    st.session_state.history[-SEQ_LEN:],
                    agent.get_features(st.session_state.history[-SEQ_LEN:])
                )
                
                # Calcular recompensa
                reward = 35 if num == np.random.choice(top_indices) else -1  # Simulação
                done = False
                
                agent.remember(state, num, reward, next_state, done)
                agent.replay(BATCH_SIZE)
                
                # Atualizar estatísticas
                if reward > 0:
                    st.session_state.stats['wins'] += 1
                    st.session_state.stats['balance'] += reward
                else:
                    st.session_state.stats['losses'] += 1
                    st.session_state.stats['balance'] += reward
                
                # Decaimento do epsilon
                st.session_state.stats['epsilon'] = max(EPSILON_MIN, st.session_state.stats['epsilon'] * EPSILON_DECAY)

# Visualização
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Previsões e Estratégia")
    if len(st.session_state.history) >= SEQ_LEN and st.session_state.model:
        agent = st.session_state.model
        current_state = (
            st.session_state.history[-SEQ_LEN:],
            agent.get_features(st.session_state.history[-SEQ_LEN:])
        )
        
        # Obter Q-values
        seq = np.array(current_state[0]).reshape(1, SEQ_LEN, 1)
        feat = np.array([current_state[1]])
        q_values = agent.model.predict([seq, feat], verbose=0)[0]
        
        # Top 5 números recomendados
        top_indices = np.argsort(q_values)[-5:][::-1]
        
        st.write("### Melhores Apostas (Q-Values)")
        for i, num in enumerate(top_indices):
            st.metric(f"Número {num}", f"Q-Value: {q_values[num]:.2f}")
            
            # Sugestão de vizinhos
            neighbors = []
            if num in WHEEL_ORDER:
                idx = WHEEL_ORDER.index(num)
                neighbors = [WHEEL_ORDER[(idx-1)%37], WHEEL_ORDER[(idx+1)%37]]
            
            st.write(f"Vizinhos estratégicos: {neighbors}")

with col2:
    st.subheader("Status do Sistema")
    st.metric("Saldo", f"R$ {st.session_state.stats['balance']:.2f}")
    st.metric("Vitórias/Perdas", f"{st.session_state.stats['wins']}/{st.session_state.stats['losses']}")
    st.metric("Exploração (Epsilon)", f"{st.session_state.stats['epsilon']:.2f}")
    
    st.write("### Histórico Recente")
    if st.session_state.history:
        st.write(st.session_state.history[-10:])
    
    st.write("### Hiperparâmetros")
    st.json({
        "Taxa de Aprendizado": LEARNING_RATE,
        "Fator Gamma": GAMMA,
        "Epsilon Mínimo": EPSILON_MIN,
        "Decaimento Epsilon": EPSILON_DECAY
    })

# Otimização contínua
if st.session_state.model and len(st.session_state.history) > 100:
    agent = st.session_state.model
    agent.replay(BATCH_SIZE)
    agent.update_target_model()
