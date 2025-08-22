import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
import pandas as pd # Adicionado para visualização da tabela

from collections import Counter, deque
import random
import logging
import json

# =========================
# Utils Streamlit
# =========================
def rerun():
    # Mantido por compatibilidade (não usado diretamente)
    raise st.script_runner.RerunException(st.script_request_queue.RerunData())

logging.basicConfig(filename='roleta.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()

# --- SESSION STATE INIT ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'dqn_agent' not in st.session_state:
    st.session_state.dqn_agent = None
if 'stats' not in st.session_state:
    st.session_state.stats = {'wins': 0, 'bets': 0, 'streak': 0, 'max_streak': 0, 'profit': 0.0}
if 'last_input' not in st.session_state:
    st.session_state.last_input = None
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'prev_state' not in st.session_state:
    st.session_state.prev_state = None
if 'prev_action' not in st.session_state:
    st.session_state.prev_action = None
if 'prev_actions' not in st.session_state:
    st.session_state.prev_actions = None
if 'input_bulk' not in st.session_state:
    st.session_state.input_bulk = ""
if 'clear_input_bulk' not in st.session_state:
    st.session_state.clear_input_bulk = False
# =========================================================================
# NOVOS ESTADOS PARA A TABELA DE NÚMEROS QUE SE PUXAM (CO-OCORRÊNCIA)
# =========================================================================
if 'pull_table' not in st.session_state:
    # A tabela de co-ocorrência: pull_table[num_anterior][num_atual] = contagem
    st.session_state.pull_table = {str(i): {str(j): 0 for j in range(37)} for i in range(37)}

# --- CONSTANTS ---
NUM_TOTAL = 37
SEQUENCE_LEN = 20
BET_AMOUNT = 1.0

# Replay/treino DQN
TARGET_UPDATE_FREQ = 50
REPLAY_BATCH = 100
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95

# Recompensa (shaping)
REWARD_EXACT = 35.0
REWARD_NEIGHBOR = 5.0
REWARD_COLOR = 0.0
REWARD_DOZEN = 0.0
REWARD_LOSS = -15.0
NEIGHBOR_RADIUS_FOR_REWARD = 3

# Treino LSTM incremental
LSTM_RECENT_WINDOWS = 400
LSTM_BATCH_SAMPLES = 128
LSTM_EPOCHS_PER_STEP = 2
LSTM_BATCH_SIZE = 32

# Hiperparâmetros DQN (exploração)
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.992

# wheel order (posição física na roda)
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,
               5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
WHEEL_DISTANCE = [[min(abs(i-j), 37-abs(i-j)) for j in range(37)] for i in range(37)]

# red numbers for European wheel (standard)
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

# Definindo as principais regiões da roleta
# Baseado na ordem física da roda europeia
REGIONS = {
    "zero_spiel": {0, 32, 15, 19, 4, 21, 2, 25}, # 8 números
    "voisins_du_zero": {0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35, 36}, # 17 números (Vizinhos)
    "tiers_du_cylindre": {27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33}, # 12 números (Terço)
    "orphelins": {1, 6, 9, 14, 17, 20, 31, 34}, # 8 números (Órfãos)
    "juego_del_cero": {12, 15, 32, 19, 26, 3, 35, 0},
    "petite_serie": {5, 8, 10, 11, 13, 16, 23, 24, 27, 30, 33, 36}
}

# Mapeia cada número para uma (ou mais) região
NUMBER_TO_REGION = {n: [] for n in range(NUM_TOTAL)}
for region_name, numbers in REGIONS.items():
    for num in numbers:
        NUMBER_TO_REGION[num].append(region_name)

# --- AUX FUNCTIONS ---
def number_to_color(n):
    if n == 0:
        return 0 # zero
    return 1 if n in RED_NUMBERS else 2 # 1=red,2=black

def number_to_dozen(n):
    if n == 0:
        return 0 # zero
    if 1 <= n <= 12:
        return 1
    if 13 <= n <= 24:
        return 2
    return 3

def number_to_region(n):
    """Retorna o índice da primeira região que o número pertence.
    Retorna -1 se não pertencer a nenhuma região definida."""
    for i, region_name in enumerate(REGIONS):
        if n in REGIONS[region_name]:
            return i
    return -1

def get_advanced_features(sequence):
    if sequence is None or len(sequence) < 2:
        return [0.0]*8
    seq = np.array(sequence)
    mean = np.mean(seq)
    std = np.std(seq)
    last = int(sequence[-1])
    second_last = int(sequence[-2])
    if last in WHEEL_ORDER and second_last in WHEEL_ORDER:
        last_pos = WHEEL_ORDER.index(last)
        second_last_pos = WHEEL_ORDER.index(second_last)
        wheel_speed = (last_pos - second_last_pos) % 37
        if len(sequence) > 2 and sequence[-3] in WHEEL_ORDER:
            third_pos = WHEEL_ORDER.index(sequence[-3])
            prev_speed = (second_last_pos - third_pos) % 37
            deceleration = abs(wheel_speed - prev_speed)
        else:
            deceleration = 0.0
    else:
        wheel_speed = 0.0
        deceleration = 0.0
    freq = Counter(sequence)
    hot_number = max(freq.values()) if freq else 1
    cold_number = min(freq.values()) if freq else 0
    return [
        mean / 36.0,
        std / 18.0,
        wheel_speed / 36.0,
        deceleration / 36.0,
        freq.get(last, 0) / hot_number if hot_number > 0 else 0.0,
        (hot_number - cold_number) / len(sequence) if len(sequence) > 0 else 0.0,
        len(freq) / len(sequence) if len(sequence) > 0 else 0.0,
        1.0 if last == second_last else 0.0
    ]

def sequence_to_one_hot(sequence):
    """
    Retorna array (SEQUENCE_LEN, NUM_TOTAL) com one-hot da posição na roda.
    Padding (quando falta histórico) -> vetor zeros.
    """
    seq = list(sequence[-SEQUENCE_LEN:]) if sequence else []
    pad = [-1] * max(0, (SEQUENCE_LEN - len(seq)))
    seq_padded = pad + seq
    one_hot_seq = []
    for x in seq_padded:
        if x in WHEEL_ORDER:
            pos = WHEEL_ORDER.index(x)
            one_hot_seq.append(to_categorical(pos, NUM_TOTAL))
        else:
            one_hot_seq.append(np.zeros(NUM_TOTAL))
    return np.array(one_hot_seq)

def sequence_to_state(sequence, model=None):
    """
    Retorna estado 1D para o DQN: [one_hot_seq.flatten(), features, num_probs, color_probs, dozen_probs, age_vector, new_features]
    """
    seq_slice = sequence[-SEQUENCE_LEN:] if sequence else []

    # 1. Base State Features (mantidas)
    one_hot_seq = sequence_to_one_hot(seq_slice)
    features = get_advanced_features(seq_slice)
    
    # 2. LSTM Prediction Probabilities (mantidas)
    num_probs = np.zeros(NUM_TOTAL)
    color_probs = np.zeros(3)
    dozen_probs = np.zeros(4)
    if model is not None and len(sequence) >= SEQUENCE_LEN:
        try:
            seq_arr = np.expand_dims(one_hot_seq, axis=0)
            feat_arr = np.array([features])
            raw = model.predict([seq_arr, feat_arr], verbose=0)
            if isinstance(raw, list) and len(raw) == 3:
                num_probs = np.array(raw[0][0])
                color_probs = np.array(raw[1][0])
                dozen_probs = np.array(raw[2][0])
                
                # Adicionando verificação de forma
                if num_probs.shape != (NUM_TOTAL,):
                    num_probs = np.zeros(NUM_TOTAL)
                if color_probs.shape != (3,):
                    color_probs = np.zeros(3)
                if dozen_probs.shape != (4,):
                    dozen_probs = np.zeros(4)
        except Exception as e:
            # Em caso de erro, os vetores de probabilidade permanecem zeros
            pass

    # 3. Vetor de Idade (mantido e normalizado)
    age_vector = [0] * NUM_TOTAL
    last_seen = {num: i for i, num in enumerate(sequence)}
    for num in range(NUM_TOTAL):
        if num in last_seen:
            age_vector[num] = len(sequence) - 1 - last_seen[num]
        else:
            age_vector[num] = len(sequence)
    max_age = max(age_vector) if age_vector else 1
    age_vector = [age / max(1, max_age) for age in age_vector]
    age_vector = np.array(age_vector)

    # ==================================
    # === NOVAS FEATURES ADICIONADAS ===
    # ==================================
    
    # Nova Feature 1: Run Length Encoding (Streaks e Alternâncias)
    # Apenas para o último número, cor e dúzia
    last_run_len_num = 0
    if len(sequence) >= 2:
        for i in range(1, len(sequence)):
            if sequence[-i] == sequence[-1]:
                last_run_len_num = i
            else:
                break
    last_run_len_color = 0
    if len(sequence) >= 2:
        last_color = number_to_color(sequence[-1])
        for i in range(1, len(sequence)):
            if number_to_color(sequence[-i]) == last_color:
                last_run_len_color = i
            else:
                break
    last_run_len_dozen = 0
    if len(sequence) >= 2:
        last_dozen = number_to_dozen(sequence[-1])
        for i in range(1, len(sequence)):
            if number_to_dozen(sequence[-i]) == last_dozen:
                last_run_len_dozen = i
            else:
                break
    run_length_features = np.array([last_run_len_num / SEQUENCE_LEN, last_run_len_color / SEQUENCE_LEN, last_run_len_dozen / SEQUENCE_LEN])
    
    # Nova Feature 2: Grupos do Último Número (One-hot para Cor e Dúzia)
    last_num = sequence[-1] if sequence else -1
    last_color_one_hot = to_categorical(number_to_color(last_num), 3) if last_num in range(NUM_TOTAL) else np.zeros(3)
    last_dozen_one_hot = to_categorical(number_to_dozen(last_num), 4) if last_num in range(NUM_TOTAL) else np.zeros(4)
    
    # Nova Feature 3: Proporções de Par/Ímpar e Alto/Baixo
    recent_seq = seq_slice
    even_count = sum(1 for n in recent_seq if n % 2 == 0 and n != 0)
    odd_count = sum(1 for n in recent_seq if n % 2 != 0)
    high_count = sum(1 for n in recent_seq if n >= 19 and n <= 36)
    low_count = sum(1 for n in recent_seq if n >= 1 and n <= 18)
    total_non_zero = even_count + odd_count
    
    even_odd_ratio = even_count / max(1, total_non_zero)
    high_low_ratio = high_count / max(1, high_count + low_count)
    
    group_ratio_features = np.array([even_odd_ratio, high_low_ratio])
    
    # NOVAS FEATURES DE REGIÕES DA RODA
    num_regions = len(REGIONS)
    last_region_one_hot = np.zeros(num_regions)
    region_proportions = np.zeros(num_regions)
    region_streak = 0
    
    # Calcula proporções de regiões e streak
    if len(recent_seq) > 0:
        region_counts = Counter(number_to_region(n) for n in recent_seq)
        for i in range(num_regions):
            region_proportions[i] = region_counts.get(i, 0) / len(recent_seq)

        last_region = number_to_region(recent_seq[-1])
        if last_region != -1:
            last_region_one_hot[last_region] = 1
            for i in range(1, len(recent_seq) + 1):
                if number_to_region(recent_seq[-i]) == last_region:
                    region_streak += 1
                else:
                    break
    
    region_streak_norm = region_streak / SEQUENCE_LEN
    
    # Combine todos os vetores para formar o estado final
    state = np.concatenate([
        one_hot_seq.flatten(),
        np.array(features),
        num_probs,
        color_probs,
        dozen_probs,
        age_vector,
        run_length_features,
        last_color_one_hot,
        last_dozen_one_hot,
        group_ratio_features,
        last_region_one_hot,
        region_proportions,
        np.array([region_streak_norm])
    ]).astype(np.float32)

    return state

# =========================
# MODELO LSTM – ARQUITETURA REFINADA
# =========================
def build_deep_learning_model(seq_len=SEQUENCE_LEN, num_total=NUM_TOTAL):
    """
    LSTM multi-output com maior profundidade + atenção:
      - saída 1: probabilidade para cada número (37)
      - saída 2: probabilidade para cor (3) -> zero/red/black
      - saída 3: probabilidade para dúzia (4) -> zero/d1/d2/d3
    """
    seq_input = Input(shape=(seq_len, num_total), name='sequence_input')

    # Pilha LSTM mais profunda
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(seq_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = LSTM(96, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Self-Attention simples
    x_att = Attention(name="self_attention")([x, x])

    # Mais uma LSTM para sintetizar após atenção
    x = LSTM(64, return_sequences=False)(x_att)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)

    # Features adicionais
    feat_input = Input(shape=(8,), name='features_input')
    dense_feat = Dense(48, activation='swish')(feat_input)
    dense_feat = BatchNormalization()(dense_feat)
    dense_feat = Dropout(0.2)(dense_feat)

    # Combina LSTM + features
    combined = Concatenate()([x, dense_feat])
    dense = Dense(160, activation='swish')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)

    out_num = Dense(num_total, activation='softmax', name='num_out')(dense)
    out_color = Dense(3, activation='softmax', name='color_out')(dense) # zero/red/black
    out_dozen = Dense(4, activation='softmax', name='dozen_out')(dense) # zero,d1,d2,d3

    model = Model(inputs=[seq_input, feat_input], outputs=[out_num, out_color, out_dozen])
    optimizer = Nadam(learning_rate=4e-4)
    model.compile(optimizer=optimizer,
                  loss={'num_out': 'categorical_crossentropy',
                        'color_out': 'categorical_crossentropy',
                        'dozen_out': 'categorical_crossentropy'},
                  loss_weights={'num_out': 1.0, 'color_out': 0.35, 'dozen_out': 0.35},
                  metrics={'num_out': 'accuracy'})
    return model

# =========================
# DQN Agent
# =========================
class DQNAgent:
    def __init__(self, state_size, action_size, lr=DQN_LEARNING_RATE, gamma=DQN_GAMMA, replay_size=REPLAY_SIZE):
        self.state_size = int(state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=replay_size)
        self.gamma = gamma
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.learning_rate = lr
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()
        self.train_step = 0

    def _build_model(self):
        model = tf.keras.Sequential([
            Dense(320, activation='relu', input_shape=(self.state_size,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(160, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target(self):
        try:
            self.target_model.set_weights(self.model.get_weights())
        except Exception:
            pass

    def remember(self, state, action, reward, next_state, done):
        if state is None or next_state is None:
            return
        self.memory.append((state, action, reward, next_state, done))

    def act_top_k(self, state, k=3, use_epsilon=True):
        if state is None or len(state) == 0:
            return random.sample(range(self.action_size), k)
        # Exploração
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), k)
        try:
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            top_k_actions = np.argsort(q_values)[-k:][::-1]
            return top_k_actions.tolist()
        except Exception:
            return random.sample(range(self.action_size), k)

    def act(self, state, use_epsilon=True):
        if state is None or len(state) == 0:
            return random.randrange(self.action_size)
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        try:
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            return int(np.argmax(q_values))
        except Exception:
            return random.randrange(self.action_size)

    def replay(self, batch_size=REPLAY_BATCH):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Preparar dados para o modelo (o estado é o mesmo para todas as ações)
        states = np.array([b[0] for b in batch])
        next_states = np.array([b[3] for b in batch])
        
        if states.size == 0 or next_states.size == 0:
            return
        
        try:
            # Previsões para o próximo estado (Q-learning)
            q_next = self.target_model.predict(next_states, verbose=0)
            # Previsões para o estado atual
            q_curr = self.model.predict(states, verbose=0)
        except Exception:
            return

        # Construir os lotes de treinamento
        X, Y = [], []
        for i, (state, actions, reward, next_state, done) in enumerate(batch):
            target = q_curr[i].copy()
            
            # O loop principal para ajustar o Q-value de cada ação do lote
            for action in actions:
                if done:
                    # Se o jogo terminou, a recompensa é o valor final
                    target[action] = reward
                else:
                    # Caso contrário, aplica a equação de Bellman
                    next_q = q_next[i] if i < len(q_next) else np.zeros(self.action_size)
                    target[action] = reward + self.gamma * np.max(next_q)
            
            X.append(state)
            Y.append(target)
        
        try:
            self.model.fit(np.array(X), np.array(Y), epochs=1, verbose=0)
        except Exception as e:
            logger.error(f"Erro no treinamento do DQN: {e}")
            pass
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_weights(path)
        self.update_target()

    def save(self, path):
        self.model.save_weights(path)

# --- Neighbors ---
def optimal_neighbors(number, max_neighbors=2):
    """
    Retorna lista de vizinhos (à esquerda e direita alternados) da roda.
    """
    if number not in WHEEL_ORDER:
        return []
    idx = WHEEL_ORDER.index(number)
    neigh = []
    for i in range(1, max_neighbors + 1):
        neigh.append(WHEEL_ORDER[(idx - i) % NUM_TOTAL])
        neigh.append(WHEEL_ORDER[(idx + i) % NUM_TOTAL])
    return list(dict.fromkeys(neigh))
    
# =========================================================================
# NOVAS FUNÇÕES: TABELA DE NÚMEROS QUE SE PUXAM E VIZINHOS DINÂMICOS
# =========================================================================
def update_pull_table(history):
    """
    Atualiza a tabela de co-ocorrência a cada novo número.
    """
    if len(history) < 2:
        return
    
    prev_num = str(history[-2])
    current_num = str(history[-1])
    
    if prev_num in st.session_state.pull_table and current_num in st.session_state.pull_table[prev_num]:
        st.session_state.pull_table[prev_num][current_num] += 1
        
def get_pull_weights(last_number, pull_table, alpha=0.5):
    """
    Calcula os pesos de 'pull' baseados na tabela de co-ocorrência.
    """
    if str(last_number) not in pull_table:
        return np.ones(NUM_TOTAL)
    
    counts = pull_table[str(last_number)]
    total_count = sum(counts.values())
    
    if total_count == 0:
        return np.ones(NUM_TOTAL)
        
    weights = np.array([counts.get(str(i), 0) / total_count for i in range(NUM_TOTAL)])
    
    # Aplica um suavizador (smoothing) para evitar zero weights
    weights = weights + alpha
    
    return weights / weights.sum()

def get_smart_neighbor_suggestions(top_k_predictions, pull_table, n_neighbors=3):
    """
    Sugere vizinhos dinâmicos com base nos números que mais se 'puxam'.
    """
    suggestions = set(top_k_predictions)
    
    if len(st.session_state.history) > 0:
        last_num = st.session_state.history[-1]
        
        # 1. Sugere os vizinhos físicos mais próximos do último número
        suggestions.update(optimal_neighbors(last_num, max_neighbors=1))
        
        # 2. Sugere os vizinhos da tabela de pull dos top K
        for num in top_k_predictions:
            if str(num) in pull_table:
                counts = pull_table[str(num)]
                # Seleciona os N vizinhos com maior contagem
                top_neighbors_pull = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:n_neighbors]
                suggestions.update([int(n) for n, count in top_neighbors_pull])
    
    return sorted(list(suggestions))

# =========================
# RECOMPENSA FOCADA E SIMPLIFICADA
# =========================
def compute_reward(action_numbers, outcome_number, bet_amount=BET_AMOUNT,
                   max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD):
    """
    Recompensa focada: apenas premia acerto de número ou vizinho.
    Perda em todos os outros casos.
    """
    reward = 0.0
    action_numbers = set([a for a in action_numbers if 0 <= a <= 36])

    # 1. Acerto Exato (maior recompensa e prioridade)
    if outcome_number in action_numbers:
        reward = REWARD_EXACT
    # 2. Acerto Vizinho (recompensa menor)
    else:
        all_neighbors = set()
        for a in action_numbers:
            all_neighbors.update(optimal_neighbors(a, max_neighbors=max_neighbors_for_reward))
        
        if outcome_number in all_neighbors:
            reward = REWARD_NEIGHBOR
        # 3. Perda total (penalidade)
        else:
            reward = REWARD_LOSS

    return reward * bet_amount

# --- PREDICTION POSTPROCESSING ---
def predict_next_numbers(model, history, pull_table, top_k=3):
    if history is None or len(history) < SEQUENCE_LEN or model is None:
        return []
    try:
        seq_one_hot = sequence_to_one_hot(history).reshape(1, SEQUENCE_LEN, NUM_TOTAL)
        feat = np.array([get_advanced_features(history[-SEQUENCE_LEN:])])
        raw = model.predict([seq_one_hot, feat], verbose=0)
        # raw -> [num_probs, color_probs, dozen_probs]
        if isinstance(raw, list) and len(raw) == 3:
            num_probs = raw[0][0]
            color_probs = raw[1][0]
            dozen_probs = raw[2][0]
        else:
            num_probs = np.array(raw)[0]
            color_probs = np.array([0.0, 0.0, 0.0])
            dozen_probs = np.array([0.0, 0.0, 0.0, 0.0])
    except Exception:
        return []

    # temperature + heurísticas
    temperature = 0.8
    adjusted = np.log(num_probs + 1e-12) / temperature
    adjusted = np.exp(adjusted)
    adjusted /= adjusted.sum()

    weighted = []
    freq_counter = Counter(history[-100:])
    last_num = history[-1] if len(history) > 0 else None
    
    # NOVOS PESOS DA TABELA DE PUXAR
    pull_weights = get_pull_weights(last_num, pull_table, alpha=1.0)
    
    for num in range(NUM_TOTAL):
        freq_factor = 1 + np.exp(freq_counter.get(num, 0) / 3 - 1)
        if last_num in WHEEL_ORDER:
            dist = WHEEL_DISTANCE[WHEEL_ORDER.index(last_num)][WHEEL_ORDER.index(num)]
            distance_factor = max(0.1, 2.5 - (dist / 12.0))
        else:
            distance_factor = 1.0
        momentum = sum(1 for i in range(1,4) if len(history)>=i and history[-i] == num)
        momentum_factor = 1 + momentum*0.25
        
        # Ponderação combinada com o novo fator de "pull"
        weighted.append(adjusted[num] * freq_factor * distance_factor * momentum_factor * pull_weights[num])
        
    weighted = np.array(weighted)
    if weighted.sum() == 0:
        return []
    weighted /= weighted.sum()

    top_indices = list(np.argsort(weighted)[-top_k:][::-1])
    color_pred = int(np.argmax(color_probs))
    dozen_pred = int(np.argmax(dozen_probs))
    return {
        'top_numbers': [(int(i), float(weighted[i])) for i in top_indices],
        'num_probs': num_probs,
        'color_probs': color_probs,
        'dozen_pred': dozen_pred,
        'dozen_probs': dozen_probs,
        'color_pred': color_pred
    }

# =========================
# LSTM: construção de dataset e treino recente
# =========================
def build_lstm_supervised_from_history(history):
    """
    Constrói X_seq, X_feat, y_num, y_color, y_dozen a partir do histórico.
    Retorna arrays numpy prontos para treino.
    """
    X_seq, X_feat, y_num, y_color, y_dozen = [], [], [], [], []
    if len(history) <= SEQUENCE_LEN:
        return None

    start_idx = max(0, len(history) - (SEQUENCE_LEN + 1) - LSTM_RECENT_WINDOWS)
    for i in range(start_idx, len(history) - SEQUENCE_LEN - 1):
        seq_slice = history[i:i+SEQUENCE_LEN]
        target = history[i+SEQUENCE_LEN]

        X_seq.append(sequence_to_one_hot(seq_slice))
        X_feat.append(get_advanced_features(seq_slice))

        if target in WHEEL_ORDER:
            pos = WHEEL_ORDER.index(target)
            y_num.append(to_categorical(pos, NUM_TOTAL))
        else:
            y_num.append(np.zeros(NUM_TOTAL))

        color_label = number_to_color(target)
        y_color.append(to_categorical(color_label, 3))

        dozen_label = number_to_dozen(target)
        y_dozen.append(to_categorical(dozen_label, 4))

    if len(X_seq) == 0:
        return None

    X_seq = np.array(X_seq)
    X_feat = np.array(X_feat)
    y_num = np.array(y_num)
    y_color = np.array(y_color)
    y_dozen = np.array(y_dozen)
    return X_seq, X_feat, y_num, y_color, y_dozen

def train_lstm_on_recent_minibatch(model, history):
    """
    Treina o LSTM usando amostras aleatórias de janelas recentes,
    evitando reprocessar todo histórico em cada passo.
    """
    data = build_lstm_supervised_from_history(history)
    if data is None:
        return
    X_seq, X_feat, y_num, y_color, y_dozen = data
    n = len(X_seq)
    if n == 0:
        return

    # Amostragem aleatória sem reposição
    k = min(n, LSTM_BATCH_SAMPLES)
    idx = np.random.choice(n, k, replace=False)
    try:
        model.fit([X_seq[idx], X_feat[idx]],
                  [y_num[idx], y_color[idx], y_dozen[idx]],
                  epochs=LSTM_EPOCHS_PER_STEP,
                  batch_size=LSTM_BATCH_SIZE,
                  verbose=0)
        logger.info(f"LSTM mini-train: {k} amostras de {n} ({LSTM_EPOCHS_PER_STEP} épocas).")
    except Exception as e:
        logger.error(f"Erro no treinamento LSTM: {e}")

# --- UI ---
st.set_page_config(layout="centered")
st.title("🔥 ROULETTE AI - LSTM multi-saída + DQN (REVISADO + REWARD / TREINO RECENTE)")

st.markdown("### Inserir histórico manualmente (ex: 0,32,15,19,4,21)")

# 1) Garantir chaves no session_state
if 'input_bulk' not in st.session_state:
    st.session_state.input_bulk = ""
if 'clear_input_bulk' not in st.session_state:
    st.session_state.clear_input_bulk = False

# 2) APLICAR LIMPEZA ANTES DE CRIAR O WIDGET
if st.session_state.clear_input_bulk:
    st.session_state.input_bulk = ""
    st.session_state.clear_input_bulk = False

# 3) Criar o text_area
input_bulk = st.text_area("Cole números separados por vírgula", key="input_bulk")

# 4) Botão para adicionar histórico
if st.button("Adicionar histórico"):
    if st.session_state.input_bulk and st.session_state.input_bulk.strip():
        try:
            new_nums = [
                int(x.strip())
                for x in st.session_state.input_bulk.split(",")
                if x.strip().isdigit() and 0 <= int(x.strip()) <= 36
            ]
            
            # Atualiza a pull_table para os novos números
            if st.session_state.history:
                prev_num = st.session_state.history[-1]
                for num in new_nums:
                    st.session_state.pull_table[str(prev_num)][str(num)] += 1
                    prev_num = num
            
            st.session_state.history.extend(new_nums)
            st.success(f"Adicionados {len(new_nums)} números ao histórico.")
            st.session_state.clear_input_bulk = True
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao processar números: {e}")
    else:
        st.warning("Insira números válidos para adicionar.")

st.markdown("---")

with st.form("num_form", clear_on_submit=True):
    num_input = st.number_input("Digite o último número (0-36):", min_value=0, max_value=36, step=1, key="current_number")
    submitted = st.form_submit_button("Enviar")
    if submitted:
        st.session_state.last_input = int(num_input)

if st.session_state.last_input is not None:
    try:
        num = int(st.session_state.last_input)
        
        # ATUALIZA A TABELA DE PUXAR ANTES DE ADICIONAR O NOVO NÚMERO
        if len(st.session_state.history) > 0:
            update_pull_table(st.session_state.history + [num])
            
        st.session_state.history.append(num)
        logger.info(f"Número novo inserido pelo usuário: {num}")
        st.session_state.last_input = None

        state_example = sequence_to_state(st.session_state.history, st.session_state.model)
        if state_example is not None and (st.session_state.dqn_agent is None):
            st.session_state.dqn_agent = DQNAgent(state_size=state_example.shape[0], action_size=NUM_TOTAL)
            logger.info("Agente DQN criado")

        if st.session_state.prev_state is not None and st.session_state.prev_actions is not None:
            agent = st.session_state.dqn_agent
            reward = compute_reward(st.session_state.prev_actions, num, bet_amount=BET_AMOUNT,
                                    max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD)
            next_state = sequence_to_state(st.session_state.history, st.session_state.model)

            if agent is not None:
                agent.remember(st.session_state.prev_state, st.session_state.prev_actions, reward, next_state, False)
                logger.info(f"Memorizado: ações={st.session_state.prev_actions}, resultado={num}, recompensa={reward}")

            st.session_state.stats['bets'] += 1
            st.session_state.stats['profit'] += reward
            if reward > 0:
                st.session_state.stats['wins'] += 1
                st.session_state.stats['streak'] += 1
                st.session_state.stats['max_streak'] = max(st.session_state.stats['max_streak'], st.session_state.stats['streak'])
            else:
                st.session_state.stats['streak'] = 0

            st.session_state.step_count += 1
            if agent is not None and st.session_state.step_count % DQN_TRAIN_EVERY == 0:
                agent.replay(REPLAY_BATCH)
                logger.info(f"DQN treinado no passo {st.session_state.step_count}")
            if agent is not None and st.session_state.step_count % TARGET_UPDATE_FREQ == 0:
                agent.update_target()
                logger.info(f"Target DQN atualizado no passo {st.session_state.step_count}")

        if st.session_state.model is None and len(st.session_state.history) >= SEQUENCE_LEN*2:
            st.session_state.model = build_deep_learning_model()
            logger.info("Modelo LSTM criado")

        if st.session_state.model is not None and len(st.session_state.history) > SEQUENCE_LEN*2:
            with st.spinner("Treinando LSTM com mini-batches recentes..."):
                train_lstm_on_recent_minibatch(st.session_state.model, st.session_state.history)
            st.session_state.prev_state = sequence_to_state(st.session_state.history, st.session_state.model)
            pred_info = predict_next_numbers(st.session_state.model, st.session_state.history, st.session_state.pull_table, top_k=3)

            if pred_info:
                st.subheader("🎯 Previsões (LSTM + pós-processamento)")
                for n, conf in pred_info['top_numbers']:
                    st.write(f"Número: **{n}** — Prob: {conf:.2f}")

                st.markdown("---")
                st.subheader("Sugestões de Vizinhos Dinâmicos")
                top_k_numbers = [n for n, conf in pred_info['top_numbers']]
                suggestions = get_smart_neighbor_suggestions(top_k_numbers, st.session_state.pull_table, n_neighbors=2)
                if suggestions:
                    st.write(f"Sugeridos: **{', '.join(map(str, suggestions))}**")

                st.markdown("---")
                st.subheader("Estatísticas de Desempenho (DQN)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Total de Apostas", value=st.session_state.stats['bets'])
                with col2:
                    st.metric(label="Vitórias", value=st.session_state.stats['wins'])
                with col3:
                    st.metric(label="Lucro/Prejuízo", value=f"R$ {st.session_state.stats['profit']:.2f}")

                # ================================================
                # LINHAS REMOVIDAS PARA OCULTAR A TABELA DE CO-OCORRÊNCIA
                # O código abaixo foi comentado para não exibir a tabela
                # ================================================
                # st.markdown("---")
                # st.subheader("Tabela de Co-ocorrência ('Números que se Puxam')")
                # pull_df = pd.DataFrame(st.session_state.pull_table).fillna(0).astype(int)
                # st.dataframe(pull_df)
                
                # ================================================

                st.markdown("---")
                st.subheader("Histórico Recente")
                st.write(st.session_state.history[-20:])
        
    except Exception as e:
        st.error(f"Erro ao processar o número: {e}")
        logger.error(f"Erro fatal: {e}")
        st.session_state.last_input = None
        st.rerun()

