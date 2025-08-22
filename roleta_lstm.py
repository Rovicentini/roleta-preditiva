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

from collections import Counter, deque
import random
import logging

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
    st.session_state.stats = {'wins': 0, 'bets': 0, 'profit': 0.0, 'cost': 0.0}
if 'last_input' not in st.session_state:
    st.session_state.last_input = None
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'prev_state' not in st.session_state:
    st.session_state.prev_state = None
if 'prev_action' not in st.session_state:
    st.session_state.prev_action = None
if 'input_bulk' not in st.session_state:
    st.session_state.input_bulk = ""
if 'clear_input_bulk' not in st.session_state:
    st.session_state.clear_input_bulk = False

# --- CONSTANTS ---
NUM_TOTAL = 37
SEQUENCE_LEN = 20
BET_AMOUNT_UNIT = 1.0

# Replay/treino DQN
TARGET_UPDATE_FREQ = 50
REPLAY_BATCH = 100
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95

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

# =========================================================
# === LISTA DE NÚMEROS QUE SE "PUXAM" ===
# =========================================================
PULL_NUMBERS = {
    0: [10, 20, 30],
    1: [17, 7],
    2: [2, 22],
    3: [3, 33],
    4: [21, 9],
    5: [25, 15, 35],
    6: [20, 17, 7],
    7: [7, 17, 20],
    8: [30, 0, 20],
    9: [9, 19],
    10: [0, 20, 30],
    11: [30, 0, 20],
    12: [33, 15],
    13: [20, 7],
    14: [17, 7],
    15: [9, 5, 35],
    16: [3, 33],
    17: [17, 20, 7],
    18: [2, 22],
    19: [19, 9],  
    20: [17, 7],
    21: [2, 22],   
    22: [2, 22],
    23: [0, 10],
    24: [35, 15, 25],
    25: [20, 22],
    26: [0, 10, 30],
    27: [17, 7, 20],
    28: [7, 17, 20],
    29: [7, 17, 20],
    30: [0, 20, 30],
    31: [9, 19],
    32: [0, 10, 20, 30],
    33: [3, 33],
    34: [7, 20],
    35: [3, 33, 15],
    36: [20, 30],

}

# =========================================================
# === NOVAS ESTRATÉGIAS DE APOSTA (AÇÕES PARA O DQN) ===
# =========================================================
# Definimos um dicionário para mapear índices de ação para descrições.
BETTING_STRATEGIES = {
    0: "Apostar nos 3 números mais prováveis",
    1: "Apostar no 1 número mais provável + vizinhos (2 de cada lado)",
    2: "Apostar na região mais provável",
    3: "Apostar nos números 'quentes' (mais frequentes)",
    4: "Apostar nos números 'frios' (menos frequentes)",
    5: "Apostar em ímpar/par mais provável",
    6: "Apostar em alto/baixo mais provável",
    7: "Apostar na cor mais provável",
    8: "Apostar na dúzia mais provável"
}
ACTION_SIZE = len(BETTING_STRATEGIES)

# --- AUX FUNCTIONS ---
def number_to_color(n):
    if n == 0:
        return 0  # zero
    return 1 if n in RED_NUMBERS else 2  # 1=red,2=black

def number_to_dozen(n):
    if n == 0:
        return 0  # zero
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
    seq_slice = sequence[-SEQUENCE_LEN:] if sequence else []
    one_hot_seq = sequence_to_one_hot(seq_slice)
    features = get_advanced_features(seq_slice)
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
        except Exception:
            pass
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
    last_num = sequence[-1] if sequence else -1
    last_color_one_hot = to_categorical(number_to_color(last_num), 3) if last_num in range(NUM_TOTAL) else np.zeros(3)
    last_dozen_one_hot = to_categorical(number_to_dozen(last_num), 4) if last_num in range(NUM_TOTAL) else np.zeros(4)
    recent_seq = seq_slice
    even_count = sum(1 for n in recent_seq if n % 2 == 0 and n != 0)
    odd_count = sum(1 for n in recent_seq if n % 2 != 0)
    high_count = sum(1 for n in recent_seq if n >= 19 and n <= 36)
    low_count = sum(1 for n in recent_seq if n >= 1 and n <= 18)
    total_non_zero = even_count + odd_count
    even_odd_ratio = even_count / max(1, total_non_zero)
    high_low_ratio = high_count / max(1, high_count + low_count)
    group_ratio_features = np.array([even_odd_ratio, high_low_ratio])
    num_regions = len(REGIONS)
    last_region_one_hot = np.zeros(num_regions)
    region_proportions = np.zeros(num_regions)
    region_streak = 0
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
    pull_features = np.zeros(NUM_TOTAL)
    if len(sequence) > 0:
        last_num = sequence[-1]
        pulled_nums = PULL_NUMBERS.get(last_num, [])
        for num in pulled_nums:
            if 0 <= num < NUM_TOTAL:
                pull_features[num] += 1
        for num_key, nums_list in PULL_NUMBERS.items():
            if last_num in nums_list:
                pull_features[num_key] += 1
    pull_features_sum = np.sum(pull_features)
    if pull_features_sum > 0:
        pull_features = pull_features / pull_features_sum
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
        np.array([region_streak_norm]),
        pull_features
    ]).astype(np.float32)
    return state

# =========================
# MODELO LSTM – ARQUITETURA REFINADA
# =========================
def build_deep_learning_model(seq_len=SEQUENCE_LEN, num_total=NUM_TOTAL):
    seq_input = Input(shape=(seq_len, num_total), name='sequence_input')
    x = LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4))(seq_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = LSTM(96, return_sequences=True, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    x_att = Attention(name="self_attention")([x, x])
    x = LSTM(64, return_sequences=False)(x_att)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    feat_input = Input(shape=(8,), name='features_input')
    dense_feat = Dense(48, activation='swish')(feat_input)
    dense_feat = BatchNormalization()(dense_feat)
    dense_feat = Dropout(0.2)(dense_feat)
    combined = Concatenate()([x, dense_feat])
    dense = Dense(160, activation='swish')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    out_num = Dense(num_total, activation='softmax', name='num_out')(dense)
    out_color = Dense(3, activation='softmax', name='color_out')(dense)
    out_dozen = Dense(4, activation='softmax', name='dozen_out')(dense)
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
    def __init__(self, state_size, action_size=ACTION_SIZE, lr=DQN_LEARNING_RATE, gamma=DQN_GAMMA, replay_size=REPLAY_SIZE):
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
        states = np.array([b[0] for b in batch])
        next_states = np.array([b[3] for b in batch])
        if states.size == 0 or next_states.size == 0:
            return
        try:
            q_next = self.target_model.predict(next_states, verbose=0)
            q_curr = self.model.predict(states, verbose=0)
        except Exception:
            return
        X, Y = [], []
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = q_curr[i].copy()
            if done:
                target[action] = reward
            else:
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
    if number not in WHEEL_ORDER:
        return []
    idx = WHEEL_ORDER.index(number)
    neigh = []
    for i in range(1, max_neighbors+1):
        neigh.append(WHEEL_ORDER[(idx - i) % NUM_TOTAL])
        neigh.append(WHEEL_ORDER[(idx + i) % NUM_TOTAL])
    return list(dict.fromkeys(neigh))

def get_region_numbers(region_name):
    return list(REGIONS.get(region_name, []))

def get_hot_numbers(history, k=5):
    if len(history) < 20: return []
    freq = Counter(history)
    return [num for num, _ in freq.most_common(k)]

def get_cold_numbers(history, k=5):
    if len(history) < 20: return []
    freq = Counter(history)
    all_nums = set(range(NUM_TOTAL))
    cold_nums = sorted(list(all_nums - set(freq.keys())), key=lambda x: np.random.rand())
    cold_nums.extend([num for num, _ in freq.most_common()[-k:]])
    return cold_nums[:k]

def get_high_low_numbers(history, probs, k=1):
    if len(probs) < 3: return []
    if np.argmax(probs[1:]) == 0: # Alto
        return list(range(1,19))
    else: # Baixo
        return list(range(19,37))

def get_odd_even_numbers(history, probs, k=1):
    if len(probs) < 3: return []
    if np.argmax(probs[1:]) == 0: # Impar
        return [n for n in range(1, 37) if n % 2 != 0]
    else: # Par
        return [n for n in range(1, 37) if n % 2 == 0]

def get_color_numbers(history, probs, k=1):
    if len(probs) < 3: return []
    if np.argmax(probs[1:]) == 0: # Vermelho
        return list(RED_NUMBERS)
    else: # Preto
        return list(set(range(1, 37)) - RED_NUMBERS)

def get_dozen_numbers(history, probs, k=1):
    if len(probs) < 4: return []
    dozen_pred = np.argmax(probs[1:])
    if dozen_pred == 0:
        return list(range(1, 13))
    elif dozen_pred == 1:
        return list(range(13, 25))
    else:
        return list(range(25, 37))

# =========================================================
# === PLANOS DE APOSTA (NOVO) ===
# =========================================================
def plan_bet(strategy_id, pred_info, history):
    """
    Função que traduz a estratégia do DQN em um plano de aposta real.
    Retorna uma lista de tuplas (número, tipo_aposta, payout_multiplier, custo).
    """
    if not pred_info: return []
    bet_plan = []
    
    # Mapeamento para simplificar o plano
    num_probs = pred_info.get('num_probs', np.zeros(NUM_TOTAL))
    color_probs = pred_info.get('color_probs', np.zeros(3))
    dozen_probs = pred_info.get('dozen_probs', np.zeros(4))
    
    if strategy_id == 0: # 3 mais prováveis
        top_3 = [t[0] for t in pred_info['top_numbers']]
        for num in top_3:
            bet_plan.append({'number': num, 'type': 'single', 'payout': 35, 'cost': 1})
    elif strategy_id == 1: # 1 mais provável + vizinhos
        top_1 = pred_info['top_numbers'][0][0]
        neighbors = optimal_neighbors(top_1, max_neighbors=2)
        bet_numbers = [top_1] + neighbors
        for num in bet_numbers:
            bet_plan.append({'number': num, 'type': 'single', 'payout': 35, 'cost': 1})
    elif strategy_id == 2: # Região mais provável
        if len(history) < 20: return []
        recent_seq = history[-20:]
        region_counts = Counter(number_to_region(n) for n in recent_seq)
        most_likely_region_idx = -1
        if region_counts:
            most_likely_region_idx = max(region_counts, key=region_counts.get)
        if most_likely_region_idx != -1:
            region_name = list(REGIONS.keys())[most_likely_region_idx]
            bet_numbers = get_region_numbers(region_name)
            for num in bet_numbers:
                bet_plan.append({'number': num, 'type': 'single', 'payout': 35, 'cost': 1})
    elif strategy_id == 3: # Números quentes
        hot_nums = get_hot_numbers(history, k=5)
        for num in hot_nums:
            bet_plan.append({'number': num, 'type': 'single', 'payout': 35, 'cost': 1})
    elif strategy_id == 4: # Números frios
        cold_nums = get_cold_numbers(history, k=5)
        for num in cold_nums:
            bet_plan.append({'number': num, 'type': 'single', 'payout': 35, 'cost': 1})
    elif strategy_id == 5: # Par/Impar
        bet_numbers = get_odd_even_numbers(history, color_probs)
        for num in bet_numbers:
            bet_plan.append({'number': num, 'type': 'parity', 'payout': 1, 'cost': 1})
    elif strategy_id == 6: # Alto/Baixo
        bet_numbers = get_high_low_numbers(history, dozen_probs)
        for num in bet_numbers:
            bet_plan.append({'number': num, 'type': 'high_low', 'payout': 1, 'cost': 1})
    elif strategy_id == 7: # Cor
        bet_numbers = get_color_numbers(history, color_probs)
        for num in bet_numbers:
            bet_plan.append({'number': num, 'type': 'color', 'payout': 1, 'cost': 1})
    elif strategy_id == 8: # Dúzia
        bet_numbers = get_dozen_numbers(history, dozen_probs)
        for num in bet_numbers:
            bet_plan.append({'number': num, 'type': 'dozen', 'payout': 2, 'cost': 1})
            
    return bet_plan

# =========================================================
# === RECOMPENSA FOCADA NO LUCRO (NOVO) ===
# =========================================================
def compute_reward(bet_plan, outcome_number, bet_amount_unit=BET_AMOUNT_UNIT):
    total_cost = 0
    total_payout = 0
    outcome_color = number_to_color(outcome_number)
    outcome_dozen = number_to_dozen(outcome_number)

    for bet in bet_plan:
        total_cost += bet_amount_unit

        # Verifica se o número de aposta corresponde ao resultado
        if bet['number'] == outcome_number:
            total_payout += bet['payout'] * bet_amount_unit
        
        # Tipos de aposta mais genéricos (dúzias, cores, etc.)
        if bet['type'] == 'dozen' and number_to_dozen(bet['number']) == outcome_dozen:
            total_payout += bet['payout'] * bet_amount_unit
        elif bet['type'] == 'color' and number_to_color(bet['number']) == outcome_color:
            total_payout += bet['payout'] * bet_amount_unit

    profit = total_payout - total_cost
    return profit

# --- PREDICTION POSTPROCESSING ---
def predict_next_numbers(model, history, top_k=3):
    if history is None or len(history) < SEQUENCE_LEN or model is None:
        return []
    try:
        seq_one_hot = sequence_to_one_hot(history).reshape(1, SEQUENCE_LEN, NUM_TOTAL)
        feat = np.array([get_advanced_features(history[-SEQUENCE_LEN:])])
        raw = model.predict([seq_one_hot, feat], verbose=0)
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
    temperature = 0.8
    adjusted = np.log(num_probs + 1e-12) / temperature
    adjusted = np.exp(adjusted)
    adjusted /= adjusted.sum()
    weighted = []
    freq_counter = Counter(history[-100:])
    last_num = history[-1] if len(history) > 0 else None
    for num in range(NUM_TOTAL):
        freq_factor = 1 + np.exp(freq_counter.get(num, 0) / 3 - 1)
        if last_num in WHEEL_ORDER:
            dist = WHEEL_DISTANCE[last_num][num]
            distance_factor = max(0.1, 2.5 - (dist / 12.0))
        else:
            distance_factor = 1.0
        momentum = sum(1 for i in range(1,4) if len(history)>=i and history[-i] == num)
        momentum_factor = 1 + momentum*0.25
        pull_factor = 1.0
        if last_num in PULL_NUMBERS and num in PULL_NUMBERS[last_num]:
            pull_factor = 1.5
        elif any(last_num in v for v in PULL_NUMBERS.values() if num in v):
            pull_factor = 1.2
        weighted.append(adjusted[num] * freq_factor * distance_factor * momentum_factor * pull_factor)
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
        'dozen_probs': dozen_probs,
        'color_pred': color_pred,
        'dozen_pred': dozen_pred
    }

# =========================
# LSTM: construção de dataset e treino recente
# =========================
def build_lstm_supervised_from_history(history):
    data = []
    if len(history) <= SEQUENCE_LEN:
        return None
    start_idx = max(0, len(history) - (SEQUENCE_LEN + 1) - LSTM_RECENT_WINDOWS)
    for i in range(start_idx, len(history) - SEQUENCE_LEN - 1):
        seq_slice = history[i:i+SEQUENCE_LEN]
        target = history[i+SEQUENCE_LEN]
        X_seq = sequence_to_one_hot(seq_slice)
        X_feat = get_advanced_features(seq_slice)
        y_num = to_categorical(WHEEL_ORDER.index(target), NUM_TOTAL) if target in WHEEL_ORDER else np.zeros(NUM_TOTAL)
        y_color = to_categorical(number_to_color(target), 3)
        y_dozen = to_categorical(number_to_dozen(target), 4)
        data.append((X_seq, X_feat, y_num, y_color, y_dozen))
    if not data: return None
    X_seq, X_feat, y_num, y_color, y_dozen = zip(*data)
    return np.array(X_seq), np.array(X_feat), np.array(y_num), np.array(y_color), np.array(y_dozen)

def train_lstm_on_recent_minibatch(model, history):
    data = build_lstm_supervised_from_history(history)
    if data is None: return
    X_seq, X_feat, y_num, y_color, y_dozen = data
    n = len(X_seq)
    if n == 0: return
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
st.title("🔥 ROULETTE AI - LSTM + DQN com Estratégias de Aposta")
st.markdown("### Inserir histórico manualmente (ex: 0,32,15,19,4,21)")
if 'input_bulk' not in st.session_state:
    st.session_state.input_bulk = ""
if 'clear_input_bulk' not in st.session_state:
    st.session_state.clear_input_bulk = False
if st.session_state.clear_input_bulk:
    st.session_state.input_bulk = ""
    st.session_state.clear_input_bulk = False
input_bulk = st.text_area("Cole números separados por vírgula", key="input_bulk")
if st.button("Adicionar histórico"):
    if st.session_state.input_bulk and st.session_state.input_bulk.strip():
        try:
            new_nums = [
                int(x.strip())
                for x in st.session_state.input_bulk.split(",")
                if x.strip().isdigit() and 0 <= int(x.strip()) <= 36
            ]
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
        st.session_state.history.append(num)
        logger.info(f"Número novo inserido pelo usuário: {num}")
        st.session_state.last_input = None
        state_example = sequence_to_state(st.session_state.history, st.session_state.model)
        if state_example is not None and (st.session_state.dqn_agent is None):
            st.session_state.dqn_agent = DQNAgent(state_size=state_example.shape[0], action_size=ACTION_SIZE)
            logger.info("Agente DQN criado")
        if st.session_state.prev_state is not None and st.session_state.prev_action is not None:
            agent = st.session_state.dqn_agent
            bet_plan = st.session_state.bet_plan
            reward = compute_reward(bet_plan, num)
            next_state = sequence_to_state(st.session_state.history, st.session_state.model)
            if agent is not None:
                agent.remember(st.session_state.prev_state, st.session_state.prev_action, reward, next_state, False)
                logger.info(f"Memorizado: estratégia={st.session_state.prev_action}, resultado={num}, recompensa={reward}")
            st.session_state.stats['bets'] += 1
            st.session_state.stats['profit'] += reward
            st.session_state.stats['cost'] += sum(b['cost'] for b in bet_plan)
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
    except Exception as e:
        logger.exception("Erro inesperado ao processar entrada")
        st.error(f"Erro inesperado: {e}")
state = sequence_to_state(st.session_state.history, st.session_state.model)
agent = st.session_state.dqn_agent
pred_info = predict_next_numbers(st.session_state.model, st.session_state.history, top_k=3)
if state is not None and agent is None:
    st.session_state.dqn_agent = DQNAgent(state_size=state.shape[0], action_size=ACTION_SIZE)
    agent = st.session_state.dqn_agent
if agent is not None and state is not None and pred_info:
    strategy_id = agent.act(state, use_epsilon=True)
    st.subheader("🤖 Ações sugeridas pela IA (DQN)")
    st.write(f"**Estratégia escolhida:** **__{BETTING_STRATEGIES[strategy_id]}__**")
    bet_plan = plan_bet(strategy_id, pred_info, st.session_state.history)
    st.session_state.bet_plan = bet_plan
    if bet_plan:
        st.write("**Plano de Aposta:**")
        for bet in bet_plan:
            st.write(f"- Apostar no número: **{bet['number']}** (tipo: {bet['type']})")
    else:
        st.warning("Não foi possível gerar um plano de aposta com a estratégia escolhida.")
else:
    st.warning("Adicione mais números para a IA começar a analisar e gerar previsões.")
st.session_state.prev_state = state
st.session_state.prev_action = strategy_id if 'strategy_id' in locals() else None
st.markdown("---")
st.subheader("📊 Estatísticas da sessão")
st.write(f"Total de apostas: {st.session_state.stats['bets']}")
st.write(f"Total gasto: R$ {st.session_state.stats['cost']:.2f}")
st.write(f"Lucro/Prejuízo: R$ {st.session_state.stats['profit']:.2f}")
st.write(f"Números no histórico: {len(st.session_state.history)}")


