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
from tensorflow.keras.layers import LeakyReLU

from collections import Counter, deque
import random
import logging

# =========================
# Utils Streamlit
# =========================
def rerun():
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
if 'co_occurrence_matrix' not in st.session_state:
    st.session_state.co_occurrence_matrix = np.zeros((37, 37))
    
# MUDANÇA: Adicionado estado para armazenar estatísticas de normalização
if 'feat_stats' not in st.session_state:
    # Exemplo de valores. VOCÊ DEVE SUBSTITUIR ISSO POR ESTATÍSTICAS REAIS
    # CALCULADAS EM UM GRANDE HISTÓRICO.
    st.session_state.feat_stats = {
        'means': np.array([0.5, 0.25, 0.5, 0.2, 0.5, 0.5, 0.5, 0.1]),
        'stds': np.array([0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05])
    }

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
DQN_GAMMA = 0.50

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
# MUDANÇA: Adicionado limiar de confiança para aposta
CONFIDENCE_THRESHOLD = 0.1

# wheel order (posição física na roda)
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,
                5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
WHEEL_DISTANCE = [[min(abs(i-j), 37-abs(i-j)) for j in range(37)] for i in range(37)]

# red numbers for European wheel (standard)
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

# Definindo as principais regiões da roleta
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
    
# MUDANÇA: A função agora pode normalizar usando estatísticas fornecidas
def get_advanced_features(sequence, feat_means=None, feat_stds=None):
    if sequence is None or len(sequence) < 2:
        # Se os dados estiverem faltando, retorna features normalizadas para zero
        return np.zeros(8)
    
    seq = np.array(sequence)
    
    # 1. Média e Desvio Padrão
    mean = np.mean(seq) / 36.0
    std = np.std(seq) / 18.0
    
    # 2. Velocidade e desaceleração da roda
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
        
    wheel_speed /= 36.0
    deceleration /= 36.0

    # 3. Frequências e repetições
    freq = Counter(sequence)
    hot_number_count = max(freq.values()) if freq else 1
    cold_number_count = min(freq.values()) if freq else 0
    
    last_freq_norm = freq.get(last, 0) / hot_number_count if hot_number_count > 0 else 0.0
    freq_range_norm = (hot_number_count - cold_number_count) / len(sequence) if len(sequence) > 0 else 0.0
    unique_ratio = len(freq) / len(sequence) if len(sequence) > 0 else 0.0
    is_repeat = 1.0 if last == second_last else 0.0

    features = np.array([
        mean, std, wheel_speed, deceleration,
        last_freq_norm, freq_range_norm, unique_ratio, is_repeat
    ])

    # MUDANÇA: Normalização usando as estatísticas fornecidas
    if feat_means is not None and feat_stds is not None:
        features = (features - feat_means) / (feat_stds + 1e-6) # Adiciona epsilon para evitar divisão por zero
    
    return features

# Nova função para atualizar a matriz de co-ocorrência
def update_co_occurrence_matrix(matrix, history):
    if len(history) >= 2:
        prev_num = history[-2]
        curr_num = history[-1]
        if 0 <= prev_num < NUM_TOTAL and 0 <= curr_num < NUM_TOTAL:
            matrix[prev_num][curr_num] += 1
    return matrix

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

def sequence_to_state(sequence, model=None, feat_means=None, feat_stds=None):
    seq_slice = sequence[-SEQUENCE_LEN:] if sequence else []
    
    # MUDANÇA: Passando as estatísticas para a função de features
    features = get_advanced_features(seq_slice, feat_means, feat_stds)
    one_hot_seq = sequence_to_one_hot(seq_slice)

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
            else:
                num_probs = np.array(raw)[0]
                color_probs = np.array([0.0, 0.0, 0.0])
                dozen_probs = np.array([0.0, 0.0, 0.0, 0.0])
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
    age_vector = np.array([age / max(1, max_age) for age in age_vector])

    last_run_len_num = 0
    if len(sequence) >= 2:
        for i in range(1, len(sequence) + 1):
            if sequence[-i] == sequence[-1]:
                last_run_len_num = i
            else:
                break
    last_run_len_color = 0
    if len(sequence) >= 2:
        last_color = number_to_color(sequence[-1])
        for i in range(1, len(sequence) + 1):
            if number_to_color(sequence[-i]) == last_color:
                last_run_len_color = i
            else:
                break
    last_run_len_dozen = 0
    if len(sequence) >= 2:
        last_dozen = number_to_dozen(sequence[-1])
        for i in range(1, len(sequence) + 1):
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

    last_num_pulling_strength = np.zeros(NUM_TOTAL)
    if len(sequence) > 0:
        last_num = sequence[-1]
        if 0 <= last_num < NUM_TOTAL:
            last_num_pulling_strength = st.session_state.co_occurrence_matrix[last_num, :].copy()
            total_co_occurrences = np.sum(last_num_pulling_strength)
            if total_co_occurrences > 0:
                last_num_pulling_strength /= total_co_occurrences
    
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
        last_num_pulling_strength
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

    # MUDANÇA: Arquitetura mais enxuta e LeakyReLU
    def _build_model(self):
        model = tf.keras.Sequential([
            Dense(256, input_shape=(self.state_size,)),
            LeakyReLU(alpha=0.1),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128),
            LeakyReLU(alpha=0.1),
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

    # MUDANÇA: Adicionado limiar de confiança
    def act_top_k(self, state, k=3, use_epsilon=True):
        if state is None or len(state) == 0:
            return random.sample(range(self.action_size), k)
        
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), k)
        
        try:
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            
            top_k_actions = []
            sorted_indices = np.argsort(q_values)[::-1]
            
            # Filtra por limiar de confiança
            for idx in sorted_indices:
                if q_values[idx] > CONFIDENCE_THRESHOLD:
                    top_k_actions.append(int(idx))
                if len(top_k_actions) >= k:
                    break
            
            # Se não houver números acima do limiar, retorna os mais prováveis
            if not top_k_actions:
                return list(sorted_indices[:k])
            
            return top_k_actions
            
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
        for i, (state, actions, reward, next_state, done) in enumerate(batch):
            target = q_curr[i].copy()
            
            for action in actions:
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

# =========================
# RECOMPENSA FOCADA E SIMPLIFICADA
# =========================
# MUDANÇA: Recompensa com penalidade por número de ações
def compute_reward(action_numbers, outcome_number, bet_amount=BET_AMOUNT,
                   max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD):
    reward = 0.0
    action_numbers = set([a for a in action_numbers if 0 <= a <= 36])
    
    # Adiciona uma pequena penalidade por cada aposta
    bet_penalty = 0.1 * len(action_numbers)

    # 1. Acerto Exato
    if outcome_number in action_numbers:
        reward = REWARD_EXACT - bet_penalty
    # 2. Acerto Vizinho
    else:
        all_neighbors = set()
        for a in action_numbers:
            all_neighbors.update(optimal_neighbors(a, max_neighbors=max_neighbors_for_reward))
        
        if outcome_number in all_neighbors:
            reward = REWARD_NEIGHBOR - bet_penalty
        # 3. Perda total
        else:
            reward = REWARD_LOSS - bet_penalty

    return reward * bet_amount

# --- PREDICTION POSTPROCESSING ---
def predict_next_numbers(model, history, top_k=3):
    if history is None or len(history) < SEQUENCE_LEN or model is None:
        return []
    try:
        # MUDANÇA: Passando as estatísticas para a função de features
        feat = np.array([get_advanced_features(history[-SEQUENCE_LEN:],
                                             st.session_state.feat_stats['means'],
                                             st.session_state.feat_stats['stds'])])
        seq_one_hot = sequence_to_one_hot(history).reshape(1, SEQUENCE_LEN, NUM_TOTAL)
        raw = model.predict([seq_one_hot, feat], verbose=0)
        
        if isinstance(raw, list) and len(raw) == 3:
            num_probs = raw[0][0]
            color_probs = raw[1][0]
            dozen_probs = raw[2][0]
        else:
            num_probs = np.array(raw)[0]
            color_probs = np.array([0.0, 0.0, 0.0])
            dozen_probs = np.array([0.0, 0.0, 0.0, 0.0])
    except Exception as e:
        logger.error(f"Erro na previsão LSTM: {e}")
        return []

    temperature = 0.4
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
        weighted.append(adjusted[num] * freq_factor * distance_factor * momentum_factor)
    
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
    X_seq, X_feat, y_num, y_color, y_dozen = [], [], [], [], []
    if len(history) <= SEQUENCE_LEN:
        return None

    start_idx = max(0, len(history) - (SEQUENCE_LEN + 1) - LSTM_RECENT_WINDOWS)
    for i in range(start_idx, len(history) - SEQUENCE_LEN - 1):
        seq_slice = history[i:i+SEQUENCE_LEN]
        target = history[i+SEQUENCE_LEN]

        X_seq.append(sequence_to_one_hot(seq_slice))
        # MUDANÇA: Passando as estatísticas para a função de features
        X_feat.append(get_advanced_features(seq_slice,
                                           st.session_state.feat_stats['means'],
                                           st.session_state.feat_stats['stds']))

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
    data = build_lstm_supervised_from_history(history)
    if data is None:
        return
    X_seq, X_feat, y_num, y_color, y_dozen = data
    n = len(X_seq)
    if n == 0:
        return

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
st.title("🔥 ROULETTE AI - LSTM multi-saída + DQN (REVISADO)")

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
            
            for i in range(len(new_nums)):
                st.session_state.history.append(new_nums[i])
                st.session_state.co_occurrence_matrix = update_co_occurrence_matrix(st.session_state.co_occurrence_matrix, st.session_state.history)

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
        
        st.session_state.co_occurrence_matrix = update_co_occurrence_matrix(st.session_state.co_occurrence_matrix, st.session_state.history)
        
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
            # MUDANÇA: Passando as estatísticas para o state
            next_state = sequence_to_state(st.session_state.history, st.session_state.model,
                                            st.session_state.feat_stats['means'],
                                            st.session_state.feat_stats['stds'])

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
            
            # MUDANÇA: Passando as estatísticas para o state
            st.session_state.prev_state = sequence_to_state(st.session_state.history, st.session_state.model,
                                                             st.session_state.feat_stats['means'],
                                                             st.session_state.feat_stats['stds'])
            pred_info = predict_next_numbers(st.session_state.model, st.session_state.history, top_k=3)

            if pred_info:
                st.subheader("🎯 Previsões (LSTM + pós-processamento)")
                for n, conf in pred_info['top_numbers']:
                    st.write(f"Número: **{n}** — Prob: {conf:.2%}")
                color_names = {0: "Zero", 1: "Vermelho", 2: "Preto"}
                dozen_names = {0: "Zero", 1: "1ª dúzia (1-12)", 2: "2ª dúzia (13-24)", 3: "3ª dúzia (25-36)"}
                st.write(f"Cor mais provável: **{color_names.get(pred_info['color_pred'],'-')}** — probs: {pred_info['color_probs']}")
                st.write(f"Dúzia mais provável: **{dozen_names.get(pred_info['dozen_pred'],'-')}** — probs: {pred_info['dozen_probs']}")
        else:
            # MUDANÇA: Passando as estatísticas para o state
            st.session_state.prev_state = sequence_to_state(st.session_state.history, st.session_state.model,
                                                             st.session_state.feat_stats['means'],
                                                             st.session_state.feat_stats['stds'])

    except Exception as e:
        logger.exception("Erro inesperado ao processar entrada")
        st.error(f"Erro inesperado: {e}")

# MUDANÇA: Passando as estatísticas para o state
state = sequence_to_state(st.session_state.history, st.session_state.model,
                           st.session_state.feat_stats['means'],
                           st.session_state.feat_stats['stds'])
agent = st.session_state.dqn_agent

if state is not None and agent is None:
    st.session_state.dqn_agent = DQNAgent(state_size=state.shape[0], action_size=NUM_TOTAL)
    agent = st.session_state.dqn_agent
    logger.info("Agente DQN criado (depois de estado)")

if agent is not None and state is not None:
    top_actions = agent.act_top_k(state, k=3, use_epsilon=True)
else:
    top_actions = random.sample(range(NUM_TOTAL), 3)
    
# MUDANÇA: Salvando as ações sugeridas
st.session_state.prev_actions = top_actions

st.subheader("🤖 Ações sugeridas pela IA (DQN) com vizinhos")
for action_num in top_actions:
    neighbors = optimal_neighbors(action_num, max_neighbors=NEIGHBOR_RADIUS_FOR_REWARD)
    st.write(f"- Aposta: **{action_num}** (Vizinhos: {', '.join(map(str, neighbors))})")

st.markdown("---")
st.subheader("📊 Estatísticas do Agente")
stats = st.session_state.stats
if stats['bets'] > 0:
    win_rate = (stats['wins'] / stats['bets']) * 100
    st.write(f"**Apostas:** {stats['bets']} | **Vitórias:** {stats['wins']} | **Taxa de Vitoria:** {win_rate:.2f}%")
    st.write(f"**Seq. de Vitórias:** {stats['streak']} | **Máx. Seq. de Vitórias:** {stats['max_streak']}")
    profit_color = "green" if stats['profit'] >= 0 else "red"
    st.markdown(f"**Lucro total:** <span style='color:{profit_color}'>${stats['profit']:.2f}</span>", unsafe_allow_html=True)
else:
    st.write("Ainda não há estatísticas de apostas.")

st.subheader("🎲 Histórico")
st.write(", ".join(map(str, st.session_state.history[::-1])))


