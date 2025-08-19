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
if 'prev_actions' not in st.session_state:
    st.session_state.prev_actions = None
if 'input_bulk' not in st.session_state:
    st.session_state.input_bulk = ""
if 'clear_input_bulk' not in st.session_state:
    st.session_state.clear_input_bulk = False
if 'pair_freq' not in st.session_state:
    st.session_state.pair_freq = Counter()
if 'triplet_freq' not in st.session_state:
    st.session_state.triplet_freq = Counter()

# --- CONSTANTS ---
NUM_TOTAL = 37
SEQUENCE_LEN = 20
BET_AMOUNT = 1.0
FEATURE_DIM = 10
TARGET_UPDATE_FREQ = 50
REPLAY_BATCH = 64
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95
REWARD_EXACT = 35.0
REWARD_NEIGHBOR = 8.0
REWARD_LOSS = -15.0
NEIGHBOR_RADIUS_FOR_REWARD = 1
LSTM_RECENT_WINDOWS = 400
LSTM_BATCH_SAMPLES = 128
LSTM_EPOCHS_PER_STEP = 2
LSTM_BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.992
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,
               5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
WHEEL_DISTANCE = [[min(abs(i-j), 37-abs(i-j)) for j in range(37)] for i in range(37)]
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

# =========================
# Lógica Principal em Função
# =========================
def update_pair_triplet_counters_from(history, start_index):
    """Atualiza os contadores de pares e trincas."""
    if start_index is None:
        start_index = 0
    for i in range(start_index, len(history)):
        if i >= 1:
            pair = (history[i-1], history[i])
            st.session_state.pair_freq[pair] += 1
        if i >= 2:
            triplet = (history[i-2], history[i-1], history[i])
            st.session_state.triplet_freq[triplet] += 1

def number_to_color(n):
    if n == 0: return 0
    return 1 if n in RED_NUMBERS else 2

def number_to_dozen(n):
    if n == 0: return 0
    if 1 <= n <= 12: return 1
    if 13 <= n <= 24: return 2
    return 3

def get_advanced_features(sequence):
    if sequence is None or len(sequence) < 2:
        return [0.0] * FEATURE_DIM
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
    last_pair_count = 0.0
    last_triplet_count = 0.0
    if len(sequence) >= 2:
        last_pair = (sequence[-2], sequence[-1])
        last_pair_count = st.session_state.pair_freq.get(last_pair, 0)
    if len(sequence) >= 3:
        last_triplet = (sequence[-3], sequence[-2], sequence[-1])
        last_triplet_count = st.session_state.triplet_freq.get(last_triplet, 0)
    max_pair_freq = max(st.session_state.pair_freq.values()) if st.session_state.pair_freq else 1
    max_triplet_freq = max(st.session_state.triplet_freq.values()) if st.session_state.triplet_freq else 1
    normalized_pair_freq = (last_pair_count / max_pair_freq) if max_pair_freq > 0 else 0.0
    normalized_triplet_freq = (last_triplet_count / max_triplet_freq) if max_triplet_freq > 0 else 0.0
    base_features = [
        mean / 36.0,
        std / 18.0,
        wheel_speed / 36.0,
        deceleration / 36.0,
        freq.get(last, 0) / hot_number if hot_number > 0 else 0.0,
        (hot_number - cold_number) / len(sequence) if len(sequence) > 0 else 0.0,
        len(freq) / len(sequence) if len(sequence) > 0 else 0.0,
        1.0 if last == second_last else 0.0
    ]
    return base_features + [normalized_pair_freq, normalized_triplet_freq]

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
    one_hot_seq = sequence_to_one_hot(sequence)
    features = get_advanced_features(sequence[-SEQUENCE_LEN:]) if sequence else [0.0] * FEATURE_DIM
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
                out = np.array(raw)
                if out.shape[-1] == NUM_TOTAL:
                    num_probs = out[0]
        except Exception:
            num_probs = np.zeros(NUM_TOTAL)
    state = np.concatenate([one_hot_seq.flatten(), np.array(features), num_probs, color_probs, dozen_probs]).astype(np.float32)
    return state

def build_deep_learning_model():
    seq_input = Input(shape=(SEQUENCE_LEN, NUM_TOTAL), name='sequence_input')
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
    feat_input = Input(shape=(FEATURE_DIM,), name='features_input')
    dense_feat = Dense(48, activation='swish')(feat_input)
    dense_feat = BatchNormalization()(dense_feat)
    dense_feat = Dropout(0.2)(dense_feat)
    combined = Concatenate()([x, dense_feat])
    dense = Dense(160, activation='swish')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)
    out_num = Dense(NUM_TOTAL, activation='softmax', name='num_out')(dense)
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
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), k)
        try:
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            top_k_actions = np.argsort(q_values)[-k:][::-1]
            return top_k_actions.tolist()
        except Exception:
            return random.sample(range(self.action_size), k)
    
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
        except Exception:
            pass
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def optimal_neighbors(number, max_neighbors=2):
    if number not in WHEEL_ORDER:
        return []
    idx = WHEEL_ORDER.index(number)
    neigh = []
    for i in range(1, max_neighbors + 1):
        neigh.append(WHEEL_ORDER[(idx - i) % NUM_TOTAL])
        neigh.append(WHEEL_ORDER[(idx + i) % NUM_TOTAL])
    return list(dict.fromkeys(neigh))

def compute_reward(action_numbers, outcome_number, bet_amount=BET_AMOUNT,
                   max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD):
    reward = 0.0
    action_numbers = list(set([a for a in action_numbers if 0 <= a <= 36]))
    if outcome_number in action_numbers:
        reward = REWARD_EXACT
    elif max_neighbors_for_reward > 0:
        all_neighbors = set()
        for a in action_numbers:
            all_neighbors.update(optimal_neighbors(a, max_neighbors=max_neighbors_for_reward))
        if outcome_number in all_neighbors:
            reward = REWARD_NEIGHBOR
    if reward == 0.0:
        reward = REWARD_LOSS
    return reward * bet_amount

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
        momentum = sum(1 for i in range(1, 4) if len(history) >= i and history[-i] == num)
        momentum_factor = 1 + momentum * 0.25
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

def build_lstm_supervised_from_history(history):
    X_seq, X_feat, y_num, y_color, y_dozen = [], [], [], [], []
    if len(history) <= SEQUENCE_LEN:
        return None
    start_idx = max(0, len(history) - (SEQUENCE_LEN + 1) - LSTM_RECENT_WINDOWS)
    for i in range(start_idx, len(history) - SEQUENCE_LEN - 1):
        seq_slice = history[i:i + SEQUENCE_LEN]
        target = history[i + SEQUENCE_LEN]
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

# =========================
# Lógica do App
# =========================
def process_new_number(num):
    """
    Processa um novo número, atualiza o estado da aplicação e treina as IAs.
    """
    st.session_state.history.append(num)
    update_pair_triplet_counters_from(st.session_state.history, len(st.session_state.history) - 1)
    logger.info(f"Número novo inserido pelo usuário: {num}")

    state_example = sequence_to_state(st.session_state.history, st.session_state.model)
    if state_example is not None and (st.session_state.dqn_agent is None):
        st.session_state.dqn_agent = DQNAgent(state_size=state_example.shape[0], action_size=NUM_TOTAL)
        logger.info("Agente DQN criado")
    
    if st.session_state.prev_state is not None and st.session_state.prev_actions is not None:
        agent = st.session_state.dqn_agent
        next_state = sequence_to_state(st.session_state.history, st.session_state.model)
        total_reward = 0.0
        
        for action_number in st.session_state.prev_actions:
            individual_reward = compute_reward(action_numbers=[action_number],
                                               outcome_number=num,
                                               bet_amount=BET_AMOUNT,
                                               max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD)
            
            if agent is not None:
                agent.remember(st.session_state.prev_state, action_number, individual_reward, next_state, False)
                logger.info(f"Memorizado: aposta={action_number}, resultado={num}, recompensa={individual_reward}")
            
            total_reward += individual_reward

        st.session_state.stats['bets'] += 1
        st.session_state.stats['profit'] += total_reward
        
        if total_reward > 0:
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

    if st.session_state.model is None and len(st.session_state.history) >= SEQUENCE_LEN * 2:
        st.session_state.model = build_deep_learning_model()
        logger.info("Modelo LSTM criado")

    if st.session_state.model is not None and len(st.session_state.history) > SEQUENCE_LEN * 2:
        with st.spinner("Treinando LSTM com mini-batches recentes..."):
            train_lstm_on_recent_minibatch(st.session_state.model, st.session_state.history)

    st.session_state.prev_state = sequence_to_state(st.session_state.history, st.session_state.model)
    st.session_state.last_input = None

# =========================
# UI e Fluxo Principal
# =========================
st.set_page_config(layout="centered")
st.title("🔥 ROULETTE AI - LSTM multi-saída + DQN (Refinado)")

st.markdown("### Inserir histórico manualmente (ex: 0,32,15,19,4,21)")

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
            if new_nums:
                prev_len = len(st.session_state.history)
                st.session_state.history.extend(new_nums)
                update_pair_triplet_counters_from(st.session_state.history, max(prev_len, 0))
                st.success(f"Adicionados {len(new_nums)} números ao histórico.")
            else:
                st.warning("Nenhum número válido encontrado (0–36).")
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
        process_new_number(num)
    except Exception as e:
        logger.exception("Erro inesperado ao processar entrada")
        st.error(f"Erro inesperado: {e}")

st.markdown("---")

# Se o modelo LSTM existe, exibe as previsões
if st.session_state.model is not None:
    pred_info = predict_next_numbers(st.session_state.model, st.session_state.history, top_k=3)
    st.subheader("🎯 Previsões (LSTM + pós-processamento)")
    if pred_info:
        for n, conf in pred_info['top_numbers']:
            st.write(f"Número: **{n}** — Prob: {conf:.2%}")

# Exibe as sugestões DQN (sempre, já que a lógica de treino é incremental)
state = sequence_to_state(st.session_state.history, st.session_state.model)
agent = st.session_state.dqn_agent

if state is not None and agent is None:
    st.session_state.dqn_agent = DQNAgent(state_size=state.shape[0], action_size=NUM_TOTAL)
    agent = st.session_state.dqn_agent
    logger.info("Agente DQN criado (depois de estado)")

if agent is not None and state is not None:
    top_actions = agent.act_top_k(state, k=3, use_epsilon=True)
else:
    top_actions = random.sample(range(NUM_TOTAL), 3)

st.subheader("🤖 Ações sugeridas pela IA (DQN) com vizinhos")
for action in top_actions:
    neighbors = optimal_neighbors(action, max_neighbors=2)
    st.write(f"Aposte no número: **{action}** | Vizinhos (2 cada lado): {neighbors}")

st.markdown("---")
st.subheader("📊 Estatísticas da sessão")
st.write(f"Total de apostas: {st.session_state.stats['bets']}")
st.write(f"Vitórias: {st.session_state.stats['wins']}")
st.write(f"Lucro acumulado: R$ {st.session_state.stats['profit']:.2f}")
st.write(f"Sequência máxima de vitórias: {st.session_state.stats['max_streak']}")
st.write(f"Números no histórico: {len(st.session_state.history)}")
