import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.utils import to_categorical
from collections import Counter, deque
import random
import logging

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
if 'input_bulk_temp' not in st.session_state:
    st.session_state.input_bulk_temp = ""

# --- CONSTANTS ---
NUM_TOTAL = 37
SEQUENCE_LEN = 20
BET_AMOUNT = 1.0
TARGET_UPDATE_FREQ = 50
REPLAY_BATCH = 64
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95

# wheel order (posição física na roda)
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,
               5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
WHEEL_DISTANCE = [[min(abs(i-j), 37-abs(i-j)) for j in range(37)] for i in range(37)]

# red numbers for European wheel (standard)
RED_NUMBERS = {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}

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
    return np.array(one_hot_seq)  # shape (SEQUENCE_LEN, NUM_TOTAL)

def sequence_to_state(sequence, model=None):
    one_hot_seq = sequence_to_one_hot(sequence)  # (SEQ, 37)
    features = get_advanced_features(sequence[-SEQUENCE_LEN:]) if sequence else [0.0]*8

    num_probs = np.zeros(NUM_TOTAL)
    color_probs = np.zeros(3)   # [zero, red, black]
    dozen_probs = np.zeros(4)   # [zero, d1, d2, d3]

    if model is not None and len(sequence) >= SEQUENCE_LEN:
        try:
            seq_arr = np.expand_dims(one_hot_seq, axis=0)  # (1, SEQ, 37)
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

def build_deep_learning_model(seq_len=SEQUENCE_LEN, num_total=NUM_TOTAL):
    seq_input = Input(shape=(seq_len, num_total), name='sequence_input')
    lstm1 = LSTM(128, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(seq_input)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)

    lstm2 = LSTM(64, return_sequences=False)(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.25)(lstm2)

    feat_input = Input(shape=(8,), name='features_input')
    dense_feat = Dense(32, activation='swish')(feat_input)
    dense_feat = BatchNormalization()(dense_feat)
    dense_feat = Dropout(0.2)(dense_feat)

    combined = Concatenate()([lstm2, dense_feat])
    dense = Dense(128, activation='swish')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.3)(dense)

    out_num = Dense(num_total, activation='softmax', name='num_out')(dense)
    out_color = Dense(3, activation='softmax', name='color_out')(dense)
    out_dozen = Dense(4, activation='softmax', name='dozen_out')(dense)

    model = Model(inputs=[seq_input, feat_input], outputs=[out_num, out_color, out_dozen])
    optimizer = Nadam(learning_rate=5e-4)
    model.compile(optimizer=optimizer,
                  loss={'num_out': 'categorical_crossentropy',
                        'color_out': 'categorical_crossentropy',
                        'dozen_out': 'categorical_crossentropy'},
                  loss_weights={'num_out': 1.0, 'color_out': 0.3, 'dozen_out': 0.3},
                  metrics={'num_out': 'accuracy'})
    return model

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size, lr=DQN_LEARNING_RATE, gamma=DQN_GAMMA, replay_size=REPLAY_SIZE):
        self.state_size = int(state_size)
        self.action_size = action_size
        self.memory = deque(maxlen=replay_size)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target()
        self.train_step = 0

    def _build_model(self):
        model = tf.keras.Sequential([
            Dense(256, activation='relu', input_shape=(self.state_size,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
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
        X = []
        Y = []
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

    def load(self, path):
        self.model.load_weights(path)
        self.update_target()

    def save(self, path):
        self.model.save_weights(path)

# --- Neighbors & Reward helpers ---
def optimal_neighbors(number, max_neighbors=2):
    if number not in WHEEL_ORDER:
        return []
    idx = WHEEL_ORDER.index(number)
    neigh = []
    for i in range(1, max_neighbors+1):
        neigh.append(WHEEL_ORDER[(idx - i) % NUM_TOTAL])
        neigh.append(WHEEL_ORDER[(idx + i) % NUM_TOTAL])
    return list(dict.fromkeys(neigh))

def compute_reward(action_numbers, outcome_number, bet_amount=BET_AMOUNT, max_neighbors=2):
    valid = set()
    for n in action_numbers:
        valid.add(n)
        neigh = optimal_neighbors(n, max_neighbors=max_neighbors)
        valid.update(neigh)
    if outcome_number in valid:
        return 35.0 * bet_amount
    return -1.0 * bet_amount

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
        weighted.append(adjusted[num] * freq_factor * distance_factor * momentum_factor)
    weighted = np.array(weighted)
    if weighted.sum() == 0:
        return []
    weighted /= weighted.sum()

    top_indices = list(np.argsort(weighted)[-top_k:][::-1])
    color_pred = np.argmax(color_probs)
    dozen_pred = np.argmax(dozen_probs)
    return {
        'top_numbers': [(int(i), float(weighted[i])) for i in top_indices],
        'num_probs': num_probs,
        'color_probs': color_probs,
        'dozen_probs': dozen_probs,
        'color_pred': int(color_pred),
        'dozen_pred': int(dozen_pred)
    }

# --- UI ---
st.set_page_config(layout="centered")
st.title("🔥 ROULETTE AI - LSTM multi-saída + DQN (REVISADO)")

st.markdown("### Inserir histórico manualmente (ex: 0,32,15,19,4,21)")

# Campo de input separado do session_state para evitar erro "cannot be modified after widget"
input_bulk_temp = st.text_area("Cole números separados por vírgula", value=st.session_state.input_bulk_temp, key="input_bulk_temp")

if st.button("Adicionar histórico"):
    if input_bulk_temp and input_bulk_temp.strip():
        try:
            new_nums = [int(x.strip()) for x in input_bulk_temp.split(",") if x.strip().isdigit() and 0 <= int(x.strip()) <= 36]
            st.session_state.history.extend(new_nums)
            st.success(f"Adicionados {len(new_nums)} números ao histórico.")
            logger.info(f"Usuário adicionou {len(new_nums)} números: {new_nums}")
            # limpa o campo após adição
            st.session_state.input_bulk_temp = ""
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
            st.session_state.dqn_agent = DQNAgent(state_size=state_example.shape[0], action_size=NUM_TOTAL)
            logger.info("Agente DQN criado")

        if st.session_state.prev_state is not None and st.session_state.prev_action is not None:
            agent = st.session_state.dqn_agent
            if agent is not None:
                top_actions = agent.act_top_k(st.session_state.prev_state, k=3, use_epsilon=True)
                reward = compute_reward(top_actions, num, bet_amount=BET_AMOUNT)
                done = False
                agent.remember(st.session_state.prev_state, st.session_state.prev_action, reward, state_example, done)
                st.session_state.stats['bets'] += 1
                if reward > 0:
                    st.session_state.stats['wins'] += 1
                    st.session_state.stats['streak'] += 1
                    st.session_state.stats['profit'] += reward - BET_AMOUNT
                    if st.session_state.stats['streak'] > st.session_state.stats['max_streak']:
                        st.session_state.stats['max_streak'] = st.session_state.stats['streak']
                else:
                    st.session_state.stats['streak'] = 0
                    st.session_state.stats['profit'] -= BET_AMOUNT

                if st.session_state.stats['bets'] % DQN_TRAIN_EVERY == 0:
                    agent.replay()
                if st.session_state.stats['bets'] % TARGET_UPDATE_FREQ == 0:
                    agent.update_target()

        pred = predict_next_numbers(st.session_state.model, st.session_state.history, top_k=3)
        state_for_dqn = sequence_to_state(st.session_state.history, st.session_state.model)

        st.session_state.prev_state = state_for_dqn
        if pred and 'top_numbers' in pred:
            top_pred = pred['top_numbers'][0][0]
            st.session_state.prev_action = top_pred
        else:
            st.session_state.prev_action = random.randint(0, NUM_TOTAL-1)

    except Exception as e:
        st.error(f"Erro ao adicionar número: {e}")

if len(st.session_state.history) == 0:
    st.warning("Histórico vazio. Adicione números para iniciar predições.")

st.markdown("### Histórico completo:")
st.write(st.session_state.history[-100:])

# Carregar/criar modelo LSTM
if st.session_state.model is None:
    with st.spinner("Criando modelo LSTM..."):
        st.session_state.model = build_deep_learning_model()

# Mostrar estatísticas
stats = st.session_state.stats
st.markdown("---")
st.markdown(f"**Estatísticas:** Apostas: {stats['bets']} | Acertos: {stats['wins']} | Sequência de vitórias: {stats['streak']} | Maior sequência: {stats['max_streak']} | Lucro estimado: R$ {stats['profit']:.2f}")

# Mostrar predições LSTM
if 'pred' in locals() and pred:
    st.markdown("### Predições LSTM multi-saída:")
    top = pred['top_numbers']
    st.write([f"Número {num} — probabilidade ajustada {prob:.2f}" for num, prob in top])
    st.write(f"Previsão de cor (0=zero,1=vermelho,2=preto): {pred['color_pred']}")
    st.write(f"Previsão de dezena (0=zero,1=1-12,2=13-24,3=25-36): {pred['dozen_pred']}")

# Mostrar ação DQN atual
if st.session_state.dqn_agent is not None and st.session_state.prev_action is not None:
    st.markdown("### Ação DQN selecionada para aposta:")
    st.write(f"Apostar no número: {st.session_state.prev_action}")

# Botão para limpar histórico
if st.button("Limpar histórico e estatísticas"):
    st.session_state.history = []
    st.session_state.stats = {'wins': 0, 'bets': 0, 'streak': 0, 'max_streak': 0, 'profit': 0.0}
    st.session_state.prev_state = None
    st.session_state.prev_action = None
    st.session_state.dqn_agent = None
    st.session_state.last_input = None
    st.session_state.input_bulk_temp = ""
    st.success("Histórico e estatísticas limpos.")

