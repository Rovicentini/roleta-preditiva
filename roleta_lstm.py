# app_revised.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention, BatchNormalization
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from collections import Counter, deque
import random
import time
import logging

logging.basicConfig(filename='roleta.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger()

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

try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except Exception:
    KERAS_TUNER_AVAILABLE = False

NUM_TOTAL = 37
SEQUENCE_LEN = 20
BET_AMOUNT = 1.0
TARGET_UPDATE_FREQ = 50
REPLAY_BATCH = 64
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95

WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
WHEEL_DISTANCE = [[min(abs(i-j), 37-abs(i-j)) for j in range(37)] for i in range(37)]

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
            deceleration = 0
    else:
        wheel_speed = 0
        deceleration = 0
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

def sequence_to_state(sequence, model=None):
    seq = list(sequence[-SEQUENCE_LEN:]) if len(sequence) >= 0 else []
    pad = [ -1 ] * max(0, (SEQUENCE_LEN - len(seq)))
    seq_padded = pad + seq
    seq_norm = [(x/36.0 if (isinstance(x, (int, float)) and x>=0) else -1.0) for x in seq_padded]
    features = get_advanced_features(sequence[-SEQUENCE_LEN:]) if sequence else [0]*8
    probs = [0.0]*NUM_TOTAL
    if model is not None and len(sequence) >= SEQUENCE_LEN:
        try:
            seq_arr = np.array(sequence[-SEQUENCE_LEN:]).reshape(1, SEQUENCE_LEN, 1)
            feat_arr = np.array([features])
            raw = model.predict([seq_arr, feat_arr], verbose=0)
            if raw is not None and len(raw) > 0:
                probs = raw[0].tolist()
        except Exception:
            probs = [0.0]*NUM_TOTAL
    state = np.array(seq_norm + features + probs, dtype=np.float32)
    return state

def build_deep_learning_model(seq_len=SEQUENCE_LEN, num_total=NUM_TOTAL):
    seq_input = Input(shape=(seq_len, 1), name='sequence_input')
    lstm1 = LSTM(256, return_sequences=True,
                 kernel_regularizer=l2(0.001),
                 recurrent_regularizer=l2(0.001))(seq_input)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.4)(lstm1)
    lstm2 = LSTM(128, return_sequences=True)(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    attention = Attention(use_scale=True)([lstm2, lstm2])
    lstm3 = LSTM(64)(attention)
    feat_input = Input(shape=(8,), name='features_input')
    dense_feat = Dense(64, activation='swish')(feat_input)
    dense_feat = BatchNormalization()(dense_feat)
    dense_feat = Dropout(0.3)(dense_feat)
    combined = Concatenate()([lstm3, dense_feat])
    dense1 = Dense(256, activation='swish',
                   kernel_regularizer=l2(0.001))(combined)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    dense2 = Dense(128, activation='swish')(dense1)
    dense2 = BatchNormalization()(dense2)
    output = Dense(num_total, activation='softmax')(dense2)
    model = Model(inputs=[seq_input, feat_input], outputs=output)
    optimizer = Nadam(learning_rate=5e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)])
    return model

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

    # --- ALTERAÇÃO: Novo método para retornar TOP K ações ---
    def act_top_k(self, state, k=3, use_epsilon=True):
        if state is None or len(state) == 0:
            return random.sample(range(self.action_size), k)
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.sample(range(self.action_size), k)
        try:
            q_values = self.model.predict(np.array([state]), verbose=0)[0]
            top_k_actions = np.argsort(q_values)[-k:][::-1]  # top k maiores
            return top_k_actions.tolist()
        except Exception:
            return random.sample(range(self.action_size), k)

    # Método original de ação removido, vamos deixar para backward compatibility
    def act(self, state, use_epsilon=True):
        # Só retorna 1 ação (o maior Q)
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

def compute_reward(action_numbers, outcome_number, bet_amount=BET_AMOUNT, history=None):
    # Modificado para aceitar lista de ações e considerar vizinhos para acerto
    valid_numbers = set()
    for num in action_numbers:
        valid_numbers.add(num)
        if history is not None:
            neighbors = optimal_neighbors(num, history, max_neighbors=2)
            valid_numbers.update(neighbors)
    if outcome_number in valid_numbers:
        return 35.0 * bet_amount
    else:
        return -1.0 * bet_amount

def predict_next_numbers(model, history):
    if history is None or len(history) < SEQUENCE_LEN or model is None:
        return []
    try:
        seq = np.array(history[-SEQUENCE_LEN:]).reshape(1, SEQUENCE_LEN, 1)
        feat = np.array([get_advanced_features(history[-SEQUENCE_LEN:])])
        raw_pred = model.predict([seq, feat], verbose=0)[0]
    except Exception:
        return []
    temperature = 0.7
    adjusted = np.log(raw_pred + 1e-10) / temperature
    adjusted = np.exp(adjusted)
    adjusted /= adjusted.sum()
    weighted = []
    freq_counter = Counter(history[-100:])
    for num in range(NUM_TOTAL):
        freq_factor = 1 + np.exp(freq_counter.get(num, 0) / 3 - 1)
        if history and history[-1] in WHEEL_ORDER and num in WHEEL_ORDER:
            dist = WHEEL_DISTANCE[history[-1]][num]
            distance_factor = 2.5 - (dist / 12.0)
        else:
            distance_factor = 1.0
        momentum = sum(1 for i in range(1,4) if len(history)>=i and history[-i] == num)
        momentum_factor = 1 + momentum*0.3
        weighted.append(adjusted[num] * freq_factor * distance_factor * momentum_factor)
    weighted = np.array(weighted)
    if weighted.sum() == 0:
        return []
    weighted /= weighted.sum()
    top_indices = list(np.argsort(weighted)[-3:][::-1])  # LIMITADO para top 3 para menos confusão
    return [(i, float(weighted[i])) for i in top_indices]

def optimal_neighbors(number, history, max_neighbors=2):
    if number not in WHEEL_ORDER:
        return []
    idx = WHEEL_ORDER.index(number)
    neigh = []
    for i in range(1, max_neighbors+1):
        neigh.extend([WHEEL_ORDER[(idx-i)%37], WHEEL_ORDER[(idx+i)%37]])
    return list(dict.fromkeys(neigh))

st.set_page_config(layout="centered")
st.title("🔥 ROULETTE AI - LSTM + DQN (REVISADO)")

st.markdown("### Inserir histórico manualmente (ex: 0,32,15,19,4,21)")

input_bulk = st.text_area("Cole números separados por vírgula", value="")

if st.button("Adicionar histórico"):
    if input_bulk.strip():
        try:
            new_nums = [int(x.strip()) for x in input_bulk.split(",") if x.strip().isdigit() and 0 <= int(x.strip()) <= 36]
            st.session_state.history.extend(new_nums)
            st.success(f"Adicionados {len(new_nums)} números ao histórico.")
            logger.info(f"Usuário adicionou {len(new_nums)} números: {new_nums}")
            input_bulk = ""
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

        # --- Alteração aqui: considerar múltiplas ações (top 3) para cálculo de recompensa ---
        if st.session_state.prev_state is not None and st.session_state.prev_action is not None:
            agent = st.session_state.dqn_agent
            if agent is not None:
                # Obtem top 3 ações para calcular recompensa
                top_actions = agent.act_top_k(st.session_state.prev_state, k=3, use_epsilon=False)
            else:
                top_actions = [st.session_state.prev_action]

            reward = compute_reward(top_actions, num, bet_amount=BET_AMOUNT, history=st.session_state.history)
            next_state = sequence_to_state(st.session_state.history, st.session_state.model)

            if agent is not None:
                # Vamos guardar a ação principal (top 1) para o replay, mas pode-se modificar para usar top_actions se quiser
                agent.remember(st.session_state.prev_state, top_actions[0], reward, next_state, False)
                logger.info(f"Memorizado: ações={top_actions}, resultado={num}, recompensa={reward}")

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
            with st.spinner("Treinando LSTM (curto)..."):
                X_seq, X_feat, y = [], [], []
                for i in range(len(st.session_state.history) - SEQUENCE_LEN - 1):
                    seq = st.session_state.history[i:i+SEQUENCE_LEN]
                    X_seq.append(seq)
                    X_feat.append(get_advanced_features(seq))
                    y.append(st.session_state.history[i+SEQUENCE_LEN])
                if len(X_seq) > 0:
                    X_seq = np.array(X_seq).reshape(-1, SEQUENCE_LEN, 1)
                    X_feat = np.array(X_feat)
                    y_cat = tf.keras.utils.to_categorical(y, NUM_TOTAL)
                    try:
                        st.session_state.model.fit([X_seq, X_feat], y_cat, epochs=6, batch_size=32, verbose=0,
                                                  callbacks=[EarlyStopping(patience=2, restore_best_weights=True),
                                                             ReduceLROnPlateau(factor=0.5, patience=1)])
                        logger.info(f"Modelo LSTM treinado com {len(X_seq)} amostras")
                    except Exception as e:
                        logger.error(f"Erro no treinamento LSTM: {e}")
    except Exception as e:
        st.error("Erro ao processar entrada: " + str(e))

state_example = sequence_to_state(st.session_state.history, st.session_state.model)
if state_example is None or len(state_example) == 0:
    pass
else:
    if st.session_state.dqn_agent is None:
        st.session_state.dqn_agent = DQNAgent(state_size=state_example.shape[0], action_size=NUM_TOTAL)
        logger.info("Agente DQN criado")

if len(st.session_state.history) >= SEQUENCE_LEN and st.session_state.model is not None:
    predictions = predict_next_numbers(st.session_state.model, st.session_state.history)
    st.subheader("🎯 Previsões (LSTM + pós-processamento)")
    if predictions:
        for n, conf in predictions:
            st.write(f"Número: **{n}** — Prob: {conf:.2%}")
else:
    predictions = []

state = sequence_to_state(st.session_state.history, st.session_state.model)
agent = st.session_state.dqn_agent

if agent is not None and state is not None:
    top_actions = agent.act_top_k(state, k=3)
else:
    top_actions = random.sample(range(NUM_TOTAL), 3)

# Mostrar as ações e seus vizinhos:
st.subheader("🤖 Ações sugeridas pela IA (DQN) com vizinhos")
for action in top_actions:
    neighbors = optimal_neighbors(action, st.session_state.history, max_neighbors=2)
    st.write(f"Aposte no número: **{action}** | Vizinhos: {neighbors}")

# Guardar para próxima interação a ação principal (top 1)
st.session_state.prev_state = state
st.session_state.prev_action = top_actions[0]

st.markdown("---")
st.subheader("📊 Estatísticas da sessão")
st.write(f"Total de apostas: {st.session_state.stats['bets']}")
st.write(f"Vitórias: {st.session_state.stats['wins']}")
st.write(f"Lucro acumulado: R$ {st.session_state.stats['profit']:.2f}")
st.write(f"Sequência máxima de vitórias: {st.session_state.stats['max_streak']}")
