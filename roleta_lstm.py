# app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

from collections import Counter, deque
import random
import time


defaults = {
    'stats': {'bets': 0, 'hits': 0, 'balance': 0.0},
    'dqn_agent': None,
    'model': None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value



# Optional: Keras Tuner (may need pip install keras-tuner)
try:
    import keras_tuner as kt
    KERAS_TUNER_AVAILABLE = True
except Exception:
    KERAS_TUNER_AVAILABLE = False

# ---------- CONFIGURAÇÕES GLOBAIS ----------
NUM_TOTAL = 37            # 0..36 (roleta europeia)
SEQUENCE_LEN = 20         # janela temporal
BET_AMOUNT = 1.0          # aposta unitária (ajuste conforme desejar)
TARGET_UPDATE_FREQ = 50   # passos para sincronizar target network do DQN
REPLAY_BATCH = 64
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5       # treinar DQN a cada N novas experiências
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.95

# roda (ordem física) da roleta europeia (padrão)
WHEEL_ORDER = [0,32,15,19,4,21,2,25,17,34,6,27,13,36,11,30,8,23,10,5,24,16,33,1,20,14,31,9,22,18,29,7,28,12,35,3,26]
WHEEL_DISTANCE = [[min(abs(i-j), 37-abs(i-j)) for j in range(37)] for i in range(37)]

# ---------- STREAMLIT - estado de sessão ----------
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

# ---------- FUNÇÕES AUXILIARES (corrigidas) ----------
def get_advanced_features(sequence):
    """Retorna 8 features normalizadas com base no seu código, com correções"""
    if len(sequence) < 2:
        return [0.0]*8

    seq = np.array(sequence)
    mean = np.mean(seq)
    std = np.std(seq)
    last = int(sequence[-1])
    second_last = int(sequence[-2])

    # wheel dynamics (posições)
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
    """
    Constrói o vetor de estado para o DQN:
    - flatten dos últimos SEQUENCE_LEN números (normalizados)
    - 8 features
    - top-k probabilidades previstas pela LSTM (se model fornecido)
    """
    seq = list(sequence[-SEQUENCE_LEN:]) if len(sequence) >= SEQUENCE_LEN else list(sequence)
    # pad left with -1 for positions not filled
    pad = [ -1 ] * (SEQUENCE_LEN - len(seq))
    seq_padded = pad + seq
    seq_norm = [(x/36.0 if x>=0 else -1.0) for x in seq_padded]  # -1 for padding

    features = get_advanced_features(sequence[-SEQUENCE_LEN:]) if sequence else [0]*8

    probs = [0.0]*NUM_TOTAL
    if model is not None and len(sequence) >= SEQUENCE_LEN:
        seq_arr = np.array(sequence[-SEQUENCE_LEN:]).reshape(1, SEQUENCE_LEN, 1)
        feat_arr = np.array([features])
        try:
            raw = model.predict([seq_arr, feat_arr], verbose=0)[0]
            probs = raw.tolist()
        except Exception:
            probs = [0.0]*NUM_TOTAL

    state = np.array(seq_norm + features + probs, dtype=np.float32)
    return state

# ---------- LSTM PREDICTOR (corrigido & modular) ----------
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

# ---------- DQN AGENT (com target network) ----------
class DQNAgent:
    def __init__(self, state_size, action_size, lr=DQN_LEARNING_RATE, gamma=DQN_GAMMA, replay_size=REPLAY_SIZE):
        self.state_size = state_size
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
            Dropout(0.3, seed=None),
            Dense(128, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def update_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, use_epsilon=True):
        if use_epsilon and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        return int(np.argmax(q_values))

    def replay(self, batch_size=REPLAY_BATCH):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = np.array([b[0] for b in batch])
        next_states = np.array([b[3] for b in batch])
        q_next = self.target_model.predict(next_states, verbose=0)
        q_curr = self.model.predict(states, verbose=0)

        X = []
        Y = []
        for i, (state, action, reward, next_state, done) in enumerate(batch):
            target = q_curr[i]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(q_next[i])
            X.append(state)
            Y.append(target)
        self.model.fit(np.array(X), np.array(Y), epochs=1, verbose=0)
        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, path):
        self.model.load_weights(path)
        self.update_target()

    def save(self, path):
        self.model.save_weights(path)

# ---------- REWARD FUNCTION ----------
def compute_reward(action_number, outcome_number, bet_amount=BET_AMOUNT):
    """
    Reward net: +35 if win (net profit), -1 if lose.
    This is a simplified model: real casinos pay 35:1 (profit),
    here we define net reward as 35 for win, -1 for loss.
    """
    if action_number == outcome_number:
        return 35.0 * bet_amount
    else:
        return -1.0 * bet_amount

# ---------- PREDIÇÃO E DECISÃO HÍBRIDA ----------
def predict_next_numbers(model, history):
    """Retorna top probabilidades usando LSTM e pós-processamento como antes (corrigido)"""
    if len(history) < SEQUENCE_LEN or model is None:
        return []

    seq = np.array(history[-SEQUENCE_LEN:]).reshape(1, SEQUENCE_LEN, 1)
    feat = np.array([get_advanced_features(history[-SEQUENCE_LEN:])])
    raw_pred = model.predict([seq, feat], verbose=0)[0]

    # temperatura
    temperature = 0.7
    adjusted = np.log(raw_pred + 1e-10) / temperature
    adjusted = np.exp(adjusted)
    adjusted /= adjusted.sum()

    # weighted_pred (similar ao seu)
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
    weighted /= weighted.sum()
    top_indices = list(np.argsort(weighted)[-5:][::-1])
    return [(i, float(weighted[i])) for i in top_indices]

# ---------- STREAMLIT UI ----------
st.set_page_config(layout="centered")
st.title("🔥 ROULETTE AI - LSTM + DQN + HYPER TUNER")

# Input form
with st.form("num_form", clear_on_submit=True):
    num_input = st.number_input("Digite o último número (0-36):", min_value=0, max_value=36, step=1, key="current_number")
    submitted = st.form_submit_button("Enviar")
    if submitted:
        st.session_state.last_input = int(num_input)

if st.session_state.last_input is not None:
    try:
        num = int(st.session_state.last_input)
        st.session_state.history.append(num)
        st.session_state.last_input = None

        # build LSTM model lazily
        if st.session_state.model is None and len(st.session_state.history) >= SEQUENCE_LEN*2:
            st.session_state.model = build_deep_learning_model()

        # incremental training of LSTM (online-ish)
        if st.session_state.model is not None and len(st.session_state.history) > SEQUENCE_LEN*2:
            with st.spinner("Treinando LSTM (curto)..."):
                X_seq, X_feat, y = [], [], []
                for i in range(len(st.session_state.history) - SEQUENCE_LEN - 1):
                    seq = st.session_state.history[i:i+SEQUENCE_LEN]
                    X_seq.append(seq)
                    X_feat.append(get_advanced_features(seq))
                    y.append(st.session_state.history[i+SEQUENCE_LEN])
                X_seq = np.array(X_seq).reshape(-1, SEQUENCE_LEN, 1)
                X_feat = np.array(X_feat)
                y_cat = tf.keras.utils.to_categorical(y, NUM_TOTAL)
                st.session_state.model.fit([X_seq, X_feat], y_cat, epochs=6, batch_size=32, verbose=0,
                                          callbacks=[EarlyStopping(patience=2, restore_best_weights=True),
                                                     ReduceLROnPlateau(factor=0.5, patience=1)])
    except Exception as e:
        st.error("Erro ao processar entrada: " + str(e))

# inicializa DQN se necessário
state_example = sequence_to_state(st.session_state.history, st.session_state.model)
if st.session_state.dqn_agent is None:
    st.session_state.dqn_agent = DQNAgent(state_size=state_example.shape[0], action_size=NUM_TOTAL)

# Quando tivermos dados suficientes, gerar predição e ação do DQN
if len(st.session_state.history) >= SEQUENCE_LEN:
    # predictions (LSTM)
    predictions = predict_next_numbers(st.session_state.model, st.session_state.history)
    st.subheader("🎯 Previsões (LSTM + pós-processamento)")
    if predictions:
        for n, conf in predictions:
            st.write(f"Número: **{n}** — Prob: {conf:.2%}")

    # Construir estado para DQN
    state = sequence_to_state(st.session_state.history, st.session_state.model)
    agent = st.session_state.dqn_agent
    action = agent.act(state)  # número escolhido para apostar
    st.subheader("🤖 Decisão do Agente (DQN)")
    st.write(f"O agente recomenda apostar no número **{action}** (epsilon={agent.epsilon:.3f})")

    # Calcular vizinhos ótimos (simples)
    def optimal_neighbors(number, history, max_neighbors=2):
        if number not in WHEEL_ORDER:
            return []
        idx = WHEEL_ORDER.index(number)
        neigh = []
        for i in range(1, max_neighbors+1):
            neigh.extend([WHEEL_ORDER[(idx-i)%37], WHEEL_ORDER[(idx+i)%37]])
        return list(dict.fromkeys(neigh))  # remove duplicates preserving order

    neighs = optimal_neighbors(action, st.session_state.history, max_neighbors=2)
    st.write("Vizinhos sugeridos:", neighs)

    # Simular a aposta *quando* o próximo número chegar (quando o usuário insere o próximo)
    st.info("Aposte somente se for experimental. O agente escolhe um número e aprende com o resultado real inserido em seguida.")

    # Se houver um novo último número (então já foi adicionado acima), podemos computar recompensa e treinar DQN
    if st.session_state.step_count is None:
        st.session_state.step_count = 0

    # Quando tivermos um novo evento (acabamos de inserir), registramos experiência
    if len(st.session_state.history) >= SEQUENCE_LEN + 1:
        # último outcome
        outcome = st.session_state.history[-1]
        # assumimos que a ação tomada foi a recomendada no passo anterior;
        # para isso guardamos action e state anterior. Simplicidade: recompute previous state as history without last
        prev_history = st.session_state.history[:-1]
        prev_state = sequence_to_state(prev_history, st.session_state.model)
        prev_action = action  # action chosen just before the outcome (approx)
        reward = compute_reward(prev_action, outcome, bet_amount=BET_AMOUNT)
        # next_state (after outcome)
        next_state = sequence_to_state(st.session_state.history, st.session_state.model)
        done = False  # episódio contínuo
        # remember and periodically train
        agent.remember(prev_state, prev_action, reward, next_state, done)
        st.session_state.stats['bets'] += 1
        st.session_state.stats['profit'] += reward
        if reward > 0:
            st.session_state.stats['wins'] += 1
            st.session_state.stats['streak'] += 1
            st.session_state.stats['max_streak'] = max(st.session_state.stats['max_streak'], st.session_state.stats['streak'])
        else:
            st.session_state.stats['streak'] = 0

        st.session_state.step_count += 1
        # train DQN periodically
        if st.session_state.step_count % DQN_TRAIN_EVERY == 0:
            agent.replay(REPLAY_BATCH)
        # update target network periodically
        if st.session_state.step_count % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

# Performance panel
if st.session_state.stats['bets'] > 0:
    st.subheader("📊 Performance")
    profit = st.session_state.stats['profit']
    wins = st.session_state.stats['wins']
    bets = st.session_state.stats['bets']
    winrate = wins / bets if bets > 0 else 0.0
    col1, col2, col3 = st.columns(3)
    col1.metric("Apostas", bets)
    col2.metric("Vitórias", wins, f"{winrate:.1%} taxa de acerto")
    col3.metric("Lucro (simulado)", f"{profit:.2f} unidades")
    st.caption("Recompensa modelada: +35 para acerto (net), -1 para erro (net). Ajuste BET_AMOUNT conforme desejar.")

# Histórico compacto
if st.session_state.history:
    st.subheader("Últimos números")
    st.write(" → ".join(map(str, st.session_state.history[-30:])))

# Otimização / HyperTuner (opcional)
with st.expander("⚙️ Otimização de Hiperparâmetros (Keras Tuner)"):
    if not KERAS_TUNER_AVAILABLE:
        st.write("Keras Tuner não encontrado. Para usar: `pip install keras-tuner`")
    else:
        st.write("Você pode executar uma busca por hiperparâmetros (pesado).")
        max_trials = st.number_input("Max trials", value=12, min_value=2, max_value=200, step=1)
        run_tuner = st.button("Rodar Keras Tuner (RandomSearch)")
        if run_tuner:
            with st.spinner("Executando tuning (pode demorar)..."):
                def build_hypermodel(hp):
                    seq_input = Input(shape=(SEQUENCE_LEN,1))
                    x = LSTM(hp.Int('lstm1', 64, 256, step=64), return_sequences=True)(seq_input)
                    x = Dropout(hp.Float('dropout1', 0.2, 0.6, step=0.1))(x)
                    x = LSTM(hp.Int('lstm2', 32, 128, step=32))(x)

                    feat_input = Input(shape=(8,))
                    y = Dense(hp.Int('dense_feat', 32, 128, step=32), activation='relu')(feat_input)

                    comb = Concatenate()([x,y])
                    z = Dense(hp.Int('dense_comb', 64, 256, step=64), activation='relu')(comb)
                    out = Dense(NUM_TOTAL, activation='softmax')(z)
                    m = Model([seq_input, feat_input], out)
                    lr = hp.Choice('lr', [1e-3, 5e-4, 1e-4])
                    m.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
                    return m

                tuner = kt.RandomSearch(build_hypermodel, objective='val_accuracy',
                                        max_trials=int(max_trials), executions_per_trial=1,
                                        directory='roulette_tuner', project_name=f'run_{int(time.time())}')
                # prepare small validation set from history
                if len(st.session_state.history) < SEQUENCE_LEN*4:
                    st.warning("Histórico insuficiente para tuning. Reúna mais dados.")
                else:
                    # prepare dataset
                    Xs, Xf, Ys = [], [], []
                    for i in range(len(st.session_state.history) - SEQUENCE_LEN - 5):
                        seq = st.session_state.history[i:i+SEQUENCE_LEN]
                        Xs.append(seq)
                        Xf.append(get_advanced_features(seq))
                        Ys.append(st.session_state.history[i+SEQUENCE_LEN])
                    Xs = np.array(Xs).reshape(-1, SEQUENCE_LEN, 1)
                    Xf = np.array(Xf)
                    Yc = tf.keras.utils.to_categorical(Ys, NUM_TOTAL)
                    tuner.search([Xs, Xf], Yc, epochs=6, validation_split=0.15, verbose=0)
                    best = tuner.get_best_models(num_models=1)[0]
                    st.success("Tuning finalizado — modelo encontrado e carregado.")
                    st.session_state.model = best

# Save/Load model (optional)
with st.expander("💾 Salvar / Carregar Modelos"):
    if st.button("Salvar LSTM localmente (lstm_weights.h5)"):
        if st.session_state.model:
            st.session_state.model.save_weights("lstm_weights.h5")
            st.success("LSTM salva em lstm_weights.h5")
        else:
            st.warning("Nenhum modelo LSTM treinado ainda.")
    if st.button("Carregar LSTM (se existir lstm_weights.h5)"):
        try:
            if st.session_state.model is None:
                st.session_state.model = build_deep_learning_model()
            st.session_state.model.load_weights("lstm_weights.h5")
            st.success("Pesos carregados.")
        except Exception as e:
            st.error("Erro ao carregar: " + str(e))

    if st.button("Salvar DQN (dqn_weights.h5)"):
        try:
            st.session_state.dqn_agent.save("dqn_weights.h5")
            st.success("DQN salva em dqn_weights.h5")
        except Exception as e:
            st.error("Erro: " + str(e))
    if st.button("Carregar DQN (dqn_weights.h5)"):
        try:
            st.session_state.dqn_agent.load("dqn_weights.h5")
            st.success("DQN carregada.")
        except Exception as e:
            st.error("Erro: " + str(e))

st.markdown("---")
st.caption("Notas: este é um ambiente de pesquisa. Não há garantias de lucro. Ajuste BET_AMOUNT, políticas de recompensa e parâmetros de treino conforme seu experimento.")


