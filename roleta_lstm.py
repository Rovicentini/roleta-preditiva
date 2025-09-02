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
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from collections import Counter, deque
import random
import logging


import json

def log_dqn_step(episode, step, state, action, reward, q_values, epsilon, loss=None):
    try:
        log_entry = {
            "episode": episode,
            "step": step,
            "action": action,
            "reward": reward,
            "epsilon": round(epsilon, 4),
            "loss": float(loss) if loss is not None else None,
            "q_values_top5": np.argsort(q_values)[-5:].tolist(),
            "state_summary": {
                "mean": float(np.mean(state)),
                "std": float(np.std(state)),
                "max": float(np.max(state)),
                "min": float(np.min(state))
            }
        }
        with open("dqn_log.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        logger.error(f"Erro ao registrar log DQN: {e}")

def avaliar_previsao(apostas, sorteados, stats):
    acertos = len(set(apostas) & set([sorteados]))
    stats['acertos'] += acertos
    stats['rodadas'] += 1

    stats['top1'] += 1 if sorteados in apostas[:1] else 0
    stats['top3'] += 1 if sorteados in apostas[:3] else 0
    stats['top5'] += 1 if sorteados in apostas[:5] else 0

    return stats

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

if 'top_n_metrics' not in st.session_state:
    st.session_state.top_n_metrics = {
        'Top-1': {'hits': 0, 'total': 0},
        'Top-3': {'hits': 0, 'total': 0},
        'Top-5': {'hits': 0, 'total': 0}
    }

# MUDANÇA: Adicionado estado para armazenar estatísticas de normalização
if 'feat_stats' not in st.session_state:
    # Exemplo de valores. SUBSTITUA POR ESTATÍSTICAS REAIS DO SEU HISTÓRICO.
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
REPLAY_BATCH = 256
REPLAY_SIZE = 5000
DQN_TRAIN_EVERY = 5
DQN_LEARNING_RATE = 1e-3
DQN_GAMMA = 0.40

# Recompensa (shaping)
REWARD_EXACT = 35.0
REWARD_NEIGHBOR = 5.0
REWARD_COLOR = 0.0
REWARD_DOZEN = 0.0
REWARD_LOSS = -15.0
NEIGHBOR_RADIUS_FOR_REWARD = 3

# Treino LSTM incremental
LSTM_RECENT_WINDOWS = 100
# CORREÇÃO: não referenciar variável inexistente 'history'
LSTM_BATCH_SAMPLES = 256
LSTM_EPOCHS_PER_STEP = 5
LSTM_BATCH_SIZE = 32

# Hiperparâmetros DQN (exploração)
def get_training_params(history_length):
    if history_length < 100:
        return 10, 16
    elif history_length < 500:
        return 15, 32
    elif history_length < 1000:
        return 20, 64
    elif history_length < 5000:
        return 25, 128
    else:
        return 30, 256

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.97
# Limiar de confiança para sugerir aposta
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
    "voisins_du_zero": {0, 2, 3, 4, 7, 12, 15, 18, 19, 21, 22, 25, 26, 28, 29, 32, 35, 36}, # 17 números
    "tiers_du_cylindre": {27, 13, 36, 11, 30, 8, 23, 10, 5, 24, 16, 33}, # 12 números
    "orphelins": {1, 6, 9, 14, 17, 20, 31, 34}, # 8 números
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

# Flag para alternar entre LSTM puro e híbrido
USE_LSTM_ONLY = True  # mude para False para voltar a usar DQN

def sequence_to_state(sequence, model=None, feat_means=None, feat_stds=None):
    seq_slice = sequence[-SEQUENCE_LEN:] if sequence else []
    
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
            # CORREÇÃO: o modelo agora tem 6 saídas
            if isinstance(raw, list) and len(raw) >= 3:
                num_probs = np.array(raw[0][0])
                entropy = -np.sum(num_probs * np.log(num_probs + 1e-9))
                entropy_norm = entropy / np.log(len(num_probs))
                entropy_vector = np.array([entropy_norm])
                color_probs = np.array(raw[1][0])
                dozen_probs = np.array(raw[2][0])
            else:
                num_probs = np.array(raw)[0]
        except Exception:
            pass
        entropy = -np.sum(num_probs * np.log(num_probs + 1e-9))
        entropy_norm = entropy / np.log(len(num_probs))
        entropy_vector = np.array([entropy_norm])

    age_vector = [0] * NUM_TOTAL
    last_seen = {num: i for i, num in enumerate(sequence)}
    for num in range(NUM_TOTAL):
        if num in last_seen:
            age_vector[num] = len(sequence) - 1 - last_seen[num]
        else:
            age_vector[num] = len(sequence)
    max_age = max(age_vector) if age_vector else 1
    age_vector = np.array([age / max(1, max_age) for age in age_vector])

    # Streaks
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

    run_length_features = np.array([
        last_run_len_num / SEQUENCE_LEN,
        last_run_len_color / SEQUENCE_LEN,
        last_run_len_dozen / SEQUENCE_LEN
    ])
    
    last_num = sequence[-1] if sequence else -1
    last_color_one_hot = to_categorical(number_to_color(last_num), 3) if last_num in range(NUM_TOTAL) else np.zeros(3)
    last_dozen_one_hot = to_categorical(number_to_dozen(last_num), 4) if last_num in range(NUM_TOTAL) else np.zeros(4)
    
    recent_seq = seq_slice
    # Frequência relativa dos últimos 100 números
    freq_counter = np.zeros(NUM_TOTAL)
    freq_window = sequence[-100:] if len(sequence) >= 100 else sequence
    for num in freq_window:
        freq_counter[num] += 1
    freq_vector = freq_counter / max(1, np.sum(freq_counter))
    # Tendência por cor (vermelho, preto, verde)
    color_counts = [0, 0, 0]  # vermelho, preto, verde
    for num in freq_window:
        color = number_to_color(num)
        if color in [0, 1, 2]:
            color_counts[color] += 1
    color_vector = np.array(color_counts) / max(1, len(freq_window))
    region_counts_freq = np.zeros(3)
    for num in freq_window:
        dozen = number_to_dozen(num)
        if dozen in [0, 1, 2]:  # ignorando 0
             region_counts_freq[dozen] += 1
    region_vector = region_counts_freq / max(1, np.sum(region_counts_freq))

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
    
    # 🔹 num_probs vem primeiro
    state = np.concatenate([
        num_probs,
        one_hot_seq.flatten(),
        np.array(features),
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
        last_num_pulling_strength,
        freq_vector,
        color_vector,
        region_vector,
        entropy_vector,
    ]).astype(np.float32)

    return state

def filtrar_apostas_por_confianca(probabilidades, q_values, freq_vector, limiar=0.6):
    score = (
        0.5 * np.array(probabilidades) +
        0.3 * np.array(q_values) +
        0.2 * np.array(freq_vector)
    )
    indices = np.where(score >= limiar)[0]
    return indices.tolist()

# =========================
# MODELO LSTM – ARQUITETURA REFINADA
# =========================
def build_deep_learning_model(seq_len=SEQUENCE_LEN, num_total=NUM_TOTAL):
    seq_input = Input(shape=(seq_len, num_total), name='sequence_input')

    # Bidirecional + BN + Dropout
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(1e-4)))(seq_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # LSTM bidirecional
    x = Bidirectional(LSTM(96, return_sequences=True, kernel_regularizer=l2(1e-4)))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

# CNN para padrões locais
    from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
    cnn_out = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    cnn_out = GlobalMaxPooling1D()(cnn_out)

    
    x_att = Attention(name="self_attention")([x, x])
    x_att = LSTM(64, return_sequences=False)(x_att)

# Combina CNN + LSTM final
    x = Concatenate()([cnn_out, x_att])

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
    out_neighbors = Dense(num_total, activation='softmax', name='neighbors_out')(dense)
    out_regions = Dense(len(REGIONS), activation='softmax', name='regions_out')(dense)
    out_even_odd_high_low = Dense(4, activation='softmax', name='eohl_out')(dense)

    model = Model(inputs=[seq_input, feat_input],
                  outputs=[out_num, out_color, out_dozen, out_neighbors, out_regions, out_even_odd_high_low])
    
    optimizer = Nadam(learning_rate=6e-4)
    model.compile(optimizer=optimizer,
                  loss={'num_out': 'categorical_crossentropy',
                        'color_out': 'categorical_crossentropy',
                        'dozen_out': 'categorical_crossentropy',
                        'neighbors_out': 'categorical_crossentropy',
                        'regions_out': 'categorical_crossentropy',
                        'eohl_out': 'categorical_crossentropy'},
                  loss_weights={'num_out': 1.5, 'color_out': 0.25, 'dozen_out': 0.25, 'neighbors_out': 0.7, 'regions_out': 0.3, 'eohl_out': 0.2},
                  metrics={'num_out': 'accuracy'})

    return model

# --- PREDICTION POSTPROCESSING ---
def predict_next_numbers(model, history, top_k=3):
    if history is None or len(history) < SEQUENCE_LEN or model is None:
        return []
    try:
        feat = np.array([get_advanced_features(history[-SEQUENCE_LEN:],
                                                st.session_state.feat_stats['means'],
                                                st.session_state.feat_stats['stds'])])
        seq_one_hot = sequence_to_one_hot(history).reshape(1, SEQUENCE_LEN, NUM_TOTAL)
        raw = model.predict([seq_one_hot, feat], verbose=0)
        
        # O modelo agora retorna 6 saídas
        if isinstance(raw, list) and len(raw) >= 3:
            num_probs = np.array(raw[0][0])
            color_probs = np.array(raw[1][0])
            dozen_probs = np.array(raw[2][0])
        else:
            num_probs = np.array(raw)[0]

        # ✅ Cálculo da entropia com base em num_probs
        entropy = -np.sum(num_probs * np.log(num_probs + 1e-9))
        entropy_norm = entropy / np.log(len(num_probs))
        entropy_vector = np.array([entropy_norm])

    except Exception as e:
        logger.error(f"Erro na previsão LSTM: {e}")
        return []


    temperature = 0.25
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
        momentum_factor = 1 + momentum*0.5
        
        neighbor_factor = 1 + neighbors_probs[WHEEL_ORDER.index(num)] * 3
        
        # Fatores de região e Even/Odd/High/Low
        region_factor = 1.0
        eohl_factor = 1.0
        
        if len(history) > 0:
            current_region_idx = number_to_region(num)
            if current_region_idx != -1:
                region_factor = 1 + regions_probs[current_region_idx] * 1.5

            if num % 2 == 0 and num != 0: # par
                eohl_factor = 1 + eohl_probs[0] * 1.5
            elif num % 2 != 0: # ímpar
                eohl_factor = 1 + eohl_probs[1] * 1.5
            if 19 <= num <= 36: # alto
                eohl_factor *= (1 + eohl_probs[2] * 1.5)
            elif 1 <= num <= 18: # baixo
                eohl_factor *= (1 + eohl_probs[3] * 1.5)

        weighted.append(adjusted[num] * freq_factor * distance_factor * momentum_factor * neighbor_factor * region_factor * eohl_factor)

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
        'neighbors_probs': neighbors_probs,
        'regions_probs': regions_probs,
        'eohl_probs': eohl_probs,
        'color_pred': color_pred,
        'dozen_pred': dozen_pred
    }

def calculate_top_n_accuracy(predictions, actual_number, top_n_values=[1, 3, 5]):
    """
    Calcula a Top-N Accuracy com base nas previsões e no número sorteado.

    Args:
        predictions (list): Lista de tuplas (número, probabilidade) ordenadas.
        actual_number (int): O número sorteado.
        top_n_values (list): Lista de valores N para calcular a acurácia.

    Returns:
        dict: {'Top-1': True/False, 'Top-3': True/False, ...}
    """
    sorted_numbers = [num for num, _ in predictions]
    accuracy_results = {}
    for n in top_n_values:
        top_n_numbers = sorted_numbers[:n]
        accuracy_results[f'Top-{n}'] = actual_number in top_n_numbers
    return accuracy_results

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
        state_input = Input(shape=(self.state_size,))
        x = Dense(512)(state_input)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)

        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)


        value_stream = Dense(128)(x)
        value_stream = LeakyReLU(alpha=0.1)(value_stream)
        value_stream = Dense(1, activation='linear', name='value_out')(value_stream)

        advantage_stream = Dense(128)(x)
        advantage_stream = LeakyReLU(alpha=0.1)(advantage_stream)
        advantage_stream = Dense(self.action_size, activation='linear', name='advantage_out')(advantage_stream)

        q_values = Lambda(lambda x: x[0] + (x[1] - K.mean(x[1], axis=1, keepdims=True)),
                          output_shape=(self.action_size,))([value_stream, advantage_stream])

        model = Model(inputs=state_input, outputs=q_values)
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
            top_k_actions = []
            sorted_indices = np.argsort(q_values)[::-1]
            for idx in sorted_indices:
                if q_values[idx] > CONFIDENCE_THRESHOLD:
                    top_k_actions.append(int(idx))
                if len(top_k_actions) >= k:
                    break
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
            next_q_target = self.model.predict(next_states, verbose=0)
            next_actions = np.argmax(next_q_target, axis=1)
            next_q_values = self.target_model.predict(next_states, verbose=0)
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
                    target[action] = reward + self.gamma * next_q_values[i][next_actions[i]]
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

def filter_actions_by_region(actions, max_neighbors=NEIGHBOR_RADIUS_FOR_REWARD):
    """
    Filtra as ações sugeridas para garantir uma cobertura diversificada de regiões.
    """
    selected_actions = []
    covered_regions = set()
    for action in actions:
        action_neighbors = [action] + optimal_neighbors(action, max_neighbors=max_neighbors)
        is_covered = any(num in covered_regions for num in action_neighbors)
        if not is_covered:
            selected_actions.append(action)
            covered_regions.update(action_neighbors)
        if len(selected_actions) >= 3:
            break
    return selected_actions if selected_actions else actions[:3]

# =========================
# RECOMPENSA FOCADA E SIMPLIFICADA
# =========================
def compute_reward(action_numbers, outcome_number, bet_amount=BET_AMOUNT,
                   max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD):
    action_numbers = set([a for a in action_numbers if 0 <= a <= 36])
    acertos_exatos = 1 if outcome_number in action_numbers else 0

    # Penalidade por quantidade de apostas
    bet_penalty = 0.1 * len(action_numbers)

    # Acerto vizinho
    all_neighbors = set()
    for a in action_numbers:
        all_neighbors.update(optimal_neighbors(a, max_neighbors=max_neighbors_for_reward))
    acerto_vizinho = 1 if outcome_number in all_neighbors else 0

    # Recompensa proporcional
    reward = 0.0
    reward += acertos_exatos * REWARD_EXACT
    reward += acerto_vizinho * REWARD_NEIGHBOR

    # Penalidade por apostas amplas
    if len(action_numbers) > 10:
        reward -= 1.0

    # Bônus por múltiplos acertos
    if acertos_exatos + acerto_vizinho >= 3:
        reward += 2.0

    # Bônus por streak
    if 'stats' in st.session_state and st.session_state.stats.get('streak', 0) >= 3:
        reward += 1.0

    # Penalidade proporcional ao número de apostas
    reward -= bet_penalty

    return reward * bet_amount


# =========================
# LSTM: construção de dataset e treino recente
# =========================
def build_lstm_supervised_from_history(history):
    X_seq, X_feat, y_num, y_color, y_dozen, y_neighbors, y_regions, y_eohl = [], [], [], [], [], [], [], []
    if len(history) <= SEQUENCE_LEN:
        return None

    start_idx = max(0, len(history) - (SEQUENCE_LEN + 1) - LSTM_RECENT_WINDOWS)
    for i in range(start_idx, len(history) - SEQUENCE_LEN - 1):
        seq_slice = history[i:i+SEQUENCE_LEN]
        target = history[i+SEQUENCE_LEN]

        X_seq.append(sequence_to_one_hot(seq_slice))
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
        
        # Target: Vizinhos
        y_neighbors_data = np.zeros(NUM_TOTAL)
        if target in WHEEL_ORDER:
            target_neighbors = optimal_neighbors(target, max_neighbors=NEIGHBOR_RADIUS_FOR_REWARD)
            for neighbor in target_neighbors:
                if neighbor in WHEEL_ORDER:
                    pos = WHEEL_ORDER.index(neighbor)
                    y_neighbors_data[pos] = 1.0 / len(target_neighbors)
        y_neighbors.append(y_neighbors_data)

        # Target: Regiões
        region_label = number_to_region(target)
        y_regions.append(to_categorical(region_label, len(REGIONS)) if region_label != -1 else np.zeros(len(REGIONS)))

        # Target: Par/Ímpar/Alto/Baixo (+ zero como classe 3)
        eohl_label = 3 if target == 0 else (0 if target % 2 == 0 else 1)
        y_eohl.append(to_categorical(eohl_label, 4))

    if len(X_seq) == 0:
        return None

    X_seq = np.array(X_seq)
    X_feat = np.array(X_feat)
    y_num = np.array(y_num)
    y_color = np.array(y_color)
    y_dozen = np.array(y_dozen)
    y_neighbors = np.array(y_neighbors)
    y_regions = np.array(y_regions)
    y_eohl = np.array(y_eohl)
    
    return X_seq, X_feat, y_num, y_color, y_dozen, y_neighbors, y_regions, y_eohl

def train_lstm_on_recent_minibatch(model, history):
    data = build_lstm_supervised_from_history(history)
    if data is None:
        return
    
    X_seq, X_feat, y_num, y_color, y_dozen, y_neighbors, y_regions, y_eohl = data
    n = len(X_seq)
    if n == 0:
        return

    k = min(n, LSTM_BATCH_SAMPLES)
    idx = np.random.choice(n, k, replace=False)
    try:
        model.fit([X_seq[idx], X_feat[idx]],
                  [y_num[idx], y_color[idx], y_dozen[idx], y_neighbors[idx], y_regions[idx], y_eohl[idx]],
                  epochs=LSTM_EPOCHS_PER_STEP,
                  batch_size=LSTM_BATCH_SIZE,
                  verbose=0)
        logger.info(f"LSTM mini-train: {k} amostras de {n} ({LSTM_EPOCHS_PER_STEP} épocas).")
    except Exception as e:
        logger.error(f"Erro no treinamento LSTM: {e}")
def train_lstm_on_full_history(model, history, epochs=20, batch_size=64):
    """
    Treina o LSTM usando TODO o histórico disponível (batch offline).
    Use epochs e batch_size maiores que no treino incremental.
    """
    data = build_lstm_supervised_from_history(history)
    if data is None:
        st.warning("Histórico insuficiente para treino em massa.")
        return

    X_seq, X_feat, y_num, y_color, y_dozen, y_neighbors, y_regions, y_eohl = data

    with st.spinner(f"Treinando LSTM no histórico completo ({len(X_seq)} amostras)..."):
        try:
            epochs, batch_size = get_training_params(len(history))
            model.fit([X_seq, X_feat],
                      [y_num, y_color, y_dozen, y_neighbors, y_regions, y_eohl],
                      epochs=epochs,
                      batch_size=batch_size,
                      verbose=0)
            st.success(f"LSTM treinado com {len(X_seq)} amostras (epochs={epochs}, batch_size={batch_size}).")
        except Exception as e:
            st.error(f"Erro no treino em massa do LSTM: {e}")




def preload_dqn_with_history(agent, history, model, top_k=3):
    """
    Preenche a memória de replay do DQN usando o histórico:
    - Para cada estado, usa as previsões do LSTM (top_k) como "ações tomadas"
    - Calcula a recompensa contra o próximo número real
    - Armazena (state, actions, reward, next_state) no replay
    """
    if agent is None or model is None or len(history) <= SEQUENCE_LEN:
        return

    count = 0
    with st.spinner("Pré-carregando memória do DQN a partir do histórico..."):
        for i in range(SEQUENCE_LEN, len(history)):
            past = history[:i]
            # Estado atual
            state = sequence_to_state(past, model,
                                      st.session_state.feat_stats['means'],
                                      st.session_state.feat_stats['stds'])
            # Previsões do LSTM para esse estado
            pred = predict_next_numbers(model, past, top_k=top_k)
            if pred and 'top_numbers' in pred and pred['top_numbers']:
                actions = [n for n, _ in pred['top_numbers']]
            else:
                actions = [random.randrange(NUM_TOTAL)]

            # Próximo estado e outcome observado
            next_state = sequence_to_state(history[:i+1], model,
                                           st.session_state.feat_stats['means'],
                                           st.session_state.feat_stats['stds'])
            outcome = history[i]
            reward = compute_reward(actions, outcome,
                                    bet_amount=BET_AMOUNT,
                                    max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD)

            agent.remember(state, actions, reward, next_state, False)
            count += 1

    st.success(f"Replay do DQN pré-carregado com {count} transições.")


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
            # === Pré-treino offline após carga em massa ===
# 1) Garante que o modelo exista
            if st.session_state.model is None and len(st.session_state.history) >= SEQUENCE_LEN * 2:
                st.session_state.model = build_deep_learning_model()

# 2) Treina o LSTM em TODO o histórico (batch offline)
            if st.session_state.model is not None and len(st.session_state.history) > SEQUENCE_LEN * 2:
                train_lstm_on_full_history(
                    st.session_state.model,
                    st.session_state.history
                )

# 3) Inicializa o DQN (se necessário)
            exemplo_estado = sequence_to_state(
                st.session_state.history,
                st.session_state.model,
                st.session_state.feat_stats['means'],
                st.session_state.feat_stats['stds']
            )
            if st.session_state.dqn_agent is None and exemplo_estado is not None:
                st.session_state.dqn_agent = DQNAgent(state_size=exemplo_estado.shape[0], action_size=NUM_TOTAL)

# 4) Pré-carrega replay do DQN com pares (state, actions=LSTM_topk, reward, next_state)
            if st.session_state.dqn_agent is not None and st.session_state.model is not None:
                preload_dqn_with_history(
                    st.session_state.dqn_agent,
                    st.session_state.history,
                    st.session_state.model,
                    top_k=3
                )
    # 5) Executa algumas iterações de treino em cima do replay carregado
                with st.spinner("Executando treino inicial do DQN..."):
                    for _ in range(40):   # ajuste fino: 20–100
                        st.session_state.dqn_agent.replay(REPLAY_BATCH)
                    st.session_state.dqn_agent.update_target()
                st.success("DQN pré-treinado com o histórico.")
# Opcional: Atualiza prev_state para próxima decisão já usar o modelo afinado
            st.session_state.prev_state = sequence_to_state(
                st.session_state.history,
                st.session_state.model,
                st.session_state.feat_stats['means'],
                st.session_state.feat_stats['stds']
            )

            st.session_state.clear_input_bulk = True
            st.rerun()
        except Exception as e:
            st.error(f"Erro ao processar números: {e}")
    else:
        st.warning("Insira números válidos para adicionar.")

st.markdown("---")

# Formulário para entrada de um único número (último resultado)
with st.form("num_form", clear_on_submit=True):
    num_input = st.number_input("Digite o último número (0-36):", min_value=0, max_value=36, step=1, key="current_number")
    submitted = st.form_submit_button("Enviar")
    if submitted:
        st.session_state.last_input = int(num_input)

# Processa quando o usuário informa o último número
if st.session_state.last_input is not None:
    try:
        num = int(st.session_state.last_input)

        # Checa acurácia das previsões anteriores (Top-N)
        if 'lstm_predictions' in st.session_state and st.session_state.lstm_predictions:
            acuracias = calculate_top_n_accuracy(st.session_state.lstm_predictions, num, top_n_values=[1, 3, 5])
            for metrica, acertou in acuracias.items():
                st.session_state.top_n_metrics[metrica]['total'] += 1
                if acertou:
                    st.session_state.top_n_metrics[metrica]['hits'] += 1
            st.session_state.lstm_predictions = None

        st.session_state.history.append(num)
        st.session_state.co_occurrence_matrix = update_co_occurrence_matrix(
            st.session_state.co_occurrence_matrix,
            st.session_state.history
        )

        # ✅ Avalia a previsão da rodada
        st.session_state.stats = avaliar_previsao(apostas_final, num, st.session_state.stats)

    except Exception as e:
        logger.error(f"Erro ao processar entrada: {e}")

    # ✅ Exibe painel de métricas (fora do try, mas dentro do if)
    st.markdown("### 📊 Métricas de Acurácia")
    st.write(f"Total de rodadas: {st.session_state.stats['rodadas']}")
    st.write(f"Total de acertos: {st.session_state.stats['acertos']}")
    st.write(f"Top‑1: {st.session_state.stats['top1']} acertos")
    st.write(f"Top‑3: {st.session_state.stats['top3']} acertos")
    st.write(f"Top‑5: {st.session_state.stats['top5']} acertos")

        # Inicializa DQN se ainda não existir
        if st.session_state.dqn_agent is None and len(st.session_state.history) >= SEQUENCE_LEN:
            exemplo_estado = sequence_to_state(
                st.session_state.history,
                st.session_state.model,
                st.session_state.feat_stats['means'],
                st.session_state.feat_stats['stds']
            )
            if exemplo_estado is not None:
                st.session_state.dqn_agent = DQNAgent(
                    state_size=exemplo_estado.shape[0],
                    action_size=NUM_TOTAL
            )


        # Reforço com resultado anterior
        if st.session_state.dqn_agent is None and len(st.session_state.history) >= SEQUENCE_LEN:
    exemplo_estado = sequence_to_state(
        st.session_state.history,
        st.session_state.model,
        st.session_state.feat_stats['means'],
        st.session_state.feat_stats['stds']
    )
    if exemplo_estado is not None:
        st.session_state.dqn_agent = DQNAgent(
            state_size=exemplo_estado.shape[0],
            action_size=NUM_TOTAL
        )

# Reforço com resultado anterior
if st.session_state.prev_state is not None and st.session_state.prev_actions is not None:
    recompensa = compute_reward(
        st.session_state.prev_actions,
        num,
        bet_amount=BET_AMOUNT,
        max_neighbors_for_reward=NEIGHBOR_RADIUS_FOR_REWARD
    )

    proximo_estado = sequence_to_state(
        st.session_state.history,
        st.session_state.model,
        st.session_state.feat_stats['means'],
        st.session_state.feat_stats['stds']
    )

    if st.session_state.dqn_agent is not None:
        st.session_state.dqn_agent.remember(
            st.session_state.prev_state,
            st.session_state.prev_actions,
            recompensa,
            proximo_estado,
            False
        )

    if st.session_state.dqn_agent is not None:
        try:
            q_vals = st.session_state.dqn_agent.model.predict(
                np.array([st.session_state.prev_state]),
                verbose=0
            )[0]

            log_dqn_step(
                episode=1,
                step=st.session_state.step_count,
                state=st.session_state.prev_state,
                action=st.session_state.prev_actions,
                reward=recompensa,
                q_values=q_vals,
                epsilon=st.session_state.dqn_agent.epsilon
            )
        except Exception as e:
            logger.error(f"Erro ao logar passo DQN: {e}")

        # ✅ Aplicando filtro de apostas por confiança combinada
        apostas_final = filtrar_apostas_por_confianca(num_probs, q_vals, freq_vector)


            # Atualiza estatísticas
            st.session_state.stats['bets'] += 1
            st.session_state.stats['profit'] += recompensa
            if recompensa > 0:
                st.session_state.stats['wins'] += 1
                st.session_state.stats['streak'] += 1
                st.session_state.stats['max_streak'] = max(st.session_state.stats['max_streak'], st.session_state.stats['streak'])
            else:
                st.session_state.stats['streak'] = 0

            st.session_state.step_count += 1
            if st.session_state.dqn_agent is not None:
                st.session_state.dqn_agent.replay(REPLAY_BATCH)
            if st.session_state.dqn_agent is not None and st.session_state.step_count % TARGET_UPDATE_FREQ == 0:
                st.session_state.dqn_agent.update_target()

        # Inicializa e treina LSTM se já houver dados suficientes
        if st.session_state.model is None and len(st.session_state.history) >= SEQUENCE_LEN * 2:
            st.session_state.model = build_deep_learning_model()
        if st.session_state.model is not None and len(st.session_state.history) > SEQUENCE_LEN * 2:
            train_lstm_on_recent_minibatch(st.session_state.model, st.session_state.history)
            st.session_state.prev_state = sequence_to_state(st.session_state.history, st.session_state.model,
                                                            st.session_state.feat_stats['means'],
                                                            st.session_state.feat_stats['stds'])

            pred_info = predict_next_numbers(st.session_state.model, st.session_state.history, top_k=3)
            if pred_info and 'top_numbers' in pred_info:
                st.session_state.lstm_predictions = pred_info['top_numbers']

                st.subheader("🎯 Previsões LSTM")
                for n, conf in pred_info['top_numbers']:
                    st.write(f"Número: **{n}** — Probabilidade: {conf:.2%}")

        # Define ações sugeridas
        if USE_LSTM_ONLY and st.session_state.model is not None:
            pred_info = predict_next_numbers(st.session_state.model, st.session_state.history, top_k=3)
            acoes_sugeridas = [n for n, _ in pred_info['top_numbers']] if pred_info else random.sample(range(NUM_TOTAL), 3)
        elif st.session_state.dqn_agent is not None and st.session_state.prev_state is not None:
            possiveis_acoes = st.session_state.dqn_agent.act_top_k(st.session_state.prev_state, k=5, use_epsilon=True)
            acoes_sugeridas = filter_actions_by_region(possiveis_acoes, max_neighbors=NEIGHBOR_RADIUS_FOR_REWARD)
        else:
            acoes_sugeridas = random.sample(range(NUM_TOTAL), 3)

        st.session_state.prev_actions = acoes_sugeridas

        st.subheader("🤖 Ações sugeridas")
        for acao in acoes_sugeridas:
            vizinhos = optimal_neighbors(acao, max_neighbors=NEIGHBOR_RADIUS_FOR_REWARD)
            st.write(f"- Apostar no {acao} (vizinhos: {', '.join(map(str, vizinhos))})")

    except Exception as e:
        st.error(f"Erro inesperado: {e}")

st.markdown("---")
st.subheader("📊 Estatísticas")
if st.session_state.stats['bets'] > 0:
    taxa_vitoria = (st.session_state.stats['wins'] / st.session_state.stats['bets']) * 100
    st.write(f"Apostas: {st.session_state.stats['bets']} | Vitórias: {st.session_state.stats['wins']} | Taxa de Vitória: {taxa_vitoria:.2f}%")
    lucro_color = "green" if st.session_state.stats['profit'] >= 0 else "red"
    st.markdown(f"Lucro total: <span style='color:{lucro_color}'>${st.session_state.stats['profit']:.2f}</span>", unsafe_allow_html=True)
else:
    st.write("Nenhuma estatística disponível ainda.")

# Histórico
st.subheader("📜 Histórico de números")
st.write(", ".join(map(str, st.session_state.history[::-1])))

# Acurácia Top-N
st.subheader("🎯 Acurácia LSTM")
for metrica, dados in st.session_state.top_n_metrics.items():
    if dados['total'] > 0:
        acuracia = (dados['hits'] / dados['total']) * 100
        st.metric(label=metrica, value=f"{acuracia:.2f}%", help=f"Baseado em {dados['total']} previsões.")
    else:
        st.metric(label=metrica, value="N/A")
















