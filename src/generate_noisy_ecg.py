import os
import numpy as np
import wfdb
import random
import json
import csv
from datetime import datetime
from scipy import signal
from tqdm import tqdm
try:
    # esecuzione da src/
    from load_ecg import MIT_BIH_DIR, NOISE_DIR, SEGMENTED_SIGNAL_DIR, create_directories, get_train_test_split
except ImportError:
    # esecuzione da root del repo
    from src.load_ecg import MIT_BIH_DIR, NOISE_DIR, SEGMENTED_SIGNAL_DIR, create_directories, get_train_test_split

# --- General Configurations ---
SEGMENT_LENGTH = 1024
OVERLAP_LENGTH = 512

# Logging and reproducibility
LOG_PARAMETERS = True
METADATA_DIR = os.path.join(SEGMENTED_SIGNAL_DIR, 'metadata')
RANDOM_SEED_BASE = 42

# Noise generation configs
TARGET_SNR_RANGE = (5, 25)  # dB
USE_SNR_TARGET = True

NOISE_COMBINATIONS = [
    ('BW',), ('MA',), ('PLI',),
    ('BW', 'MA'), ('BW', 'PLI'), ('MA', 'PLI'),
    ('BW', 'MA', 'PLI')
]
COMBINATION_WEIGHTS = [1.0, 1.0, 1.0, 1.2, 1.2, 1.2, 0.8]

PLI_FREQUENCIES = [50, 60]
PLI_HARMONICS_PROB = 0.3
PLI_3RD_HARMONIC_AMP = 0.15
PLI_5TH_HARMONIC_AMP = 0.08

MA_FREQ_RANGE = (20, 100)
MA_BURST_PROB = 0.4

BW_FREQ_RANGE = (0.1, 1.0)

# Map dei tipi di rumore ai record dell'NSTDB (Noise Stress Test DB)
# BW -> "bw", MA -> "ma". PLI non è presente nel DB e resta sintetico.
NOISE_RECORD_MAP = {
    'BW': 'bw',
    'MA': 'ma',
}

create_directories()
os.makedirs(METADATA_DIR, exist_ok=True)

print(f"INFO: Using ECG_DIR: {MIT_BIH_DIR}")
print(f"INFO: Using NOISE_DIR: {NOISE_DIR}")
print(f"INFO: Generated Segment Output Directory: {SEGMENTED_SIGNAL_DIR}")
print(f"INFO: SNR-based scaling: {USE_SNR_TARGET}")
print(f"INFO: Target SNR range: {TARGET_SNR_RANGE} dB")
print(f"INFO: Noise combinations: {len(NOISE_COMBINATIONS)} types")

# --- Helpers ---

def load_ecg_full(record_name):
    record_path = os.path.join(MIT_BIH_DIR, record_name)
    try:
        record = wfdb.rdrecord(record_path)
        ecg_signal_full = record.p_signal[:, 0]
        fs = int(record.fs)
        if len(ecg_signal_full) == 0:
            raise ValueError("ECG signal is empty.")
        return ecg_signal_full.astype(np.float32), fs
    except Exception as e:
        print(f"❌ Error loading record {record_name}: {e}")
        return None, None

def load_noise_realistically(noise_name, target_length, seed=None):
    """Carica un rumore reale dall'NSTDB e restituisce un segmento di lunghezza target.

    Ritorna:
        (segmento_noise_float32, params_dict)
    params_dict include: type, source, start, length.
    """
    noise_path = os.path.join(NOISE_DIR, noise_name)
    try:
        noise_record = wfdb.rdrecord(noise_path)
        full_noise_signal = noise_record.p_signal[:, 0]
    except Exception as e:
        # Se non esistono i record di rumore puro (bw/ma/em), prova a usare i paired noisy ECG (118x/119x) per derivare il rumore
        if noise_name in ('bw', 'ma', 'em'):
            derived, params = derive_noise_from_paired_examples(noise_name, target_length, seed)
            if derived is not None:
                return derived, params
        print(f"❌ Error loading noise {noise_name}: {e}")
        return np.zeros(target_length, dtype=np.float32), {
            'type': noise_name,
            'source': 'NSTDB',
            'error': str(e),
            'start': 0,
            'length': int(target_length),
        }
    current_len = len(full_noise_signal)
    if current_len == 0:
        return np.zeros(target_length, dtype=np.float32), {
            'type': noise_name,
            'source': 'NSTDB',
            'error': 'empty_noise',
            'start': 0,
            'length': int(target_length),
        }

    # Gestione finestra con seed per riproducibilità
    rng = np.random.default_rng(seed) if seed is not None else np.random
    if current_len < target_length:
        # Ripeti e taglia
        tiled = np.tile(full_noise_signal, int(np.ceil(target_length / current_len)))[:target_length]
        start_index = 0
        seg = tiled.astype(np.float32)
    elif current_len > target_length:
        start_index = int(rng.integers(0, current_len - target_length + 1))
        seg = full_noise_signal[start_index:start_index + target_length].astype(np.float32)
    else:
        start_index = 0
        seg = full_noise_signal.astype(np.float32)

    params = {
        'type': str(noise_name),
        'source': 'NSTDB',
        'start': int(start_index),
        'length': int(target_length),
    }
    return seg, params


def list_paired_noisy_bases(noise_letter: str):
    """Cerca in NOISE_DIR basi tipo 118e.. / 119e.. (o m/b) e restituisce la lista dei path base senza estensione.
    noise_letter in {'e','m','b'}.
    """
    bases = []
    for root, _, files in os.walk(NOISE_DIR):
        for f in files:
            if f.endswith('.hea'):
                base = os.path.splitext(f)[0]
                if len(base) >= 4 and base[:3] in ('118', '119') and base[3].lower() == noise_letter.lower():
                    bases.append(os.path.join(root, base))
    bases.sort()
    return bases


def derive_noise_from_paired_examples(noise_name: str, target_length: int, seed=None):
    """Deriva un segmento di rumore sottraendo il clean dal corrispondente ECG rumoroso (NSTDB 118*/119*).

    noise_name: 'bw'|'ma'|'em' mappati a lettere 'b'|'m'|'e'.
    Ritorna (noise_seg, params) oppure (None, params) se non disponibile.
    """
    letter_map = {'bw': 'b', 'ma': 'm', 'em': 'e'}
    letter = letter_map.get(noise_name.lower())
    if letter is None:
        return None, {'error': f'unsupported noise_name {noise_name}'}

    candidates = list_paired_noisy_bases(letter)
    if not candidates:
        return None, {'error': f'no paired examples for letter {letter} in {NOISE_DIR}'}

    rng = np.random.default_rng(seed) if seed is not None else np.random
    base_path = candidates[int(rng.integers(0, len(candidates)))]

    try:
        noisy_sig, noisy_fields = wfdb.rdsamp(base_path)
        noisy = noisy_sig[:, 0].astype(np.float32)
        # clean corrispondente: record 118 o 119 in MIT_BIH_DIR
        rec_id = os.path.basename(base_path)[:3]
        clean_base = os.path.join(MIT_BIH_DIR, rec_id)
        clean_sig, clean_fields = wfdb.rdsamp(clean_base)
        clean = clean_sig[:, 0].astype(np.float32)
        # Allinea per lunghezza minima
        min_len = min(len(noisy), len(clean))
        if min_len < target_length:
            return None, {'error': f'paired length too short ({min_len}) for target {target_length}', 'base': base_path}
        max_start = min_len - target_length
        start = int(rng.integers(0, max_start + 1))
        noise_seg = (noisy[start:start+target_length] - clean[start:start+target_length]).astype(np.float32)
        return noise_seg, {
            'type': noise_name,
            'source': 'NSTDB_derived',
            'paired_base': os.path.basename(base_path),
            'clean_record': rec_id,
            'start': int(start),
            'length': int(target_length),
        }
    except Exception as e:
        return None, {'error': f'failed to derive from paired: {e}', 'base': base_path}

def calculate_rms(sig):
    return float(np.sqrt(np.mean(sig.astype(np.float32)**2)))

def scale_noise_to_snr(clean_signal, noise_signal, target_snr_db):
    rms_signal = calculate_rms(clean_signal)
    rms_noise = calculate_rms(noise_signal)
    if rms_noise == 0:
        return noise_signal
    target_noise_rms = rms_signal / (10**(target_snr_db / 20))
    scaling_factor = target_noise_rms / rms_noise
    return (noise_signal * scaling_factor).astype(np.float32)

def generate_realistic_bw(length, fs=360, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(length) / fs
    freq1 = np.random.uniform(BW_FREQ_RANGE[0], BW_FREQ_RANGE[1])
    freq2 = np.random.uniform(BW_FREQ_RANGE[0], BW_FREQ_RANGE[1])
    phase1 = np.random.uniform(0, 2*np.pi)
    phase2 = np.random.uniform(0, 2*np.pi)
    amp1 = np.random.uniform(0.5, 1.5)
    amp2 = np.random.uniform(0.3, 0.8)
    bw_signal = (amp1 * np.sin(2*np.pi*freq1*t + phase1) + amp2 * np.sin(2*np.pi*freq2*t + phase2)).astype(np.float32)
    params = {'type': 'BW', 'frequencies': [float(freq1), float(freq2)], 'phases': [float(phase1), float(phase2)], 'amplitudes': [float(amp1), float(amp2)], 'seed': seed}
    return bw_signal, params

def generate_realistic_ma(length, fs=360, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, 1, length).astype(np.float32)
    nyquist = fs / 2.0
    low_freq = MA_FREQ_RANGE[0] / nyquist
    high_freq = MA_FREQ_RANGE[1] / nyquist
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    ma_signal = signal.filtfilt(b, a, noise).astype(np.float32)
    has_burst = False
    if np.random.random() < MA_BURST_PROB:
        has_burst = True
        burst_start = np.random.randint(0, max(1, length - 200))
        burst_length = np.random.randint(50, 200)
        burst_end = min(burst_start + burst_length, length)
        burst_amplitude = np.random.uniform(2.0, 4.0)
        ma_signal[burst_start:burst_end] += (burst_amplitude * np.random.normal(0, 1, burst_end - burst_start)).astype(np.float32)
    params = {'type': 'MA', 'freq_range': MA_FREQ_RANGE, 'has_burst': bool(has_burst), 'seed': seed}
    return ma_signal, params

def generate_realistic_pli(length, fs=360, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(length) / fs
    base_freq = float(np.random.choice(PLI_FREQUENCIES))
    freq_deviation = float(np.random.uniform(-0.5, 0.5))
    effective_freq = base_freq + freq_deviation
    phase = float(np.random.uniform(0, 2*np.pi))
    pli_signal = np.sin(2*np.pi*effective_freq*t + phase).astype(np.float32)
    harmonics_added = []
    if np.random.random() < PLI_HARMONICS_PROB:
        phase_3rd = float(np.random.uniform(0, 2*np.pi))
        pli_signal += (PLI_3RD_HARMONIC_AMP * np.sin(2*np.pi*(3*effective_freq)*t + phase_3rd)).astype(np.float32)
        harmonics_added.append(3)
    if np.random.random() < PLI_HARMONICS_PROB:
        phase_5th = float(np.random.uniform(0, 2*np.pi))
        pli_signal += (PLI_5TH_HARMONIC_AMP * np.sin(2*np.pi*(5*effective_freq)*t + phase_5th)).astype(np.float32)
        harmonics_added.append(5)
    params = {'type': 'PLI', 'base_frequency': base_freq, 'effective_frequency': effective_freq, 'phase': phase, 'harmonics': harmonics_added, 'seed': seed}
    return pli_signal, params

def choose_noise_combination(segment_idx, seed=None):
    if seed is not None:
        np.random.seed(seed + segment_idx)
    weights = np.array(COMBINATION_WEIGHTS)
    weights = weights / np.sum(weights)
    choice_idx = np.random.choice(len(NOISE_COMBINATIONS), p=weights)
    return NOISE_COMBINATIONS[choice_idx]

def generate_combined_noise_snr_based(ecg_signal, noise_combination, target_snr_db, segment_seed, fs):
    length = len(ecg_signal)
    noise_components = {}
    noise_params = {}
    total_noise = np.zeros(length, dtype=np.float32)
    component_seeds = {}
    for i, noise_type in enumerate(noise_combination):
        component_seeds[noise_type] = segment_seed + i * 1000
    # BW e MA: usa rumore reale da NOISE_DIR (NSTDB); fallback sintetico se mancante
    if 'BW' in noise_combination:
        try:
            bw_noise, bw_params = load_noise_realistically(NOISE_RECORD_MAP['BW'], length, component_seeds['BW'])
            noise_components['BW'] = bw_noise
            noise_params['BW'] = bw_params
        except Exception as e:
            print(f"⚠️  Fallback BW sintetico: {e}")
            bw_noise, bw_params = generate_realistic_bw(length, fs, component_seeds['BW'])
            bw_params = {**bw_params, 'source': 'synthetic_fallback'}
            noise_components['BW'] = bw_noise
            noise_params['BW'] = bw_params
    if 'MA' in noise_combination:
        try:
            ma_noise, ma_params = load_noise_realistically(NOISE_RECORD_MAP['MA'], length, component_seeds['MA'])
            noise_components['MA'] = ma_noise
            noise_params['MA'] = ma_params
        except Exception as e:
            print(f"⚠️  Fallback MA sintetico: {e}")
            ma_noise, ma_params = generate_realistic_ma(length, fs, component_seeds['MA'])
            ma_params = {**ma_params, 'source': 'synthetic_fallback'}
            noise_components['MA'] = ma_noise
            noise_params['MA'] = ma_params
    if 'PLI' in noise_combination:
        pli_noise, pli_params = generate_realistic_pli(length, fs, component_seeds['PLI'])
        noise_components['PLI'] = pli_noise
        noise_params['PLI'] = pli_params
    if USE_SNR_TARGET and noise_components:
        # Sum unit components then scale to target SNR
        mix = np.zeros(length, dtype=np.float32)
        for comp in noise_components.values():
            mix += comp.astype(np.float32)
        mix = scale_noise_to_snr(ecg_signal, mix, target_snr_db)
        total_noise = mix
    else:
        for comp in noise_components.values():
            total_noise += comp.astype(np.float32)
    noisy_signal = ecg_signal.astype(np.float32) + total_noise
    # achieved SNR
    noise_rms = calculate_rms(total_noise)
    sig_rms = calculate_rms(ecg_signal)
    actual_snr = 20*np.log10(sig_rms / max(noise_rms, 1e-8)) if noise_rms > 0 else float('inf')
    segment_params = {
        'noise_combination': list(noise_combination),
        'target_snr_db': float(target_snr_db),
        'actual_snr_db': float(actual_snr),
        'segment_seed': int(segment_seed),
        'component_seeds': component_seeds,
        'noise_params': noise_params,
        'timestamp': datetime.now().isoformat()
    }
    return noisy_signal.astype(np.float32), total_noise.astype(np.float32), segment_params

def segment_signal(sig, segment_len, overlap_len):
    segments = []
    step = segment_len - overlap_len
    for i in range(0, len(sig) - segment_len + 1, step):
        segments.append(sig[i:i+segment_len].astype(np.float32))
    return np.array(segments, dtype=np.float32) if segments else np.empty((0, segment_len), dtype=np.float32)

def save_segmented_signals_with_metadata(output_dir, base_record_name, segment_idx, noisy_seg, noise_seg, clean_seg, metadata):
    try:
        os.makedirs(output_dir, exist_ok=True)
        fn = f"{base_record_name}_{segment_idx:03d}.npz"
        out_path = os.path.join(output_dir, fn)
        # X=noisy, y=noise (target)
        np.savez_compressed(out_path, X=noisy_seg.astype(np.float32), y=noise_seg.astype(np.float32))
        # Salva anche file compatibili con il consolidatore (folders mode)
        noisy_dir = os.path.join(output_dir, "noisy")
        clean_dir = os.path.join(output_dir, "clean")
        os.makedirs(noisy_dir, exist_ok=True)
        os.makedirs(clean_dir, exist_ok=True)
        np.save(os.path.join(noisy_dir, f"{base_record_name}_{segment_idx:03d}.npy"), noisy_seg.astype(np.float32))
        np.save(os.path.join(clean_dir, f"{base_record_name}_{segment_idx:03d}.npy"), clean_seg.astype(np.float32))
        if LOG_PARAMETERS:
            meta_fn = f"{base_record_name}_{segment_idx:03d}.json"
            with open(os.path.join(METADATA_DIR, meta_fn), 'w') as f:
                json.dump(metadata, f)
        return True
    except Exception as e:
        print(f"❌ Error saving segment {base_record_name}_{segment_idx:03d}: {e}")
        return False

def process_record_and_save_segments(ecg_record_name, split_dir):
    ecg_signal_full, fs = load_ecg_full(ecg_record_name)
    if ecg_signal_full is None:
        return 0
    clean_segments = segment_signal(ecg_signal_full, SEGMENT_LENGTH, OVERLAP_LENGTH)
    num_segments = clean_segments.shape[0]
    if num_segments == 0:
        return 0
    segments_processed_count = 0
    for i in range(num_segments):
        clean_seg = clean_segments[i]
        noise_combo = choose_noise_combination(i, seed=RANDOM_SEED_BASE)
        target_snr = float(np.random.uniform(*TARGET_SNR_RANGE))
        noisy_seg, noise_seg, params = generate_combined_noise_snr_based(
            clean_seg, noise_combo, target_snr, RANDOM_SEED_BASE + i, fs
        )
        ok = save_segmented_signals_with_metadata(
            split_dir, ecg_record_name, i, noisy_seg, noise_seg, clean_seg, params
        )
        if ok:
            segments_processed_count += 1
    return segments_processed_count

if __name__ == "__main__":
    # Determine splits by record using centralized split
    train_recs, val_recs, test_recs = get_train_test_split()
    split_map = {
        'train': train_recs or [],
        'validation': val_recs or [],
        'test': test_recs or [],
    }
    total = 0
    for split, recs in split_map.items():
        split_dir = os.path.join(SEGMENTED_SIGNAL_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Generating segments for split '{split}' with {len(recs)} records → {split_dir}")
        for rec in tqdm(recs, desc=f"{split}"):
            total += process_record_and_save_segments(rec, split_dir)
    print(f"✅ Generation complete. Total segments: {total}")