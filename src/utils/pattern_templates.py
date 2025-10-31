"""
Pattern templates library for noise classification.
Maps bitplane responses to physiological noise patterns with frequency characteristics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class NoisePattern:
    """Represents a physiological noise pattern with its characteristics."""
    
    def __init__(self, 
                 name: str,
                 description: str,
                 frequency_range: Tuple[float, float],
                 physiological_origin: str,
                 bitplane_signature: Dict[str, List[int]],
                 recommended_filter: str):
        self.name = name
        self.description = description
        self.frequency_range = frequency_range  # (min_hz, max_hz)
        self.physiological_origin = physiological_origin
        self.bitplane_signature = bitplane_signature  # {'bitplanes': [0,1,2], 'derivative': [3,4]}
        self.recommended_filter = recommended_filter
    
    def matches(self, active_bitplanes: List[int], noise_type: str, threshold: float = 0.5) -> bool:
        """Check if active bitplanes match this pattern's signature."""
        if noise_type.lower() not in self.bitplane_signature:
            return False
        
        expected = set(self.bitplane_signature[noise_type.lower()])
        actual = set(active_bitplanes)
        
        # Calculate Jaccard similarity
        if len(expected) == 0:
            return False
        
        intersection = len(expected & actual)
        union = len(expected | actual)
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold


# ============================================================================
# BASELINE WANDER (BW) PATTERNS
# ============================================================================

BW_RESPIRATORY = NoisePattern(
    name="Baseline Wander - Respiratory",
    description="Baseline drift synchronized with breathing (respiration artifact)",
    frequency_range=(0.15, 0.5),  # 9-30 breaths/min
    physiological_origin="Respiration-induced chest impedance changes affecting electrode contact",
    bitplane_signature={
        'bw': [0, 1],  # Very low frequencies
        'derivative': [0, 1]
    },
    recommended_filter="Butterworth highpass 0.5 Hz (order 4)"
)

BW_MOVEMENT = NoisePattern(
    name="Baseline Wander - Movement",
    description="Slow baseline drift from patient movement or position changes",
    frequency_range=(0.05, 0.3),
    physiological_origin="Electrode motion, skin stretching, body position changes",
    bitplane_signature={
        'bw': [0, 1, 2],
        'derivative': [0, 1, 2]
    },
    recommended_filter="Butterworth highpass 0.5 Hz (order 4) or cubic spline detrending"
)

BW_CARDIAC_RESPIRATORY = NoisePattern(
    name="Baseline Wander - Cardiorespiratory",
    description="Combined cardiac and respiratory modulation of baseline",
    frequency_range=(0.1, 1.0),
    physiological_origin="Interaction between heart rate (~1 Hz) and respiration (~0.3 Hz)",
    bitplane_signature={
        'bw': [0, 1, 2, 3],
        'derivative': [0, 1, 2]
    },
    recommended_filter="Butterworth highpass 0.5 Hz (order 4) with careful tuning"
)

BW_ELECTRODE_DRIFT = NoisePattern(
    name="Baseline Wander - Electrode Drift",
    description="Very slow DC drift from electrode polarization or gel drying",
    frequency_range=(0.01, 0.1),
    physiological_origin="Electrode-skin interface impedance changes, gel drying, skin hydration",
    bitplane_signature={
        'bw': [0],
        'derivative': [0]
    },
    recommended_filter="Median filter (window 200-600 ms) or linear detrending"
)


# ============================================================================
# ELECTROMYOGRAPHIC (EMG) PATTERNS
# ============================================================================

EMG_MUSCLE_BURST = NoisePattern(
    name="EMG - Muscle Burst",
    description="High-frequency bursts from skeletal muscle contractions",
    frequency_range=(20, 150),
    physiological_origin="Voluntary or involuntary skeletal muscle contractions near electrodes",
    bitplane_signature={
        'emg': [4, 5, 6, 7],  # High bitplanes = high frequencies
        'derivative': [5, 6, 7]
    },
    recommended_filter="Wavelet denoising (db6, level 4) or adaptive filter"
)

EMG_TREMOR = NoisePattern(
    name="EMG - Tremor",
    description="Rhythmic oscillations from physiological or pathological tremor",
    frequency_range=(4, 12),
    physiological_origin="Essential tremor, Parkinsonian tremor, or physiological tremor",
    bitplane_signature={
        'emg': [3, 4, 5],
        'derivative': [3, 4, 5]
    },
    recommended_filter="Notch filter at tremor frequency ± 2 Hz, or adaptive LMS filter"
)

EMG_CONTINUOUS = NoisePattern(
    name="EMG - Continuous",
    description="Sustained high-frequency noise from tonic muscle activity",
    frequency_range=(30, 200),
    physiological_origin="Continuous muscle tension (e.g., shivering, anxiety-induced tension)",
    bitplane_signature={
        'emg': [5, 6, 7],
        'derivative': [5, 6, 7]
    },
    recommended_filter="Wavelet denoising (db6, level 5) or Savitzky-Golay filter"
)

EMG_PECTORAL = NoisePattern(
    name="EMG - Pectoral Muscle",
    description="Interference from pectoralis major muscle activity",
    frequency_range=(15, 100),
    physiological_origin="Pectoral muscle contractions during arm movements or deep breathing",
    bitplane_signature={
        'emg': [4, 5, 6],
        'derivative': [4, 5, 6]
    },
    recommended_filter="Wavelet denoising (sym8, level 4) or ICA separation"
)

EMG_INTERCOSTAL = NoisePattern(
    name="EMG - Intercostal",
    description="Interference from intercostal muscles during respiration",
    frequency_range=(20, 80),
    physiological_origin="Intercostal muscle activity synchronized with breathing",
    bitplane_signature={
        'emg': [4, 5],
        'derivative': [4, 5]
    },
    recommended_filter="Wavelet denoising (coif3, level 4) preserving cardiac cycles"
)


# ============================================================================
# POWERLINE INTERFERENCE (PLI) PATTERNS
# ============================================================================

PLI_50HZ = NoisePattern(
    name="PLI - 50 Hz Fundamental",
    description="Pure 50 Hz powerline interference (European standard)",
    frequency_range=(49, 51),
    physiological_origin="Electromagnetic coupling from AC power distribution (50 Hz regions)",
    bitplane_signature={
        'pli': [3, 4, 5],  # Narrow band around 50 Hz
        'derivative': [3, 4]
    },
    recommended_filter="Notch filter 50 Hz (Q=30) or adaptive notch filter"
)

PLI_60HZ = NoisePattern(
    name="PLI - 60 Hz Fundamental",
    description="Pure 60 Hz powerline interference (North American standard)",
    frequency_range=(59, 61),
    physiological_origin="Electromagnetic coupling from AC power distribution (60 Hz regions)",
    bitplane_signature={
        'pli': [4, 5],
        'derivative': [4, 5]
    },
    recommended_filter="Notch filter 60 Hz (Q=30) or adaptive notch filter"
)

PLI_HARMONICS = NoisePattern(
    name="PLI - Harmonics",
    description="Powerline harmonics (100 Hz, 150 Hz, etc.)",
    frequency_range=(90, 180),
    physiological_origin="Non-linear loads causing harmonic distortion in powerline",
    bitplane_signature={
        'pli': [5, 6, 7],
        'derivative': [5, 6, 7]
    },
    recommended_filter="Multi-notch filter (50/100/150 Hz) or comb filter"
)

PLI_TIME_VARYING = NoisePattern(
    name="PLI - Time-Varying",
    description="Powerline interference with amplitude/frequency modulation",
    frequency_range=(48, 62),
    physiological_origin="Variable coupling due to patient/equipment movement or grid fluctuations",
    bitplane_signature={
        'pli': [3, 4, 5, 6],
        'derivative': [3, 4, 5]
    },
    recommended_filter="Adaptive notch filter with frequency tracking or Kalman filter"
)

PLI_BROADBAND = NoisePattern(
    name="PLI - Broadband Contamination",
    description="Wideband powerline noise with multiple harmonics",
    frequency_range=(45, 200),
    physiological_origin="Poor grounding, multiple interference sources, or high impedance electrodes",
    bitplane_signature={
        'pli': [3, 4, 5, 6, 7],
        'derivative': [3, 4, 5, 6]
    },
    recommended_filter="Cascaded notch filters or spectral subtraction"
)


# ============================================================================
# PATTERN LIBRARY
# ============================================================================

PATTERN_LIBRARY = {
    'BW': [
        BW_RESPIRATORY,
        BW_MOVEMENT,
        BW_CARDIAC_RESPIRATORY,
        BW_ELECTRODE_DRIFT
    ],
    'EMG': [
        EMG_MUSCLE_BURST,
        EMG_TREMOR,
        EMG_CONTINUOUS,
        EMG_PECTORAL,
        EMG_INTERCOSTAL
    ],
    'PLI': [
        PLI_50HZ,
        PLI_60HZ,
        PLI_HARMONICS,
        PLI_TIME_VARYING,
        PLI_BROADBAND
    ]
}


def identify_pattern(active_bitplanes: List[int], 
                    noise_type: str, 
                    intensity: float,
                    match_threshold: float = 0.5) -> Optional[NoisePattern]:
    """
    Identify the most likely noise pattern based on active bitplanes.
    
    Args:
        active_bitplanes: List of bitplane indices with high contribution
        noise_type: 'BW', 'EMG', or 'PLI'
        intensity: Predicted noise intensity (0-1)
        match_threshold: Minimum similarity score for pattern matching
    
    Returns:
        Best matching NoisePattern or None if no good match
    """
    if noise_type.upper() not in PATTERN_LIBRARY:
        return None
    
    patterns = PATTERN_LIBRARY[noise_type.upper()]
    best_match = None
    best_score = 0
    
    for pattern in patterns:
        if pattern.matches(active_bitplanes, noise_type, threshold=match_threshold):
            # Calculate detailed matching score
            expected = set(pattern.bitplane_signature.get(noise_type.lower(), []))
            actual = set(active_bitplanes)
            
            if len(expected) == 0:
                continue
            
            intersection = len(expected & actual)
            score = intersection / len(expected)  # Precision score
            
            if score > best_score:
                best_score = score
                best_match = pattern
    
    return best_match


def get_active_bitplanes_from_features(features: np.ndarray, 
                                       top_k: int = 5,
                                       method: str = 'std') -> List[int]:
    """
    Extract most active bitplanes from feature vector.
    
    Args:
        features: Feature vector (8192,) with bitplane encoding
        top_k: Number of top bitplanes to return
        method: 'std' (standard deviation), 'mean' (mean absolute value), or 'energy'
    
    Returns:
        List of most active bitplane indices (0-7)
    """
    # Features are organized as: [bitplane_0, bitplane_1, ..., bitplane_7, derivative_0, ..., derivative_7]
    # Each bitplane has 512 features (time samples)
    n_bitplanes = 8
    samples_per_bitplane = len(features) // (2 * n_bitplanes)
    
    bitplane_scores = []
    for i in range(n_bitplanes):
        start_idx = i * samples_per_bitplane
        end_idx = (i + 1) * samples_per_bitplane
        bitplane_data = features[start_idx:end_idx]
        
        if method == 'std':
            score = np.std(bitplane_data)
        elif method == 'mean':
            score = np.mean(np.abs(bitplane_data))
        elif method == 'energy':
            score = np.sum(bitplane_data ** 2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        bitplane_scores.append((i, score))
    
    # Sort by score and return top_k indices
    bitplane_scores.sort(key=lambda x: x[1], reverse=True)
    return [bp_idx for bp_idx, _ in bitplane_scores[:top_k]]


def format_pattern_report(pattern: NoisePattern, intensity: float, confidence: float) -> str:
    """
    Format a human-readable report for a detected pattern.
    
    Args:
        pattern: Detected NoisePattern
        intensity: Noise intensity (0-1)
        confidence: Pattern matching confidence (0-1)
    
    Returns:
        Formatted text report
    """
    report = []
    report.append(f"Pattern: {pattern.name}")
    report.append(f"Description: {pattern.description}")
    report.append(f"Intensity: {intensity:.3f}")
    report.append(f"Confidence: {confidence:.2%}")
    report.append(f"Frequency Range: {pattern.frequency_range[0]}-{pattern.frequency_range[1]} Hz")
    report.append(f"Physiological Origin: {pattern.physiological_origin}")
    report.append(f"Recommended Filter: {pattern.recommended_filter}")
    
    return "\n".join(report)
