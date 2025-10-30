#!/usr/bin/env python3
"""
Calibration Analysis for TMU-Optimized (V7)
============================================
Analizza l'impatto della calibrazione isotonica su V7:
- Raw predictions vs calibrated
- Recovery metrics (correlation improvement)
- Isotonic curves visualization
- Comparison con V4 (TMU-Regularized)

Output:
- plots/v7_calibration/calibration_curves_*.png
- plots/v7_calibration/calibration_impact.json
- plots/v7_calibration/v4_vs_v7_calibration_comparison.png
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# Setup
PLOTS_DIR = Path("plots/v7_calibration")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load V7 predictions (calibrated AND raw)
print("Loading V7 predictions...")
data_v7 = np.load("results/v7_predictions/v7_test_predictions.npz")
y_true_bw_v7 = data_v7["y_true_bw"]
y_pred_bw_v7_cal = data_v7["y_pred_bw_cal"]  # Calibrated
y_pred_bw_v7_cal_raw = data_v7["y_pred_bw_raw"]  # Raw
y_true_emg_v7 = data_v7["y_true_emg"]
y_pred_emg_v7_cal = data_v7["y_pred_emg_cal"]  # Calibrated
y_pred_emg_v7_cal_raw = data_v7["y_pred_emg_raw"]  # Raw

# Calculate raw correlations
r_raw_bw_v7, _ = pearsonr(y_true_bw_v7, y_pred_bw_v7_cal_raw)
r_raw_emg_v7, _ = pearsonr(y_true_emg_v7, y_pred_emg_v7_cal_raw)
print(f"  Raw correlations: BW={r_raw_bw_v7:.4f}, EMG={r_raw_emg_v7:.4f}")

# Load V4 for comparison
print("\nLoading V4 predictions for comparison...")
# Use known values from evaluation
print("  Using known V4 r values from evaluation")
r_cal_bw_v4 = 0.6231
r_cal_emg_v4 = 0.6797
y_pred_bw_v4 = None
y_pred_emg_v4 = None

# Calculate metrics
print("\n" + "="*80)
print("CALIBRATION IMPACT ANALYSIS - TMU-Optimized (V7)")
print("="*80)

# V7 metrics
r_cal_bw_v7, _ = pearsonr(y_true_bw_v7, y_pred_bw_v7_cal)
r_cal_emg_v7, _ = pearsonr(y_true_emg_v7, y_pred_emg_v7_cal)
mae_bw_v7 = np.mean(np.abs(y_true_bw_v7 - y_pred_bw_v7_cal))
mae_emg_v7 = np.mean(np.abs(y_true_emg_v7 - y_pred_emg_v7_cal))

# V4 metrics
if y_pred_bw_v4 is not None:
    r_cal_bw_v4, _ = pearsonr(y_true_bw_v7, y_pred_bw_v4)  # Same y_true
    r_cal_emg_v4, _ = pearsonr(y_true_emg_v7, y_pred_emg_v4)
else:
    # Use known values
    r_cal_bw_v4 = 0.6231
    r_cal_emg_v4 = 0.6797

print(f"\nV7 CALIBRATED PERFORMANCE (Test Set):")
print(f"  BW:  r = {r_cal_bw_v7:.4f}, MAE = {mae_bw_v7:.4f}")
print(f"  EMG: r = {r_cal_emg_v7:.4f}, MAE = {mae_emg_v7:.4f}")

print(f"\nV4 CALIBRATED PERFORMANCE (Test Set):")
print(f"  BW:  r = {r_cal_bw_v4:.4f}")
print(f"  EMG: r = {r_cal_emg_v4:.4f}")

# Estimate raw performance from training correlation
# Abbiamo i raw dal file!
recovery_bw = ((r_cal_bw_v7 - r_raw_bw_v7) / r_raw_bw_v7) * 100
recovery_emg = ((r_cal_emg_v7 - r_raw_emg_v7) / r_raw_emg_v7) * 100

print(f"\nCALIBRATION RECOVERY (Test Set):")
print(f"  BW:  {r_raw_bw_v7:.4f} → {r_cal_bw_v7:.4f} (+{recovery_bw:.0f}%)")
print(f"  EMG: {r_raw_emg_v7:.4f} → {r_cal_emg_v7:.4f} (+{recovery_emg:.0f}%)")

# Save metrics
metrics = {
    "v7": {
        "bw": {
            "r_raw_test": float(r_raw_bw_v7),
            "r_calibrated_test": float(r_cal_bw_v7),
            "mae_calibrated_test": float(mae_bw_v7),
            "recovery_pct": float(recovery_bw)
        },
        "emg": {
            "r_raw_test": float(r_raw_emg_v7),
            "r_calibrated_test": float(r_cal_emg_v7),
            "mae_calibrated_test": float(mae_emg_v7),
            "recovery_pct": float(recovery_emg)
        }
    },
    "v4": {
        "bw": {
            "r_calibrated_test": float(r_cal_bw_v4)
        },
        "emg": {
            "r_calibrated_test": float(r_cal_emg_v4)
        }
    },
    "comparison": {
        "delta_bw": float(r_cal_bw_v7 - r_cal_bw_v4),
        "delta_emg": float(r_cal_emg_v7 - r_cal_emg_v4),
        "delta_bw_pct": float(((r_cal_bw_v7 - r_cal_bw_v4) / r_cal_bw_v4) * 100),
        "delta_emg_pct": float(((r_cal_emg_v7 - r_cal_emg_v4) / r_cal_emg_v4) * 100)
    }
}

with open(PLOTS_DIR / "calibration_impact.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n✅ Metrics saved to: {PLOTS_DIR}/calibration_impact.json")

# ============================================================================
# PLOT 1: Calibration Curves (Scatter plots)
# ============================================================================
print("\nGenerating calibration curves plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("TMU-Optimized (V7) - Calibration Impact", fontsize=16, fontweight='bold')

# BW - Calibrated
ax = axes[0, 0]
ax.scatter(y_true_bw_v7, y_pred_bw_v7_cal, alpha=0.3, s=10)
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
ax.set_xlabel('True Intensity', fontsize=11)
ax.set_ylabel('Predicted Intensity (Calibrated)', fontsize=11)
ax.set_title(f'BW - Calibrated\nr = {r_cal_bw_v7:.4f}, MAE = {mae_bw_v7:.4f}', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# EMG - Calibrated
ax = axes[0, 1]
ax.scatter(y_true_emg_v7, y_pred_emg_v7_cal, alpha=0.3, s=10)
ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
ax.set_xlabel('True Intensity', fontsize=11)
ax.set_ylabel('Predicted Intensity (Calibrated)', fontsize=11)
ax.set_title(f'EMG - Calibrated\nr = {r_cal_emg_v7:.4f}, MAE = {mae_emg_v7:.4f}', fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# Residuals BW
ax = axes[1, 0]
residuals_bw = y_pred_bw_v7_cal - y_true_bw_v7
ax.scatter(y_true_bw_v7, residuals_bw, alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('True Intensity', fontsize=11)
ax.set_ylabel('Residuals (Pred - True)', fontsize=11)
ax.set_title(f'BW - Residuals\nMean = {np.mean(residuals_bw):.4f}', fontsize=12)
ax.grid(alpha=0.3)

# Residuals EMG
ax = axes[1, 1]
residuals_emg = y_pred_emg_v7_cal - y_true_emg_v7
ax.scatter(y_true_emg_v7, residuals_emg, alpha=0.3, s=10)
ax.axhline(y=0, color='r', linestyle='--', lw=2)
ax.set_xlabel('True Intensity', fontsize=11)
ax.set_ylabel('Residuals (Pred - True)', fontsize=11)
ax.set_title(f'EMG - Residuals\nMean = {np.mean(residuals_emg):.4f}', fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "calibration_curves_v7.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: {PLOTS_DIR}/calibration_curves_v7.png")

# ============================================================================
# PLOT 2: V4 vs V7 Calibration Comparison
# ============================================================================
print("Generating V4 vs V7 comparison plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("TMU-Regularized (V4) vs TMU-Optimized (V7) - Calibrated Performance", 
             fontsize=14, fontweight='bold')

# BW Comparison
ax = axes[0]
if y_pred_bw_v4 is not None:
    ax.scatter(y_true_bw_v7, y_pred_bw_v4, alpha=0.3, s=10, label=f'V4 (r={r_cal_bw_v4:.4f})', color='blue')
ax.scatter(y_true_bw_v7, y_pred_bw_v7_cal, alpha=0.3, s=10, label=f'V7 (r={r_cal_bw_v7:.4f})', color='red')
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Perfect')
ax.set_xlabel('True Intensity', fontsize=11)
ax.set_ylabel('Predicted Intensity', fontsize=11)
ax.set_title(f'BW Head\nΔr = {r_cal_bw_v7 - r_cal_bw_v4:.4f} ({((r_cal_bw_v7 - r_cal_bw_v4)/r_cal_bw_v4)*100:.1f}%)', 
             fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

# EMG Comparison
ax = axes[1]
if y_pred_emg_v4 is not None:
    ax.scatter(y_true_emg_v7, y_pred_emg_v4, alpha=0.3, s=10, label=f'V4 (r={r_cal_emg_v4:.4f})', color='blue')
ax.scatter(y_true_emg_v7, y_pred_emg_v7_cal, alpha=0.3, s=10, label=f'V7 (r={r_cal_emg_v7:.4f})', color='red')
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Perfect')
ax.set_xlabel('True Intensity', fontsize=11)
ax.set_ylabel('Predicted Intensity', fontsize=11)
ax.set_title(f'EMG Head\nΔr = {r_cal_emg_v7 - r_cal_emg_v4:.4f} ({((r_cal_emg_v7 - r_cal_emg_v4)/r_cal_emg_v4)*100:.1f}%)', 
             fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "v4_vs_v7_calibration_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: {PLOTS_DIR}/v4_vs_v7_calibration_comparison.png")

# ============================================================================
# PLOT 3: Recovery Bar Chart
# ============================================================================
print("Generating recovery comparison plot...")

fig, ax = plt.subplots(figsize=(10, 6))

models = ['V4-BW', 'V4-EMG', 'V7-BW', 'V7-EMG']
recovery_vals = [
    -35.8,  # V4 BW (from presentation)
    -33.0,  # V4 EMG (from presentation)
    recovery_bw,
    recovery_emg
]

colors = ['blue', 'blue', 'red', 'red']
bars = ax.bar(models, recovery_vals, color=colors, alpha=0.7, edgecolor='black')

# Annotazioni
for bar, val in zip(bars, recovery_vals):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:+.0f}%',
            ha='center', va='bottom' if val > 0 else 'top', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_ylabel('Calibration Recovery (%)', fontsize=12, fontweight='bold')
ax.set_title('Calibration Impact: Raw → Calibrated Performance Change', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', alpha=0.7, label='TMU-Regularized (V4): Reduces overfitting'),
    Patch(facecolor='red', alpha=0.7, label='TMU-Optimized (V7): Essential for performance')
]
ax.legend(handles=legend_elements, loc='best', fontsize=10)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "calibration_recovery_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: {PLOTS_DIR}/calibration_recovery_comparison.png")

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*80)
print("CALIBRATION ANALYSIS SUMMARY")
print("="*80)
print(f"\nV7 (TMU-Optimized):")
print(f"  - Raw performance molto bassa (r_BW={r_raw_bw_v7:.4f}, r_EMG={r_raw_emg_v7:.4f})")
print(f"  - Calibration ESSENZIALE: recovery +{recovery_bw:.0f}% (BW), +{recovery_emg:.0f}% (EMG)")
print(f"  - Feature reduction (95%) richiede calibration per generalizzare")
print(f"\nV4 (TMU-Regularized):")
print(f"  - Raw performance alta (r~0.999, overfitting)")
print(f"  - Calibration riduce overfitting: -35.8% (BW), -33.0% (EMG)")
print(f"  - Calibration migliora generalizzazione")
print(f"\nKey Insight:")
print(f"  - V4: Calibration corregge overfitting")
print(f"  - V7: Calibration recupera capacità predittiva")
print(f"  - Entrambi: Calibration isotonica necessaria per deployment")
print("\n" + "="*80)
print(f"\n✅ ALL PLOTS SAVED TO: {PLOTS_DIR}/")
print("   - calibration_curves_v7.png")
print("   - v4_vs_v7_calibration_comparison.png")
print("   - calibration_recovery_comparison.png")
print("   - calibration_impact.json")
print("="*80)
