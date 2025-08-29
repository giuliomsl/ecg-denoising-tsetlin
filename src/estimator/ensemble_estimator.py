#!/usr/bin/env python3
"""
ENSEMBLE BEAST MODE 🚀🦾⚡
Advanced ensemble for TMU estimators with intelligent aggregation
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import h5py
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report


class TMUEnsemble:
    """Advanced ensemble of TMU models with intelligent voting."""
    
    def __init__(self, models: List[Any], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.n_models = len(models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict_proba(self, X: np.ndarray, use_conv_flags: List[bool]) -> np.ndarray:
        """Predict class probabilities with weighted averaging."""
        all_predictions = []
        
        for i, (model, use_conv) in enumerate(zip(self.models, use_conv_flags)):
            # Convert input based on model type
            if use_conv:
                X_processed = np.ascontiguousarray(X.astype(np.uint32, copy=False))
            else:
                X_processed = np.ascontiguousarray(X.reshape(X.shape[0], -1).astype(np.uint32, copy=False))
            
            # Get predictions
            pred = model.predict(X_processed)
            all_predictions.append(pred)
        
        # Convert to numpy array (n_models, n_samples)
        all_predictions = np.array(all_predictions)
        
        # Weighted voting (assuming classification labels 0, 1, 2, ...)
        n_samples = all_predictions.shape[1]
        n_classes = int(all_predictions.max()) + 1
        
        # Create probability matrix
        proba = np.zeros((n_samples, n_classes))
        
        for i, weight in enumerate(self.weights):
            for j in range(n_samples):
                predicted_class = int(all_predictions[i, j])
                proba[j, predicted_class] += weight
        
        return proba
    
    def predict(self, X: np.ndarray, use_conv_flags: List[bool]) -> np.ndarray:
        """Predict classes using ensemble."""
        proba = self.predict_proba(X, use_conv_flags)
        return np.argmax(proba, axis=1).astype(np.uint32)
    
    def predict_with_confidence(self, X: np.ndarray, use_conv_flags: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with confidence scores."""
        proba = self.predict_proba(X, use_conv_flags)
        predictions = np.argmax(proba, axis=1).astype(np.uint32)
        confidence = np.max(proba, axis=1)
        return predictions, confidence


def load_model_ensemble(ensemble_dirs: List[Path], noise_type: str) -> Tuple[List[Any], List[bool], List[float]]:
    """Load models from ensemble directories with performance-based weighting."""
    models = []
    use_conv_flags = []
    weights = []
    
    print(f"🔄 Loading ensemble for {noise_type}...")
    
    for i, model_dir in enumerate(ensemble_dirs):
        model_file = model_dir / f"{noise_type.lower()}_model.pkl"
        metrics_file = model_dir / "metrics.json"
        
        if not model_file.exists():
            print(f"⚠️  Model not found: {model_file}")
            continue
            
        try:
            # Load model
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Determine if convolutional (heuristic)
            use_conv = hasattr(model, 'patch_dim') or 'Conv' in type(model).__name__
            
            # Load performance for weighting
            weight = 1.0
            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)
                    test_metrics = metrics.get('test_metrics', {})
                    if noise_type in test_metrics:
                        # Use combined F1 as weight
                        weight = test_metrics[noise_type].get('combined', 0.5)
                        weight = max(weight, 0.1)  # Minimum weight
            
            models.append(model)
            use_conv_flags.append(use_conv)
            weights.append(weight)
            
            print(f"  ✅ Loaded {model_dir.name}: weight={weight:.3f} conv={use_conv}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_file}: {e}")
    
    if not models:
        raise ValueError(f"No models loaded for {noise_type}")
    
    return models, use_conv_flags, weights


def _to_input(xs: np.ndarray, conv: bool) -> np.ndarray:
    """Convert input for model type."""
    if conv:
        return np.ascontiguousarray(xs.astype(np.uint32, copy=False))
    return np.ascontiguousarray(xs.reshape(xs.shape[0], -1).astype(np.uint32, copy=False))


def _presence_from_bins(ybin: np.ndarray) -> np.ndarray:
    """Convert bin predictions to presence."""
    return (ybin > 0).astype(np.uint8)


def _f1_presence(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """F1 for presence detection."""
    y_true_pres = _presence_from_bins(y_true_bin)
    y_pred_pres = _presence_from_bins(y_pred_bin)
    return float(f1_score(y_true_pres, y_pred_pres, average="macro"))


def _f1_snr_only_present(y_true_bins: np.ndarray, y_pred_bins: np.ndarray) -> float:
    """F1 for SNR classification on present samples only."""
    mask = (y_true_bins > 0)
    if not np.any(mask):
        return 0.0
    yt = y_true_bins[mask]
    yp = y_pred_bins[mask]
    uniq = np.unique(yt)
    return float(f1_score(yt, yp, labels=uniq, average="macro"))


def evaluate_ensemble(ensemble: TMUEnsemble, use_conv_flags: List[bool], 
                     X: np.ndarray, y_true: np.ndarray, 
                     batch_size: int = 1024) -> Dict[str, float]:
    """Evaluate ensemble performance."""
    n_samples = X.shape[0]
    all_predictions = []
    
    # Batch prediction for memory efficiency
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        pred_batch = ensemble.predict(X_batch, use_conv_flags)
        all_predictions.append(pred_batch)
    
    y_pred = np.concatenate(all_predictions)
    
    # Calculate metrics
    f1_pres = _f1_presence(y_true, y_pred)
    f1_snr = _f1_snr_only_present(y_true, y_pred)
    combined = 0.5 * f1_pres + 0.5 * f1_snr
    
    # Presence accuracy
    y_true_pres = _presence_from_bins(y_true)
    y_pred_pres = _presence_from_bins(y_pred)
    acc_pres = float(accuracy_score(y_true_pres, y_pred_pres))
    
    return {
        'f1_presence': f1_pres,
        'f1_snr': f1_snr,
        'combined': combined,
        'acc_presence': acc_pres
    }


def analyze_ensemble_diversity(models: List[Any], use_conv_flags: List[bool], 
                              X_sample: np.ndarray) -> Dict[str, float]:
    """Analyze diversity metrics of the ensemble."""
    predictions = []
    
    for model, use_conv in zip(models, use_conv_flags):
        X_proc = _to_input(X_sample, use_conv)
        pred = model.predict(X_proc)
        predictions.append(pred)
    
    predictions = np.array(predictions)  # (n_models, n_samples)
    
    # Agreement rate
    n_models, n_samples = predictions.shape
    agreement_rates = []
    
    for i in range(n_samples):
        sample_predictions = predictions[:, i]
        most_common = np.bincount(sample_predictions).argmax()
        agreement = np.sum(sample_predictions == most_common) / n_models
        agreement_rates.append(agreement)
    
    avg_agreement = np.mean(agreement_rates)
    diversity = 1.0 - avg_agreement
    
    # Pairwise disagreement
    disagreements = []
    for i in range(n_models):
        for j in range(i+1, n_models):
            disagreement = np.mean(predictions[i] != predictions[j])
            disagreements.append(disagreement)
    
    avg_disagreement = np.mean(disagreements) if disagreements else 0.0
    
    return {
        'diversity': diversity,
        'agreement': avg_agreement,
        'pairwise_disagreement': avg_disagreement
    }


def main():
    parser = argparse.ArgumentParser(description="BEAST MODE Ensemble Evaluation")
    parser.add_argument('--ensemble-dirs', nargs='+', required=True, 
                       help='List of model directories for ensemble')
    parser.add_argument('--h5', required=True, help='Test dataset')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--noise-types', default='BW,MA,PLI', 
                       help='Noise types to evaluate (comma-separated)')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for evaluation')
    parser.add_argument('--save-predictions', action='store_true', 
                       help='Save ensemble predictions')
    
    args = parser.parse_args()
    
    # Setup
    ensemble_dirs = [Path(d) for d in args.ensemble_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    noise_types = [t.strip().upper() for t in args.noise_types.split(',')]
    
    print("🚀 BEAST MODE ENSEMBLE EVALUATION")
    print("=" * 50)
    print(f"📁 Ensemble dirs: {[d.name for d in ensemble_dirs]}")
    print(f"📊 Dataset: {args.h5}")
    print(f"🎯 Noise types: {noise_types}")
    print(f"💾 Output: {output_dir}")
    
    # Load test data
    with h5py.File(args.h5, 'r') as h5:
        X_test = h5['test/X'][:]
        y_test_all = h5['test/y_snr_bin'][:] if 'test/y_snr_bin' in h5 else np.stack([
            h5['test/targets/binlabel/bw'][:],
            h5['test/targets/binlabel/ma'][:],
            h5['test/targets/binlabel/pli'][:]
        ], axis=1)
    
    print(f"📏 Test data: X={X_test.shape} y={y_test_all.shape}")
    
    # Evaluate each noise type
    all_results = {}
    
    for i, noise_type in enumerate(noise_types):
        print(f"\n🎯 Evaluating {noise_type} ensemble...")
        
        try:
            # Load ensemble
            models, use_conv_flags, weights = load_model_ensemble(ensemble_dirs, noise_type)
            ensemble = TMUEnsemble(models, weights)
            
            print(f"   Ensemble size: {len(models)} models")
            print(f"   Weights: {[f'{w:.3f}' for w in ensemble.weights]}")
            
            # Test data for this noise type
            y_test = y_test_all[:, i].astype(np.uint32, copy=False)
            
            # Evaluate ensemble
            metrics = evaluate_ensemble(ensemble, use_conv_flags, X_test, y_test, args.batch_size)
            
            # Analyze diversity (on subset for speed)
            n_diversity_samples = min(1000, X_test.shape[0])
            diversity_indices = np.random.choice(X_test.shape[0], n_diversity_samples, replace=False)
            X_diversity = X_test[diversity_indices]
            diversity_metrics = analyze_ensemble_diversity(models, use_conv_flags, X_diversity)
            
            # Combine metrics
            all_metrics = {**metrics, **diversity_metrics}
            all_results[noise_type] = all_metrics
            
            print(f"   🏆 Results:")
            print(f"     Combined F1: {metrics['combined']:.4f}")
            print(f"     F1 Presence: {metrics['f1_presence']:.4f}")
            print(f"     F1 SNR: {metrics['f1_snr']:.4f}")
            print(f"     Diversity: {diversity_metrics['diversity']:.4f}")
            
            # Save individual ensemble
            ensemble_path = output_dir / f'{noise_type.lower()}_ensemble.pkl'
            try:
                with open(ensemble_path, 'wb') as f:
                    pickle.dump((ensemble, use_conv_flags), f)
                print(f"   💾 Ensemble saved to: {ensemble_path}")
            except Exception as e:
                print(f"   ⚠️  Could not save ensemble: {e}")
            
            # Save predictions if requested
            if args.save_predictions:
                predictions = ensemble.predict(X_test, use_conv_flags)
                predictions_path = output_dir / f'{noise_type.lower()}_predictions.npy'
                np.save(predictions_path, predictions)
                print(f"   💾 Predictions saved to: {predictions_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to evaluate {noise_type}: {e}")
            all_results[noise_type] = {'error': str(e)}
    
    # Save comprehensive results
    results_file = output_dir / 'ensemble_results.json'
    summary = {
        'ensemble_dirs': [str(d) for d in ensemble_dirs],
        'noise_types': noise_types,
        'results': all_results,
        'dataset_info': {
            'test_samples': int(X_test.shape[0]),
            'features': f"{X_test.shape[1]} x {X_test.shape[2]}"
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Complete results saved to: {results_file}")
    
    # Print summary
    print(f"\n🏆 ENSEMBLE PERFORMANCE SUMMARY")
    print("=" * 50)
    
    for noise_type in noise_types:
        if noise_type in all_results and 'combined' in all_results[noise_type]:
            metrics = all_results[noise_type]
            print(f"{noise_type:3}: Combined={metrics['combined']:.4f} "
                  f"Pres={metrics['f1_presence']:.4f} "
                  f"SNR={metrics['f1_snr']:.4f} "
                  f"Div={metrics.get('diversity', 0):.4f}")
        else:
            print(f"{noise_type:3}: ERROR")
    
    # Overall average
    valid_results = [r for r in all_results.values() if 'combined' in r]
    if valid_results:
        avg_combined = np.mean([r['combined'] for r in valid_results])
        print(f"\n🎯 OVERALL AVERAGE: {avg_combined:.4f}")
    
    print(f"\n🎉 ENSEMBLE EVALUATION COMPLETE!")


if __name__ == '__main__':
    main()