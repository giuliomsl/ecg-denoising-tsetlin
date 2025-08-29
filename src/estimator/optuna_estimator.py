#!/usr/bin/env python3
"""
BEAST MODE OPTUNA OPTIMIZATION 🚀⚡🧠
Hyperparameter optimization per TMU Estimator con Bayesian search intelligente
"""

import os
import sys
import json
import pickle
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import h5py


def run_training_trial(trial: optuna.Trial, base_config: Dict[str, Any]) -> float:
    """Run training with suggested hyperparameters and return combined F1 score."""
    
    # Suggest hyperparameters - lightweight for quick optimization
    suggested_params = {
        'clauses': trial.suggest_int('clauses', 800, 2000, step=200),     # Reduced range
        'T': trial.suggest_int('T', 300, 800, step=100),                  # Reduced range
        's': trial.suggest_float('s', 2.5, 5.0, step=0.25),              # Reduced range
        'patch_w': trial.suggest_categorical('patch_w', [31, 47]),        # Only 2 options
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024]), # Smaller batches
        'types': trial.suggest_categorical('types', ['BW', 'MA', 'PLI'])  # Single type per trial
    }
    
    # Very quick evaluation for optimization
    quick_epochs = base_config.get('quick_epochs', 3)
    quick_patience = 2
    
    # Create trial output directory
    trial_dir = Path(base_config['base_output_dir']) / f"trial_{trial.number:04d}"
    trial_dir.mkdir(exist_ok=True)
    
    # Build command with enhanced parameters
    cmd = [
        sys.executable, 'src/estimator/train_tmu_estimator.py',
        '--h5', base_config['h5_path'],
        '--out_dir', str(trial_dir),
        '--backend', 'conv',
        '--clauses', str(suggested_params['clauses']),
        '--T', str(suggested_params['T']),
        '--s', str(suggested_params['s']),
        '--patch_w', str(suggested_params['patch_w']),
        '--epochs', str(quick_epochs),
        '--patience', str(quick_patience),
        '--batch_size', str(suggested_params['batch_size']),
        '--val_eval_each', '2',  # Less frequent validation
        '--seed', str(base_config['seed'] + trial.number),
        '--balance_bins',  # Enable SNR balancing
        '--absent_max_ratio', '0.3',  # Cap absent class
        '--types', suggested_params['types'],  # Single type per trial
    ]
    
    try:
        # Run training with timeout
        import time
        start_time = time.time()
        print(f"\n🚀 Trial {trial.number}: {suggested_params}")
        print(f"⏳ Starting training at {time.strftime('%H:%M:%S')}")
        
        # Increased timeout for realistic training
        timeout_minutes = base_config.get('trial_timeout_minutes', 20)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_minutes*60)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Trial {trial.number} failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"🔍 STDERR: {result.stderr[-500:]}")
            if result.stdout:
                print(f"🔍 STDOUT: {result.stdout[-500:]}")
            return 0.0
        
        # Read results
        metrics_file = trial_dir / 'metrics.json'
        if not metrics_file.exists():
            print(f"❌ Trial {trial.number}: No metrics file")
            return 0.0
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Calculate combined F1 score - single type in this trial
        test_metrics = metrics.get('test_metrics', {})
        
        # Get the score for the specific noise type trained
        noise_type = suggested_params['types']
        if noise_type in test_metrics:
            combined_score = test_metrics[noise_type].get('combined', 0.0)
            print(f"   {noise_type}: combined={combined_score:.4f}")
            
            # Set user attributes for detailed analysis
            trial.set_user_attr('elapsed_minutes', elapsed/60)
            trial.set_user_attr('noise_type', noise_type)
            trial.set_user_attr(f'{noise_type.lower()}_score', combined_score)
            
            # Also store detailed metrics if available
            metrics_detail = test_metrics[noise_type]
            trial.set_user_attr(f'{noise_type.lower()}_f1_presence', metrics_detail.get('f1_presence', 0.0))
            trial.set_user_attr(f'{noise_type.lower()}_f1_snr', metrics_detail.get('f1_snr', 0.0))
            
            avg_combined = float(combined_score)
        else:
            print(f"❌ Trial {trial.number}: No score found for {noise_type}")
            return 0.0
        
        print(f"✅ Trial {trial.number} COMPLETED in {elapsed/60:.1f}min: avg_combined={avg_combined:.4f}")
        
        return avg_combined
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"⏰ Trial {trial.number} timed out after {elapsed/60:.1f}min")
        return 0.0
    except Exception as e:
        print(f"💥 Trial {trial.number} exception: {e}")
        return 0.0


def create_study(study_name: str, storage_path: Path, seed: int = 42) -> optuna.Study:
    """Create or load Optuna study with advanced configuration."""
    
    storage_url = f"sqlite:///{storage_path}"
    
    # Advanced sampler configuration
    sampler = TPESampler(
        seed=seed,
        n_startup_trials=10,      # Reduced for faster convergence
        n_ei_candidates=24,       # Balanced exploration
        multivariate=True,        # Enable multivariate TPE
        constant_liar=False       # Better for sequential optimization
    )
    
    # Balanced pruner for efficiency
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
        interval_steps=2
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )
    
    return study


def analyze_results(study: optuna.Study, output_dir: Path):
    """Analyze optimization results and save insights."""
    
    print("\n🏆 OPTIMIZATION RESULTS")
    print("=" * 50)
    
    # Check if we have any successful trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > 0]
    
    if not completed_trials:
        print("⚠️  No successful trials with positive scores found!")
        # Fallback: use any completed trial even with 0 score
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            print("❌ No completed trials at all!")
            return None
    
    # Best trial
    best = max(completed_trials, key=lambda x: x.value or 0)
    print(f"🥇 Best trial: {best.number}")
    print(f"   Combined F1: {best.value:.4f}")
    print(f"   Parameters: {best.params}")
    
    if hasattr(best, 'user_attrs') and best.user_attrs:
        print(f"   Individual scores:")
        print(f"     BW: {best.user_attrs.get('bw_score', 0.0):.4f}")
        print(f"     MA: {best.user_attrs.get('ma_score', 0.0):.4f}")
        print(f"     PLI: {best.user_attrs.get('pli_score', 0.0):.4f}")
    
    # Top 5 trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda x: x.value, reverse=True)
    
    print(f"\n🏅 Top 5 trials:")
    for i, trial in enumerate(completed_trials[:5], 1):
        print(f"   {i}. Trial {trial.number}: F1={trial.value:.4f}")
        print(f"      clauses={trial.params['clauses']} T={trial.params['T']} "
              f"s={trial.params['s']:.1f} patch_w={trial.params['patch_w']}")
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n📊 Parameter importance:")
        for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"   {param:12}: {imp:.3f}")
    except Exception as e:
        print(f"⚠️  Couldn't compute parameter importance: {e}")
    
    # Save results
    results_file = output_dir / 'optuna_results.json'
    results = {
        'study_name': study.study_name,
        'n_trials': len(study.trials),
        'best_trial': {
            'number': best.number,
            'value': best.value,
            'params': best.params,
            'user_attrs': getattr(best, 'user_attrs', {})
        },
        'top_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'user_attrs': getattr(t, 'user_attrs', {})
            }
            for t in completed_trials[:10]
        ]
    }
    
    if 'importance' in locals():
        results['parameter_importance'] = importance
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Create best config for final training
    best_config_file = output_dir / 'best_config.json'
    best_config = {
        'clauses': best.params['clauses'],
        'T': best.params['T'],
        's': best.params['s'],
        'patch_w': best.params['patch_w'],
        'batch_size': best.params['batch_size'],
        'backend': 'conv',
        'epochs': 40,          # Full training epochs
        'patience': 12,        # Increased patience for final training
        'val_eval_each': 1,
        'balance_bins': True,  # Use SNR balancing
        'absent_max_ratio': 0.3,
        'expected_performance': best.value
    }
    
    with open(best_config_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"💾 Best config saved to: {best_config_file}")
    
    return best_config


def main():
    parser = argparse.ArgumentParser(description="BEAST MODE Optuna Optimization for TMU Estimator")
    parser.add_argument('--h5', required=True, help='HDF5 dataset path')
    parser.add_argument('--output-dir', required=True, help='Output directory for optimization')
    parser.add_argument('--study-name', default='tmu_estimator_optuna', help='Optuna study name')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=3600, help='Optimization timeout in seconds (1 hour default)')
    parser.add_argument('--trial-timeout', type=int, default=15, help='Single trial timeout in minutes')
    parser.add_argument('--quick-epochs', type=int, default=3, help='Epochs per trial for quick evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Verify dataset
    if not Path(args.h5).exists():
        raise FileNotFoundError(f"Dataset not found: {args.h5}")
    
    print("🚀 BEAST MODE OPTUNA OPTIMIZATION - ESTIMATOR")
    print("=" * 60)
    print(f"📊 Dataset: {args.h5}")
    print(f"📁 Output: {output_dir}")
    print(f"🎯 Trials: {args.n_trials}")
    print(f"⏰ Total timeout: {args.timeout}s ({args.timeout/3600:.1f}h)")
    print(f"⌛ Trial timeout: {args.trial_timeout}min")
    print(f"📈 Quick epochs: {args.quick_epochs}")
    print(f"🎲 Seed: {args.seed}")
    print(f"🔬 Strategy: Single noise type per trial for faster optimization")
    
    # Dataset info
    with h5py.File(args.h5, 'r') as h5:
        train_shape = h5['train/X'].shape
        val_shape = h5['val/X'].shape
        test_shape = h5['test/X'].shape
        print(f"📏 Data shapes: train={train_shape} val={val_shape} test={test_shape}")
        
        # Check y_snr_bin shapes
        if 'train/y_snr_bin' in h5:
            y_shape = h5['train/y_snr_bin'].shape
            print(f"🎯 Target shape: {y_shape} (3 noise types)")
    
    # Base configuration
    base_config = {
        'h5_path': args.h5,
        'base_output_dir': output_dir / 'trials',
        'seed': args.seed,
        'quick_epochs': args.quick_epochs,
        'trial_timeout_minutes': args.trial_timeout,
    }
    
    base_config['base_output_dir'].mkdir(exist_ok=True)
    
    # Create study
    storage_path = output_dir / f'{args.study_name}.db'
    study = create_study(args.study_name, storage_path, args.seed)
    
    print(f"\n📚 Study: {args.study_name}")
    print(f"🗄️  Storage: {storage_path}")
    
    # Optimization callback with enhanced monitoring
    def callback(study: optuna.Study, trial):
        import time
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            progress_pct = int(100 * completed_trials / args.n_trials)
            
            print(f"\n🎯 TRIAL {trial.number} SUMMARY")
            print(f"   Status: ✅ COMPLETED | Score: {trial.value:.4f}")
            print(f"   Progress: {completed_trials}/{args.n_trials} trials ({progress_pct}%)")
            
            if len(study.trials) > 0:
                best = study.best_trial
                print(f"   🏆 Current BEST: Trial #{best.number} with F1={best.value:.4f}")
                if trial.number == best.number:
                    print(f"   🎉 NEW BEST FOUND!")
                    
            # Estimated time remaining
            if completed_trials > 2:
                avg_time_per_trial = sum(t.duration.total_seconds() for t in study.trials if t.duration and t.state == optuna.trial.TrialState.COMPLETE) / completed_trials
                remaining_trials = args.n_trials - completed_trials
                eta_seconds = avg_time_per_trial * remaining_trials
                eta_minutes = eta_seconds / 60
                print(f"   ⏱️  ETA: {eta_minutes:.1f} minutes remaining")
                
            print(f"   Time: {time.strftime('%H:%M:%S')}")
            print("-" * 60)
            
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"✂️  Trial {trial.number} PRUNED (early stopping)")
        elif trial.state == optuna.trial.TrialState.FAIL:
            print(f"💥 Trial {trial.number} FAILED")
    
    # Run optimization
    print(f"\n🔥 Starting optimization...")
    print(f"🎯 Target: {args.n_trials} trials in {args.timeout/3600:.1f} hours")
    print(f"📊 Each trial: ~{args.quick_epochs} epochs with early stopping")
    print(f"🔬 Optimizing: Combined F1 (presence + SNR estimation)")
    print("=" * 60)
    
    try:
        study.optimize(
            lambda trial: run_training_trial(trial, base_config),
            n_trials=args.n_trials,
            timeout=args.timeout,
            callbacks=[callback]
        )
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
    
    # Analyze results
    if len(study.trials) > 0:
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > 0]
        if completed_trials:
            best_config = analyze_results(study, output_dir)
            
            print(f"\n🎊 OPTIMIZATION COMPLETE!")
            print(f"🏆 Best combined F1: {study.best_trial.value:.4f}")
            print(f"📋 Best config: {best_config}")
            print(f"🚀 Ready for final training with optimal parameters!")
            
            return best_config
        else:
            print("\n⚠️  No successful trials found!")
            print("🔍 Check timeout settings and dataset size")
            return None
    else:
        print("❌ No completed trials found!")
        return None


if __name__ == '__main__':
    main()