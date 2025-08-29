#!/usr/bin/env python3
"""
ECG Noise Classifier – Improved Version
Modular Python script converted from Jupyter notebook for ECG noise classification
using Tsetlin Machines with advanced features and Optuna hyperparameter optimization.

Author: Claude Code Assistant
Date: 2025-01-24
"""

import sys
import os
import platform
import subprocess
from pathlib import Path
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Iterator, Any
from dataclasses import dataclass
import shutil
import sqlite3

# Scientific computing imports
import numpy as np
import h5py
from collections import Counter
import importlib

# ML imports
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Optuna imports
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


def check_gpu() -> bool:
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print('✅ NVIDIA GPU detected')
            return True
        else:
            print('❌ No NVIDIA GPU visible')
            return False
    except FileNotFoundError:
        print('❌ nvidia-smi not found - no GPU available')
        return False


def install_packages(has_gpu: bool = False):
    """Install required packages with error handling."""
    packages = [
        'optuna>=3.0.0',
        'h5py>=3.8.0', 
        'scikit-learn>=1.3.0',
        'wandb',  # for experiment tracking
        'pyTsetlinMachine>=0.6.0',
        'pycuda' if has_gpu else None
    ]
    
    packages = [p for p in packages if p is not None]
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade'] + packages)
        print('✅ All packages installed successfully')
    except subprocess.CalledProcessError as e:
        print(f'⚠️  Package installation warning: {e}')
        print('Continuing with available packages...')


def install_custom_tmu(drive_tmu_path: Optional[Path] = None) -> bool:
    """Install custom TMU from Drive with fallback to pip version."""
    if drive_tmu_path is None:
        drive_tmu_path = Path('/content/drive/MyDrive/Colab_Projects/denoising_ecg/lib_source/tmu')
    
    if drive_tmu_path.exists():
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '-U', 
                '--no-build-isolation', '--no-cache-dir', str(drive_tmu_path)
            ])
            print(f'✅ Custom TMU installed from: {drive_tmu_path}')
            return True
        except subprocess.CalledProcessError as e:
            print(f'⚠️  Custom TMU installation failed: {e}')
            print('Falling back to pip TMU...')
    
    # Fallback to pip TMU
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tmu>=0.8.0'])
        print('✅ TMU installed from PyPI')
        return True
    except subprocess.CalledProcessError as e:
        print(f'❌ TMU installation failed: {e}')
        return False


def test_tmu_import(has_gpu: bool) -> Optional[str]:
    """Test TMU import and report capabilities."""
    try:
        import tmu
        print(f'TMU version: {getattr(tmu, "__version__", "unknown")}')
        
        from tmu.models.classification.vanilla_classifier import TMClassifier
        print('✅ TMU TMClassifier available')
        
        # Test CUDA availability
        if has_gpu:
            try:
                test_model = TMClassifier(number_of_clauses=10, T=5, s=1.0, platform='CUDA')
                print('✅ TMU CUDA support confirmed')
                del test_model
                return 'CUDA'
            except Exception as e:
                print(f'⚠️  TMU CUDA failed: {e}. Using CPU.')
                return 'CPU'
        else:
            return 'CPU'
            
    except ImportError as e:
        print(f'❌ TMU import failed: {e}')
        return None


def mount_drive_safe():
    """Safely mount Google Drive with error handling (Colab only)."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        print('✅ Google Drive mounted successfully')
        return True
    except Exception as e:
        print(f'❌ Drive mount failed: {e}')
        return False


def validate_paths(root_path: Path) -> Dict[str, Path]:
    """Validate and return all required paths."""
    paths = {
        'root': root_path,
        'data': root_path / 'data',
        'bin': root_path / 'data' / 'bin', 
        'models': root_path / 'data' / 'models',
        'src': root_path / 'src'
    }
    
    missing_paths = []
    for name, path in paths.items():
        if not path.exists():
            missing_paths.append(f'{name}: {path}')
    
    if missing_paths:
        print('⚠️  Missing paths:')
        for missing in missing_paths:
            print(f'  - {missing}')
    
    return paths


def find_hdf5_files(bin_path: Path) -> Dict[str, Path]:
    """Find available HDF5 classifier files."""
    possible_files = [
        'cls_spec_k8.h5',
        'classifier_patches.h5', 
        'estimator_patches.h5'
    ]
    
    found_files = {}
    for filename in possible_files:
        filepath = bin_path / filename
        if filepath.exists():
            found_files[filename] = filepath
            
    return found_files


@dataclass
class TrainingConfig:
    """Centralized training configuration."""
    # Model parameters
    backend: str = 'conv'  # 'conv' or 'flat'
    clauses: int = 2400
    T: int = 1200
    s: float = 7.0
    patch_w: int = 63
    platform: str = 'auto'  # 'auto', 'CUDA', 'CPU'
    append_negated: bool = False
    
    # Training parameters
    epochs: int = 30
    patience: int = 8
    val_each: int = 1
    batch_size: int = 2048
    balanced_training: bool = True
    
    # Data parameters
    use_spectral: bool = True
    local_copy: bool = True  # Copy HDF5 to local storage for speed
    
    # Logging and output
    seed: int = 123
    verbose: int = 1  # 0=quiet, 1=normal, 2=debug
    save_model: bool = True
    experiment_name: str = 'ecg_classifier_v2'
    
    # Hyperparameter search
    optuna_trials: int = 50
    optuna_timeout: Optional[int] = 3600  # 1 hour
    quick_eval_epochs: int = 5
    
    def __post_init__(self):
        """Validate and adjust configuration."""
        # Auto-detect platform if set to auto
        if self.platform == 'auto':
            # This will be set later when TMU_PLATFORM is determined
            pass
            
        # Adjust batch size for memory constraints
        if self.platform == 'CUDA' and self.batch_size > 3072:
            print(f'⚠️  Reducing batch size from {self.batch_size} to 2048 for GPU memory')
            self.batch_size = 2048
            
        # Ensure reasonable patch size
        if self.backend == 'conv' and self.patch_w > 128:
            print(f'⚠️  Large patch_w={self.patch_w}, consider reducing for efficiency')
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to JSON."""
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'TrainingConfig':
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class ECGDatasetInfo:
    """Dataset information and validation."""
    
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        self._info = self._analyze_dataset()
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """Analyze dataset structure and content."""
        info = {'splits': {}, 'metadata': {}, 'class_info': {}}
        
        try:
            with h5py.File(self.h5_path, 'r') as h5:
                # Analyze splits
                for split in ['train', 'val', 'test']:
                    if f'{split}/X' in h5 and f'{split}/y' in h5:
                        X_shape = h5[f'{split}/X'].shape
                        y_shape = h5[f'{split}/y'].shape
                        y_sample = h5[f'{split}/y'][:100]  # Sample for analysis
                        
                        info['splits'][split] = {
                            'X_shape': X_shape,
                            'y_shape': y_shape,
                            'n_samples': X_shape[0],
                            'dtype_X': str(h5[f'{split}/X'].dtype),
                            'dtype_y': str(h5[f'{split}/y'].dtype),
                            'class_distribution': dict(Counter(y_sample))
                        }
                
                # Extract metadata if available
                if 'meta_json' in h5:
                    try:
                        meta_raw = h5['meta_json'][()]
                        if isinstance(meta_raw, (bytes, bytearray)):
                            meta_raw = meta_raw.decode('utf-8', errors='ignore')
                        info['metadata'] = json.loads(meta_raw)
                    except Exception as e:
                        print(f'⚠️  Metadata parsing failed: {e}')
                        info['metadata'] = {}
                
                # Determine number of classes and features
                if 'train' in info['splits']:
                    train_info = info['splits']['train']
                    info['n_features'] = train_info['X_shape'][1] * train_info['X_shape'][2]
                    info['feature_dims'] = (train_info['X_shape'][1], train_info['X_shape'][2])  # (C, L)
                    
                    # Determine classes from metadata or data
                    if 'train' in info['metadata'] and 'class_map' in info['metadata']['train']:
                        class_map = info['metadata']['train']['class_map']
                        info['class_info'] = {
                            'n_classes': len(class_map),
                            'class_names': list(class_map.keys()),
                            'class_map': class_map
                        }
                    else:
                        # Infer from data
                        with h5py.File(self.h5_path, 'r') as h5_inner:
                            y_all = h5_inner['train/y'][:]
                            unique_classes = sorted(np.unique(y_all))
                            info['class_info'] = {
                                'n_classes': len(unique_classes),
                                'class_names': [f'Class_{i}' for i in unique_classes],
                                'class_map': {f'Class_{i}': i for i in unique_classes}
                            }
                            
        except Exception as e:
            raise RuntimeError(f'Dataset analysis failed: {e}')
            
        return info
    
    def print_summary(self):
        """Print dataset summary."""
        print('📊 Dataset Summary:')
        print('=' * 50)
        
        for split, split_info in self._info['splits'].items():
            print(f'{split.upper():>5}: X={split_info["X_shape"]} y={split_info["y_shape"]} '
                  f'({split_info["dtype_X"]}/{split_info["dtype_y"]})')
            
            # Class distribution preview
            dist_items = list(split_info['class_distribution'].items())[:5]
            dist_str = ', '.join([f'{k}:{v}' for k, v in dist_items])
            if len(split_info['class_distribution']) > 5:
                dist_str += ', ...'
            print(f'      Class dist: {dist_str}')
        
        # Feature info
        if 'feature_dims' in self._info:
            C, L = self._info['feature_dims']
            print(f'\n📏 Features: {C} channels × {L} timepoints = {self._info["n_features"]} total')
        
        # Class info
        class_info = self._info['class_info']
        print(f'🏷️  Classes: {class_info["n_classes"]} → {class_info["class_names"]}')
        
        print('=' * 50)
    
    def get_feature_dims(self) -> Tuple[int, int]:
        """Get (channels, length) dimensions."""
        return self._info['feature_dims']
    
    def get_n_classes(self) -> int:
        """Get number of classes."""
        return self._info['class_info']['n_classes']
    
    def get_class_names(self) -> List[str]:
        """Get class names."""
        return self._info['class_info']['class_names']


class BalancedDataLoader:
    """Balanced streaming data loader for HDF5 datasets."""
    
    def __init__(self, h5_path: Path, split: str, batch_size: int, balanced: bool = True):
        self.h5_path = h5_path
        self.split = split
        self.batch_size = batch_size
        self.balanced = balanced
        
        # Get dataset info
        with h5py.File(h5_path, 'r') as h5:
            self.n_samples = h5[f'{split}/X'].shape[0]
            self.input_shape = h5[f'{split}/X'].shape[1:]
            
            if balanced:
                # Pre-compute class indices for balanced sampling
                y_all = h5[f'{split}/y'][:]
                self.classes = sorted(np.unique(y_all))
                self.class_indices = {}
                for cls in self.classes:
                    indices = np.where(y_all == cls)[0]
                    np.random.shuffle(indices)
                    self.class_indices[cls] = indices
                
                self.class_cursors = {cls: 0 for cls in self.classes}
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def _get_balanced_batch_indices(self) -> np.ndarray:
        """Get indices for balanced batch."""
        samples_per_class = max(1, self.batch_size // len(self.classes))
        batch_indices = []
        
        for cls in self.classes:
            cls_indices = self.class_indices[cls]
            cursor = self.class_cursors[cls]
            
            # Handle wraparound
            if cursor + samples_per_class > len(cls_indices):
                # Take remaining + wrap around
                remaining = cls_indices[cursor:]
                needed = samples_per_class - len(remaining)
                wrapped = cls_indices[:needed] if needed > 0 else np.array([], dtype=int)
                selected = np.concatenate([remaining, wrapped])
                
                # Reshuffle for next epoch
                np.random.shuffle(cls_indices)
                self.class_cursors[cls] = needed
            else:
                selected = cls_indices[cursor:cursor + samples_per_class]
                self.class_cursors[cls] = cursor + samples_per_class
            
            batch_indices.extend(selected)
        
        # Shuffle final batch order
        batch_indices = np.array(batch_indices)
        np.random.shuffle(batch_indices)
        return batch_indices[:self.batch_size]  # Trim to exact batch size
    
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over batches."""
        with h5py.File(self.h5_path, 'r') as h5:
            X_dataset = h5[f'{self.split}/X']
            y_dataset = h5[f'{self.split}/y']
            
            if self.balanced:
                # Use balanced sampling
                for _ in range(len(self)):
                    indices = self._get_balanced_batch_indices()
                    X_batch = X_dataset[indices]
                    y_batch = y_dataset[indices]
                    yield X_batch, y_batch
            else:
                # Sequential batches with shuffling
                indices = np.random.permutation(self.n_samples)
                for i in range(0, self.n_samples, self.batch_size):
                    batch_idx = indices[i:i + self.batch_size]
                    X_batch = X_dataset[batch_idx]
                    y_batch = y_dataset[batch_idx]
                    yield X_batch, y_batch


def copy_to_local(source_path: Path, local_name: str = 'dataset_local.h5') -> Path:
    """Copy HDF5 to local storage for faster access."""
    local_path = Path(f'/content/{local_name}')
    
    if local_path.exists() and local_path.stat().st_size == source_path.stat().st_size:
        print(f'✅ Using existing local copy: {local_path}')
        return local_path
    
    print(f'📥 Copying dataset to local storage...')
    print(f'   Source: {source_path} ({source_path.stat().st_size / 1024**2:.1f} MB)')
    print(f'   Target: {local_path}')
    
    try:
        shutil.copy2(source_path, local_path)
        print(f'✅ Local copy created successfully')
        return local_path
    except Exception as e:
        print(f'⚠️  Local copy failed: {e}. Using original path.')
        return source_path


class ModelBuilder:
    """Robust model builder with multiple fallback options."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.available_backends = self._detect_backends()
        print(f'🔧 Available backends: {", ".join(self.available_backends)}')
    
    def _detect_backends(self) -> List[str]:
        """Detect available TM backends."""
        backends = []
        
        # Test TMU
        try:
            from tmu.models.classification.vanilla_classifier import TMClassifier
            backends.append('TMU')
        except ImportError:
            pass
        
        # Test pyTsetlinMachine Convolutional
        try:
            for module_name in ['pyTsetlinMachine.tm', 'pyTsetlinMachine.cTM']:
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, 'MultiClassConvolutionalTsetlinMachine2D'):
                        backends.append('pyTM_Conv')
                        break
                except ImportError:
                    continue
        except Exception:
            pass
        
        # Test pyTsetlinMachine Flat
        try:
            from pyTsetlinMachine.tm import MultiClassTsetlinMachine
            backends.append('pyTM_Flat')
        except ImportError:
            pass
        
        if not backends:
            raise RuntimeError('No Tsetlin Machine backend available!')
            
        return backends
    
    def build_model(self) -> Tuple[Any, str, callable]:
        """Build model with automatic fallback. Returns (model, backend_used, input_preprocessor)."""
        C, L = self.config.feature_dims
        
        # Try TMU first if available and requested
        if 'TMU' in self.available_backends and self.config.platform in ['CUDA', 'CPU']:
            try:
                model, preprocessor = self._build_tmu(C, L)
                return model, 'TMU', preprocessor
            except Exception as e:
                print(f'⚠️  TMU build failed: {e}. Trying fallback...')
        
        # Try pyTM Convolutional
        if 'pyTM_Conv' in self.available_backends and self.config.backend == 'conv':
            try:
                model, preprocessor = self._build_pytm_conv(C, L)
                return model, 'pyTM_Conv', preprocessor
            except Exception as e:
                print(f'⚠️  pyTM Conv build failed: {e}. Trying fallback...')
        
        # Try pyTM Flat (always works as last resort)
        if 'pyTM_Flat' in self.available_backends:
            try:
                model, preprocessor = self._build_pytm_flat(C, L)
                return model, 'pyTM_Flat', preprocessor
            except Exception as e:
                print(f'❌ pyTM Flat build failed: {e}')
        
        raise RuntimeError('All model building attempts failed!')
    
    def _build_tmu(self, C: int, L: int) -> Tuple[Any, callable]:
        """Build TMU model."""
        from tmu.models.classification.vanilla_classifier import TMClassifier
        
        kwargs = {
            'number_of_clauses': self.config.clauses,
            'T': self.config.T,
            's': self.config.s,
            'platform': self.config.platform
        }
        
        if self.config.backend == 'conv':
            kwargs['patch_dim'] = (C, min(self.config.patch_w, L))
            preprocessor = self._conv_preprocessor
        else:
            preprocessor = self._flat_preprocessor
        
        # Try with dropout first (newer TMU versions)
        try:
            kwargs.update({
                'clause_drop_p': 0.05,
                'literal_drop_p': 0.05
            })
            model = TMClassifier(**kwargs)
        except TypeError:
            # Fallback: weighted clauses
            kwargs.pop('clause_drop_p', None)
            kwargs.pop('literal_drop_p', None)
            try:
                kwargs['weighted_clauses'] = True
                model = TMClassifier(**kwargs)
            except TypeError:
                # Basic TMU
                kwargs.pop('weighted_clauses', None)
                model = TMClassifier(**kwargs)
        
        if hasattr(model, 'initialize'):
            model.initialize()
        
        print(f'✅ TMU model built: clauses={self.config.clauses}, T={self.config.T}, s={self.config.s}')
        print(f'   Platform: {self.config.platform}, Backend: {self.config.backend}')
        
        return model, preprocessor
    
    def _build_pytm_conv(self, C: int, L: int) -> Tuple[Any, callable]:
        """Build pyTsetlinMachine Conv2D model."""
        for module_name in ['pyTsetlinMachine.tm', 'pyTsetlinMachine.cTM']:
            try:
                module = importlib.import_module(module_name)
                Conv2D = getattr(module, 'MultiClassConvolutionalTsetlinMachine2D')
                
                model = Conv2D(
                    number_of_clauses=self.config.clauses,
                    T=self.config.T,
                    s=self.config.s,
                    patch_dim=(C, min(self.config.patch_w, L)),
                    append_negated=self.config.append_negated
                )
                
                print(f'✅ pyTM Conv2D model built from {module_name}')
                return model, self._conv_preprocessor
                
            except Exception:
                continue
        
        raise ImportError('No pyTM Conv2D available')
    
    def _build_pytm_flat(self, C: int, L: int) -> Tuple[Any, callable]:
        """Build pyTsetlinMachine Flat model."""
        from pyTsetlinMachine.tm import MultiClassTsetlinMachine
        
        model = MultiClassTsetlinMachine(
            number_of_clauses=self.config.clauses,
            T=self.config.T,
            s=self.config.s,
            append_negated=self.config.append_negated
        )
        
        print(f'✅ pyTM Flat model built')
        return model, self._flat_preprocessor
    
    @staticmethod
    def _conv_preprocessor(X: np.ndarray) -> np.ndarray:
        """Preprocessor for convolutional models."""
        return np.ascontiguousarray(X.astype(np.uint32, copy=False))
    
    @staticmethod
    def _flat_preprocessor(X: np.ndarray) -> np.ndarray:
        """Preprocessor for flat models."""
        X_flat = X.reshape(X.shape[0], -1)
        return np.ascontiguousarray(X_flat.astype(np.uint32, copy=False))
    
    @staticmethod
    def supports_state_api(model: Any) -> bool:
        """Check if model supports get_state/set_state."""
        return hasattr(model, 'get_state') and hasattr(model, 'set_state')


class TrainingLogger:
    """Structured training logger with recovery capability."""
    
    def __init__(self, log_dir: Path, experiment_name: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_file = log_dir / f'{experiment_name}_training.jsonl'
        self.metrics_file = log_dir / f'{experiment_name}_metrics.json'
        self.checkpoint_file = log_dir / f'{experiment_name}_checkpoint.json'
        
        # Training state
        self.epoch_logs = []
        self.best_metrics = {'epoch': -1, 'val_f1': -1.0, 'val_acc': -1.0}
        self.training_start = None
        
        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'{experiment_name}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(experiment_name)
    
    def start_training(self, config: TrainingConfig):
        """Log training start."""
        self.training_start = time.time()
        
        start_log = {
            'event': 'training_start',
            'timestamp': datetime.now().isoformat(),
            'config': config.to_dict()
        }
        
        self._write_log(start_log)
        self.logger.info(f'Training started: {self.experiment_name}')
    
    def log_epoch(self, epoch: int, train_time: float, val_acc: float, val_f1: float, 
                  additional_metrics: Optional[Dict] = None):
        """Log epoch results."""
        epoch_log = {
            'event': 'epoch_complete',
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'train_time_s': train_time,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'elapsed_total_s': time.time() - self.training_start if self.training_start else 0
        }
        
        if additional_metrics:
            epoch_log.update(additional_metrics)
        
        self.epoch_logs.append(epoch_log)
        self._write_log(epoch_log)
        
        # Update best metrics
        if val_f1 > self.best_metrics['val_f1']:
            self.best_metrics.update({
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc
            })
            self.logger.info(f'🎯 New best F1: {val_f1:.4f} (epoch {epoch})')
        
        # Save checkpoint
        self.save_checkpoint(epoch)
    
    def log_early_stop(self, epoch: int, reason: str):
        """Log early stopping."""
        stop_log = {
            'event': 'early_stop',
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'reason': reason,
            'best_metrics': self.best_metrics
        }
        
        self._write_log(stop_log)
        self.logger.info(f'Early stopping at epoch {epoch}: {reason}')
    
    def log_final_results(self, test_acc: float, test_f1: float, confusion_mat: np.ndarray,
                         classification_rep: str):
        """Log final test results."""
        final_log = {
            'event': 'training_complete',
            'timestamp': datetime.now().isoformat(),
            'test_acc': test_acc,
            'test_f1': test_f1,
            'total_training_time_s': time.time() - self.training_start if self.training_start else 0,
            'total_epochs': len(self.epoch_logs),
            'best_metrics': self.best_metrics
        }
        
        self._write_log(final_log)
        
        # Save comprehensive metrics
        final_metrics = {
            'experiment_name': self.experiment_name,
            'training_summary': final_log,
            'epoch_history': self.epoch_logs,
            'confusion_matrix': confusion_mat.tolist(),
            'classification_report': classification_rep
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        self.logger.info(f'Training completed. Final test F1: {test_f1:.4f}')
        self.logger.info(f'Results saved to: {self.metrics_file}')
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint = {
            'experiment_name': self.experiment_name,
            'epoch': epoch,
            'best_metrics': self.best_metrics,
            'epoch_logs': self.epoch_logs,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load training checkpoint if exists."""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f'Failed to load checkpoint: {e}')
        return None
    
    def _write_log(self, log_entry: Dict):
        """Write log entry to JSONL file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class GPUMonitor:
    """GPU monitoring utilities."""
    
    @staticmethod
    def get_gpu_stats() -> Optional[Dict[str, Union[int, float]]]:
        """Get current GPU statistics."""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                line = result.stdout.strip().split('\n')[0]  # First GPU
                values = [int(x.strip()) for x in line.split(',')]
                
                return {
                    'memory_used_mb': values[0],
                    'memory_total_mb': values[1],
                    'utilization_percent': values[2],
                    'temperature_c': values[3]
                }
        except Exception:
            pass
            
        return None
    
    @staticmethod
    def format_gpu_stats(stats: Optional[Dict]) -> str:
        """Format GPU stats for display."""
        if stats is None:
            return 'GPU: N/A'
        
        return (f"GPU: {stats['utilization_percent']:3d}% "
                f"{stats['memory_used_mb']}/{stats['memory_total_mb']} MB "
                f"{stats['temperature_c']:2d}°C")


def evaluate_model(model, data_loader, input_preprocessor, class_names: List[str]) -> Dict[str, Any]:
    """Comprehensive model evaluation."""
    y_true_all = []
    y_pred_all = []
    
    eval_start = time.time()
    
    for X_batch, y_batch in data_loader:
        X_processed = input_preprocessor(X_batch)
        y_pred = model.predict(X_processed)
        
        y_true_all.append(y_batch)
        y_pred_all.append(y_pred)
    
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    
    eval_time = time.time() - eval_start
    
    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm,
        'classification_report': class_report,
        'eval_time_s': eval_time,
        'n_samples': len(y_true)
    }


def train_model_advanced(config: TrainingConfig, dataset_path: Path, 
                        output_dir: Path) -> Dict[str, Any]:
    """Advanced training with all improvements."""
    
    # Set seeds for reproducibility
    np.random.seed(config.seed)
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    
    # Setup logging
    logger = TrainingLogger(output_dir, config.experiment_name)
    logger.start_training(config)
    
    # Check for existing checkpoint
    checkpoint = logger.load_checkpoint()
    start_epoch = 1
    if checkpoint and config.verbose > 0:
        print(f'📂 Found checkpoint from epoch {checkpoint["epoch"]}')
        print(f'   Best F1: {checkpoint["best_metrics"]["val_f1"]:.4f}')
        # Note: Model state recovery would require additional implementation
        # start_epoch = checkpoint['epoch'] + 1
    
    # Create data loaders
    train_loader = BalancedDataLoader(dataset_path, 'train', config.batch_size, config.balanced_training)
    val_loader = BalancedDataLoader(dataset_path, 'val', config.batch_size, balanced=False)
    test_loader = BalancedDataLoader(dataset_path, 'test', config.batch_size, balanced=False)
    
    # Build model
    builder = ModelBuilder(config)
    model, backend_used, input_preprocessor = builder.build_model()
    
    logger.logger.info(f'Model built: {backend_used} - {type(model).__name__}')
    
    # Warmup model with sample from each class
    if config.verbose > 0:
        print('🔥 Warming up model...')
    
    warmup_samples = []
    warmup_labels = []
    
    with h5py.File(dataset_path, 'r') as h5:
        y_train = h5['train/y'][:]
        for cls in range(config.n_classes):
            class_indices = np.where(y_train == cls)[0]
            if len(class_indices) >= 2:
                selected_idx = class_indices[:2]
                warmup_samples.extend(selected_idx)
                warmup_labels.extend([cls, cls])
        
        if warmup_samples:
            X_warmup = h5['train/X'][warmup_samples]
            y_warmup = np.array(warmup_labels, dtype=np.uint32)
            
            model.fit(input_preprocessor(X_warmup), y_warmup, epochs=1, incremental=True)
            _ = model.predict(input_preprocessor(X_warmup))  # Test prediction
    
    # Training variables
    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    patience_left = config.patience
    
    supports_state = builder.supports_state_api(model)
    
    if config.verbose > 0:
        print(f'\n🚀 Starting training for {config.epochs} epochs')
        print(f'   Batch size: {config.batch_size} (balanced: {config.balanced_training})')
        print(f'   Early stopping patience: {config.patience}')
        print(f'   State API support: {supports_state}')
        print('=' * 60)
    
    # Main training loop
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_start_time = time.time()
        
        # Training
        batch_count = 0
        samples_processed = 0
        batch_times = []
        
        for X_batch, y_batch in train_loader:
            batch_start = time.time()
            
            # Ensure uint32 for labels (TMU requirement)
            y_batch_uint32 = y_batch.astype(np.uint32, copy=False)
            X_processed = input_preprocessor(X_batch)
            
            model.fit(X_processed, y_batch_uint32, epochs=1, incremental=True)
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            batch_count += 1
            samples_processed += len(X_batch)
            
            # Progress reporting
            if config.verbose > 1 and (batch_count % max(1, len(train_loader) // 5) == 0):
                avg_batch_time = np.mean(batch_times[-10:])
                samples_per_sec = len(X_batch) / batch_time
                gpu_stats = GPUMonitor.get_gpu_stats()
                
                progress = batch_count / len(train_loader) * 100
                eta_seconds = (len(train_loader) - batch_count) * avg_batch_time
                
                print(f'    Batch {batch_count:3d}/{len(train_loader)} '
                      f'({progress:5.1f}%) | '
                      f'{samples_per_sec:6.1f} samp/s | '
                      f'ETA: {eta_seconds:5.1f}s | '
                      f'{GPUMonitor.format_gpu_stats(gpu_stats)}')
        
        train_time = time.time() - epoch_start_time
        avg_samples_per_sec = samples_processed / train_time
        
        # Validation every val_each epochs
        if epoch % config.val_each == 0:
            val_metrics = evaluate_model(model, val_loader, input_preprocessor, config.class_names)
            val_acc, val_f1 = val_metrics['accuracy'], val_metrics['f1_macro']
            
            # Additional metrics for logging
            additional_metrics = {
                'avg_samples_per_sec': avg_samples_per_sec,
                'val_f1_weighted': val_metrics['f1_weighted'],
                'val_f1_per_class': val_metrics['f1_per_class']
            }
            
            gpu_stats = GPUMonitor.get_gpu_stats()
            if gpu_stats:
                additional_metrics['gpu_stats'] = gpu_stats
            
            # Log epoch results
            logger.log_epoch(epoch, train_time, val_acc, val_f1, additional_metrics)
            
            # Progress display
            if config.verbose > 0:
                gpu_info = GPUMonitor.format_gpu_stats(gpu_stats)
                print(f'[EPOCH {epoch:3d}] '
                      f'acc={val_acc:.4f} f1={val_f1:.4f} | '
                      f'{train_time:6.1f}s ({avg_samples_per_sec:6.0f} samp/s) | '
                      f'{gpu_info}')
                
                # Show per-class F1 if verbose
                if config.verbose > 1:
                    f1_per_class_str = ', '.join([f'{f1:.3f}' for f1 in val_metrics['f1_per_class']])
                    print(f'         Per-class F1: [{f1_per_class_str}]')
            
            # Early stopping logic
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                patience_left = config.patience
                
                # Save best model state if supported
                if supports_state:
                    try:
                        best_state = model.get_state()
                    except Exception as e:
                        logger.logger.warning(f'Failed to save model state: {e}')
                        best_state = None
                
                if config.verbose > 0:
                    print(f'         🎯 New best F1: {best_f1:.4f} (patience reset)')
            else:
                patience_left -= 1
                if config.verbose > 0:
                    print(f'         ⏳ No improvement. Patience left: {patience_left}')
                
                if patience_left <= 0:
                    logger.log_early_stop(epoch, f'No improvement for {config.patience} epochs')
                    if config.verbose > 0:
                        print(f'\n⏹️  Early stopping at epoch {epoch}')
                    break
        else:
            # Training-only epoch
            gpu_stats = GPUMonitor.get_gpu_stats()
            if config.verbose > 0:
                gpu_info = GPUMonitor.format_gpu_stats(gpu_stats)
                print(f'[EPOCH {epoch:3d}] '
                      f'training only | '
                      f'{train_time:6.1f}s ({avg_samples_per_sec:6.0f} samp/s) | '
                      f'{gpu_info}')
    
    # Restore best model if state API is supported
    if best_state is not None and supports_state:
        try:
            model.set_state(best_state)
            logger.logger.info(f'Restored best model from epoch {best_epoch} (F1: {best_f1:.4f})')
            if config.verbose > 0:
                print(f'\n🔄 Restored best model from epoch {best_epoch} (F1: {best_f1:.4f})')
        except Exception as e:
            logger.logger.warning(f'Failed to restore best model state: {e}')
    
    # Final evaluation
    if config.verbose > 0:
        print('\n📊 Final Evaluation:')
        print('=' * 60)
    
    val_metrics = evaluate_model(model, val_loader, input_preprocessor, config.class_names)
    test_metrics = evaluate_model(model, test_loader, input_preprocessor, config.class_names)
    
    if config.verbose > 0:
        print(f'Validation: acc={val_metrics["accuracy"]:.4f} f1={val_metrics["f1_macro"]:.4f}')
        print(f'Test:       acc={test_metrics["accuracy"]:.4f} f1={test_metrics["f1_macro"]:.4f}')
        print('\nPer-class Test F1 Scores:')
        for i, (class_name, f1_score) in enumerate(zip(config.class_names, test_metrics['f1_per_class'])):
            print(f'  {class_name:10}: {f1_score:.4f}')
        
        print(f'\nConfusion Matrix:')
        print(test_metrics['confusion_matrix'])
        print(f'\nDetailed Classification Report:')
        print(test_metrics['classification_report'])
    
    # Log final results
    logger.log_final_results(
        test_metrics['accuracy'], 
        test_metrics['f1_macro'],
        test_metrics['confusion_matrix'],
        test_metrics['classification_report']
    )
    
    # Save model if requested
    if config.save_model:
        try:
            model_path = output_dir / f'{config.experiment_name}_final_model.pkl'
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.logger.info(f'Model saved to: {model_path}')
            if config.verbose > 0:
                print(f'\n💾 Model saved to: {model_path}')
        except Exception as e:
            logger.logger.warning(f'Failed to save model: {e}')
    
    return {
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'training_epochs': len(logger.epoch_logs),
        'best_epoch': best_epoch,
        'best_val_f1': best_f1,
        'backend_used': backend_used,
        'config': config.to_dict()
    }


class OptunaOptimizer:
    """Bayesian hyperparameter optimization with Optuna."""
    
    def __init__(self, base_config: TrainingConfig, dataset_path: Path, 
                 output_dir: Path, study_name: str = None):
        self.base_config = base_config
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.study_name = study_name or f'{base_config.experiment_name}_optuna'
        
        # Create database path for persistent study
        self.db_path = output_dir / f'{self.study_name}.db'
        
        # Quick evaluation configuration
        self.quick_config = TrainingConfig(
            # Copy base config but with reduced training
            **{k: v for k, v in base_config.to_dict().items()},
            epochs=base_config.quick_eval_epochs,
            patience=max(2, base_config.quick_eval_epochs // 2),
            verbose=0  # Quiet during optimization
        )
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        
        # Suggest hyperparameters
        suggested_params = {
            'clauses': trial.suggest_int('clauses', 1000, 5000, step=200),
            'T': trial.suggest_int('T', 400, 2000, step=100),
            's': trial.suggest_float('s', 3.0, 12.0, step=0.5),
        }
        
        # Backend-specific parameters
        if self.base_config.backend == 'conv':
            suggested_params['patch_w'] = trial.suggest_int('patch_w', 15, 127, step=8)
        
        # Batch size optimization
        if self.base_config.platform == 'CUDA':
            suggested_params['batch_size'] = trial.suggest_categorical(
                'batch_size', [1024, 1536, 2048, 3072]
            )
        else:
            suggested_params['batch_size'] = trial.suggest_categorical(
                'batch_size', [512, 1024, 1536]
            )
        
        # Create trial configuration
        trial_config = TrainingConfig(
            **{k: v for k, v in self.quick_config.to_dict().items()},
            **suggested_params,
            experiment_name=f'{self.study_name}_trial_{trial.number}'
        )
        
        try:
            # Run quick training
            results = train_model_advanced(trial_config, self.dataset_path, self.output_dir)
            
            val_f1 = results['val_metrics']['f1_macro']
            val_acc = results['val_metrics']['accuracy']
            
            # Log additional metrics for analysis
            trial.set_user_attr('val_accuracy', val_acc)
            trial.set_user_attr('val_f1_weighted', results['val_metrics']['f1_weighted'])
            trial.set_user_attr('training_epochs', results['training_epochs'])
            trial.set_user_attr('best_epoch', results['best_epoch'])
            trial.set_user_attr('backend_used', results['backend_used'])
            
            # Report intermediate value for pruning
            trial.report(val_f1, results['training_epochs'])
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return val_f1
            
        except Exception as e:
            print(f'❌ Trial {trial.number} failed: {e}')
            # Return poor score for failed trials
            return 0.0
    
    def optimize(self, n_trials: int = None, timeout: int = None) -> optuna.Study:
        """Run Bayesian optimization."""
        
        n_trials = n_trials or self.base_config.optuna_trials
        timeout = timeout or self.base_config.optuna_timeout
        
        # Create or load study
        storage_url = f'sqlite:///{self.db_path}'
        
        sampler = TPESampler(seed=self.base_config.seed, n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
            direction='maximize',
            sampler=sampler,
            pruner=pruner
        )
        
        print(f'\n🔍 Starting Optuna optimization:')
        print(f'   Study: {self.study_name}')
        print(f'   Target: {n_trials} trials, {timeout}s timeout')
        print(f'   Quick eval: {self.quick_config.epochs} epochs')
        print(f'   Storage: {storage_url}')
        print('=' * 60)
        
        # Add callback for progress reporting
        def callback(study: optuna.Study, trial: optuna.FrozenTrial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                print(f'Trial {trial.number:3d}: '
                      f'F1={trial.value:.4f} | '
                      f'clauses={trial.params["clauses"]} '
                      f'T={trial.params["T"]} '
                      f's={trial.params["s"]:.1f}')
                
                if len(study.trials) > 0:
                    best_trial = study.best_trial
                    print(f'         Best so far: F1={best_trial.value:.4f} '
                          f'(Trial {best_trial.number})')
            elif trial.state == optuna.trial.TrialState.PRUNED:
                print(f'Trial {trial.number:3d}: PRUNED')
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[callback]
        )
        
        # Results summary
        print('\n🏆 Optimization completed!')
        print(f'   Total trials: {len(study.trials)}')
        print(f'   Best F1: {study.best_trial.value:.4f}')
        print(f'   Best params: {study.best_params}')
        
        # Save results
        self.save_optimization_results(study)
        
        return study
    
    def save_optimization_results(self, study: optuna.Study):
        """Save optimization results and analysis."""
        
        results_file = self.output_dir / f'{self.study_name}_results.json'
        
        # Prepare results data
        results = {
            'study_name': self.study_name,
            'n_trials': len(study.trials),
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_trial.value,
                'params': study.best_trial.params,
                'user_attrs': study.best_trial.user_attrs
            },
            'optimization_history': [
                {
                    'trial': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name,
                    'user_attrs': trial.user_attrs
                }
                for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
            ]
        }
        
        # Add parameter importance analysis
        try:
            importance = optuna.importance.get_param_importances(study)
            results['parameter_importance'] = importance
        except Exception:
            results['parameter_importance'] = {}
        
        # Save to JSON
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\n💾 Optimization results saved to: {results_file}')
        
        # Create best config for final training
        best_config = TrainingConfig(
            **{k: v for k, v in self.base_config.to_dict().items()},
            **study.best_params,
            experiment_name=f'{self.base_config.experiment_name}_best_optuna'
        )
        
        best_config_file = self.output_dir / f'{self.study_name}_best_config.json'
        best_config.save(best_config_file)
        
        print(f'💾 Best config saved to: {best_config_file}')
        
        return best_config
    
    def get_optimization_plots(self, study: optuna.Study):
        """Generate optimization plots (if visualization packages available)."""
        try:
            import optuna.visualization as vis
            
            plots = {
                'optimization_history': vis.plot_optimization_history(study),
                'param_importances': vis.plot_param_importances(study),
                'parallel_coordinate': vis.plot_parallel_coordinate(study)
            }
            
            return plots
            
        except ImportError:
            print('⚠️  Optuna visualization not available. Install with: pip install optuna[visualization]')
            return None


def run_full_experiment(dataset_path: Path, output_dir: Path, 
                       enable_optuna: bool = True, 
                       optuna_trials: int = 25,
                       optuna_timeout: int = 1800,
                       tmu_platform: Optional[str] = None) -> Dict[str, Any]:
    """Run complete experiment with hyperparameter optimization and final training."""
    
    # Analyze dataset
    dataset_info = ECGDatasetInfo(dataset_path)
    dataset_info.print_summary()
    
    # Create configuration
    config = TrainingConfig(
        platform=tmu_platform or 'CPU',
        experiment_name=f'ecg_classifier_{dataset_path.stem}'
    )
    
    # Update config with dataset info
    C, L = dataset_info.get_feature_dims()
    config.n_classes = dataset_info.get_n_classes()
    config.feature_dims = (C, L)
    config.class_names = dataset_info.get_class_names()
    
    # Auto-adjust platform in post_init
    config.__post_init__()
    
    print('📋 Training Configuration:')
    print('=' * 40)
    for key, value in config.to_dict().items():
        print(f'{key:20}: {value}')
    print('=' * 40)
    
    # Create experiment output directory
    experiment_output_dir = output_dir / config.experiment_name
    experiment_output_dir.mkdir(exist_ok=True)
    
    # Save initial configuration
    initial_config_path = experiment_output_dir / 'initial_config.json'
    config.save(initial_config_path)
    
    print(f'🎯 Starting Hyperparameter Optimization Experiment')
    print(f'   Experiment: {config.experiment_name}')
    print(f'   Dataset: {dataset_path.name}')
    print(f'   Output: {experiment_output_dir}')
    print(f'   Platform: {config.platform}')
    
    best_config = config
    
    # Hyperparameter optimization
    if enable_optuna:
        print(f'\n🚀 Running Bayesian optimization with {optuna_trials} trials...')
        
        optimizer = OptunaOptimizer(
            base_config=config,
            dataset_path=dataset_path,
            output_dir=experiment_output_dir,
            study_name=f'{config.experiment_name}_hyperopt'
        )
        
        study = optimizer.optimize(
            n_trials=optuna_trials,
            timeout=optuna_timeout
        )
        
        # Get best configuration
        best_config = optimizer.save_optimization_results(study)
        
        # Display parameter importance
        try:
            importance = optuna.importance.get_param_importances(study)
            print('\n📊 Parameter Importance:')
            for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                print(f'   {param:12}: {imp:.3f}')
        except Exception:
            pass
        
        # Show top 5 trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        completed_trials.sort(key=lambda x: x.value, reverse=True)
        
        print('\n🏆 Top 5 Trials:')
        for i, trial in enumerate(completed_trials[:5], 1):
            print(f'   {i}. F1={trial.value:.4f} | '
                  f'clauses={trial.params["clauses"]} '
                  f'T={trial.params["T"]} '
                  f's={trial.params["s"]:.1f}')
            if 'patch_w' in trial.params:
                print(f'      patch_w={trial.params["patch_w"]} '
                      f'batch_size={trial.params["batch_size"]}')
    
    else:
        print('⏭️  Skipping Optuna optimization')
    
    print('\n✅ Hyperparameter optimization phase completed')
    
    # Final training configuration
    final_config = TrainingConfig(
        **{k: v for k, v in best_config.to_dict().items()},
        epochs=40,           # Longer training for final model
        patience=10,         # More patience for final training
        verbose=2,           # Detailed logging
        experiment_name=f'{config.experiment_name}_final'
    )
    
    print(f'\n🎯 Final Training Configuration:')
    print('=' * 50)
    for key, value in final_config.to_dict().items():
        if key in ['clauses', 'T', 's', 'patch_w', 'batch_size', 'epochs', 'patience']:
            print(f'  {key:15}: {value}')
    print('=' * 50)
    
    # Run final training
    print('\n🚀 Starting final training with optimized parameters...')
    
    final_results = train_model_advanced(
        config=final_config,
        dataset_path=dataset_path,
        output_dir=experiment_output_dir
    )
    
    # Results summary
    print('\n' + '='*80)
    print('🏆 FINAL RESULTS SUMMARY')
    print('='*80)
    
    test_metrics = final_results['test_metrics']
    val_metrics = final_results['val_metrics']
    
    print(f'📈 Test Performance:')
    print(f'   Accuracy:    {test_metrics["accuracy"]:.4f}')
    print(f'   F1-Macro:    {test_metrics["f1_macro"]:.4f}')
    print(f'   F1-Weighted: {test_metrics["f1_weighted"]:.4f}')
    
    print(f'\n📊 Validation Performance:')
    print(f'   Accuracy:    {val_metrics["accuracy"]:.4f}')
    print(f'   F1-Macro:    {val_metrics["f1_macro"]:.4f}')
    print(f'   F1-Weighted: {val_metrics["f1_weighted"]:.4f}')
    
    print(f'\n🔧 Training Details:')
    print(f'   Backend:     {final_results["backend_used"]}')
    print(f'   Epochs:      {final_results["training_epochs"]}')
    print(f'   Best Epoch:  {final_results["best_epoch"]}')
    print(f'   Best Val F1: {final_results["best_val_f1"]:.4f}')
    
    print(f'\n🏷️  Per-Class Test F1 Scores:')
    for class_name, f1_score in zip(final_config.class_names, test_metrics['f1_per_class']):
        print(f'   {class_name:12}: {f1_score:.4f}')
    
    print(f'\n📊 Test Confusion Matrix:')
    cm = test_metrics['confusion_matrix']
    print('     ', end='')
    for i, name in enumerate(final_config.class_names):
        print(f'{name:>8}', end='')
    print()
    for i, (name, row) in enumerate(zip(final_config.class_names, cm)):
        print(f'{name:>5}', end='')
        for val in row:
            print(f'{val:>8}', end='')
        print()
    
    # Save final summary
    summary_file = experiment_output_dir / 'experiment_summary.json'
    experiment_summary = {
        'experiment_name': config.experiment_name,
        'dataset': dataset_path.name,
        'final_results': {
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_f1_weighted': test_metrics['f1_weighted'],
            'test_f1_per_class': test_metrics['f1_per_class']
        },
        'training_info': {
            'backend_used': final_results['backend_used'],
            'total_epochs': final_results['training_epochs'],
            'best_epoch': final_results['best_epoch'],
            'best_val_f1': final_results['best_val_f1']
        },
        'final_config': final_config.to_dict()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(experiment_summary, f, indent=2)
    
    print(f'\n💾 Experiment summary saved to: {summary_file}')
    print(f'📁 All results available in: {experiment_output_dir}')
    print('\n🎉 Experiment completed successfully!')
    
    return {
        'experiment_summary': experiment_summary,
        'output_dir': experiment_output_dir,
        'final_results': final_results
    }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ECG Noise Classifier - Improved Version')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to HDF5 dataset file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--no-optuna', action='store_true',
                        help='Skip Optuna hyperparameter optimization')
    parser.add_argument('--optuna-trials', type=int, default=25,
                        help='Number of Optuna trials (default: 25)')
    parser.add_argument('--optuna-timeout', type=int, default=1800,
                        help='Optuna timeout in seconds (default: 1800)')
    parser.add_argument('--local-copy', action='store_true',
                        help='Copy dataset to local storage for speed')
    
    args = parser.parse_args()
    
    # Setup
    print('Python:', sys.version)
    print('Platform:', platform.platform())
    
    has_gpu = check_gpu()
    install_packages(has_gpu)
    
    # Try to install and test TMU
    if install_custom_tmu():
        tmu_platform = test_tmu_import(has_gpu)
    else:
        tmu_platform = None
    
    print(f'Final TMU Platform: {tmu_platform}')
    
    # Paths
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f'Dataset not found: {dataset_path}')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy to local if requested
    working_dataset_path = dataset_path
    if args.local_copy:
        working_dataset_path = copy_to_local(dataset_path, f'{dataset_path.stem}_local.h5')
    
    # Run experiment
    results = run_full_experiment(
        dataset_path=working_dataset_path,
        output_dir=output_dir,
        enable_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
        optuna_timeout=args.optuna_timeout,
        tmu_platform=tmu_platform
    )
    
    print(f'\n✨ Experiment completed successfully!')
    print(f'📁 Results saved in: {results["output_dir"]}')
    
    return results


if __name__ == '__main__':
    main()