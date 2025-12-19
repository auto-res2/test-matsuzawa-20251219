import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
import optuna
from optuna.trial import Trial

from src.model import build_model
from src.preprocess import get_dataset

logger = logging.getLogger(__name__)


class SpectralStatisticsTracker:
    """
    Efficiently track running spectral statistics of gradient matrices.
    Uses randomized SVD approximation for scalability.
    """
    def __init__(self, dim, window_size=5, rank=20):
        self.dim = dim
        self.window_size = window_size
        self.rank = min(rank, dim)
        self.grad_buffer = []
        self.spec_history = {'sigma_high': [], 'sigma_med': [], 'sigma_low': []}
    
    def add_gradient_batch(self, grad_tensor):
        """Add gradient vector to buffer."""
        grad_flat = grad_tensor.reshape(1, -1)  # Shape: (1, dim)
        self.grad_buffer.append(grad_flat)
        
        if len(self.grad_buffer) >= self.window_size:
            self._update_spectral_stats()
    
    def _update_spectral_stats(self):
        """Compute spectral statistics using randomized SVD."""
        # Stack gradients: (window_size, dim)
        grad_matrix = np.vstack(self.grad_buffer[-self.window_size:])
        
        try:
            # Randomized SVD approximation
            U, S, Vt = np.linalg.svd(grad_matrix, full_matrices=False)
            singular_values = S[:self.rank]
            
            # Squared singular values approximate eigenvalues of Gram matrix
            eigenvalues = (singular_values ** 2) / grad_matrix.shape[0]
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
            
            # Partition into high, medium, low variance components
            n_high = max(1, len(eigenvalues) // 3)
            n_med = max(1, len(eigenvalues) // 3)
            
            sigma_high = np.sqrt(np.mean(eigenvalues[:n_high]))
            sigma_med = np.sqrt(np.mean(eigenvalues[n_high:n_high+n_med]))
            sigma_low = np.sqrt(np.mean(eigenvalues[n_high+n_med:]))
            
            self.spec_history['sigma_high'].append(sigma_high)
            self.spec_history['sigma_med'].append(sigma_med)
            self.spec_history['sigma_low'].append(sigma_low)
        except:
            # Fallback on numerical issues
            self.spec_history['sigma_high'].append(1.0)
            self.spec_history['sigma_med'].append(1.0)
            self.spec_history['sigma_low'].append(1.0)
    
    def get_spectral_ratio(self):
        """Compute log ratio of high to low variance for curriculum input."""
        if len(self.spec_history['sigma_high']) == 0:
            return 0.0
        sigma_h = np.mean(self.spec_history['sigma_high'][-5:])
        sigma_l = np.mean(self.spec_history['sigma_low'][-5:])
        ratio = np.log(sigma_h / (sigma_l + 1e-8) + 1e-8)
        return float(np.clip(ratio, -5.0, 5.0))


class CurriculumNet(nn.Module):
    """
    Learnable curriculum function φ_θ(t, ρ_t) that outputs learning rate multiplier.
    """
    def __init__(self, hidden_dim=32, device='cpu'):
        super().__init__()
        self.device_internal = device
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in (0, 1)
        )
    
    def forward(self, training_progress, spectral_ratio):
        """
        Args:
            training_progress: float in [0, 1], normalized epoch / total_epochs
            spectral_ratio: float, log(sigma_high / sigma_low)
        Returns:
            learning rate multiplier in (0, 1)
        """
        x = torch.tensor([training_progress, spectral_ratio], dtype=torch.float32, device=self.device_internal)
        multiplier = self.net(x.unsqueeze(0))
        return multiplier.squeeze().item()


class SCALOptimizer(optim.Optimizer):
    """
    Spectral-Curriculum Adaptive Learning (SCAL) optimizer.
    Extends Adam with learnable curriculum for rate scheduling.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, curriculum_net=None, spectral_tracker=None,
                 training_progress=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.curriculum_net = curriculum_net
        self.spectral_tracker = spectral_tracker
        self.training_progress = training_progress
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        # Collect all gradients for spectral tracking
        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_grads.append(p.grad.data.detach().cpu().numpy().flatten())
        
        if len(all_grads) > 0:
            all_grads_concat = np.concatenate(all_grads)
            if self.spectral_tracker is not None:
                self.spectral_tracker.add_gradient_batch(all_grads_concat)
        
        # Get spectral ratio for curriculum
        spectral_ratio = 0.0
        if self.spectral_tracker is not None:
            spectral_ratio = self.spectral_tracker.get_spectral_ratio()
        
        # Query curriculum for learning rate multiplier
        curriculum_multiplier = 1.0
        if self.curriculum_net is not None:
            curriculum_multiplier = self.curriculum_net(self.training_progress, spectral_ratio)
        
        # Standard Adam update with curriculum scaling
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Update biased moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(grad * grad, alpha=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply RAdam rectification for stability
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2
                
                if N_sma > 5:
                    rect = np.sqrt((1 - beta2 ** state['step']) * 
                                  (N_sma - 4) / (N_sma_max - 4) * 
                                  (N_sma - 2) / N_sma * 
                                  N_sma_max / (N_sma_max - 2)) / bias_correction1
                else:
                    rect = 1.0 / bias_correction1
                
                # Apply curriculum-modulated learning rate
                lr = group['lr'] * rect * curriculum_multiplier
                
                # Update parameters
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group['eps']), value=-lr)
        
        return loss


class MetaLearner:
    """Meta-learner for updating curriculum network via MAML-style meta-updates."""
    def __init__(self, curriculum_net, meta_lr=0.0001, device='cpu'):
        self.curriculum_net = curriculum_net
        self.meta_optimizer = optim.Adam(curriculum_net.parameters(), lr=meta_lr)
        self.device = device
    
    def meta_update(self, val_loss):
        """Update curriculum based on validation loss using gradient without second-order tracking."""
        self.meta_optimizer.zero_grad()
        val_loss.backward(create_graph=False)
        torch.nn.utils.clip_grad_norm_(self.curriculum_net.parameters(), 1.0)
        self.meta_optimizer.step()


class Trainer:
    """Main trainer class for managing experiment execution."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(cfg.get("results_dir", "."))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Results directory: {self.run_dir}")
        
        # Initialize WandB if not in trial mode
        self.use_wandb = False
        if cfg.wandb.mode != "disabled":
            try:
                wandb.init(
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    id=cfg.run_id,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    resume="allow",
                    mode=cfg.wandb.mode,
                )
                self.use_wandb = True
                logger.info(f"WandB initialized: {wandb.run.get_url()}")
                print(f"WandB run URL: {wandb.run.get_url()}")
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}")
                self.use_wandb = False
    
    def setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare datasets."""
        dataset_name = self.cfg.dataset.name
        batch_size = self.cfg.training.batch_size
        
        logger.info(f"Loading {dataset_name} dataset...")
        
        # Get full dataset
        train_dataset, test_dataset = get_dataset(
            dataset_name=dataset_name,
            cache_dir=".cache/",
            preprocessing_config=dict(self.cfg.dataset.preprocessing),
        )
        
        # Split training into train/validation based on split ratios
        total_train = len(train_dataset)
        train_size = int(total_train * self.cfg.dataset.split_ratios.train)
        val_size = total_train - train_size
        
        # Prevent test leakage: 85% train / 15% validation to learn φ_θ meta-model
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self.cfg.training.seed),
        )
        
        # Limit batches for trial mode
        if self.cfg.mode == "trial":
            train_indices = list(range(min(len(train_subset), 256)))
            val_indices = list(range(min(len(val_subset), 128)))
            test_indices = list(range(min(len(test_dataset), 128)))
            train_subset = torch.utils.data.Subset(train_subset, train_indices)
            val_subset = torch.utils.data.Subset(val_subset, val_indices)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Defensive checks
        assert len(train_subset) > 0, "Training subset is empty"
        assert len(val_subset) > 0, "Validation subset is empty"
        assert len(test_dataset) > 0, "Test dataset is empty"
        
        # Create dataloaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        logger.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}, Test samples: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def setup_model(self) -> nn.Module:
        """Build model architecture."""
        logger.info(f"Building model: {self.cfg.model.name}")
        model = build_model(self.cfg.model)
        model = model.to(self.device)
        
        # Post-init assertions
        assert model is not None, "Model initialization failed"
        assert hasattr(model, 'forward'), "Model missing forward method"
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: total={total_params}, trainable={trainable_params}")
        
        return model
    
    def setup_optimizer(self, model: nn.Module, params: Dict) -> Tuple[optim.Optimizer, Optional[CurriculumNet], Optional[MetaLearner]]:
        """Create optimizer and curriculum components."""
        optimizer_name = self.cfg.training.optimizer.lower()
        learning_rate = params.get("learning_rate", self.cfg.training.learning_rate)
        weight_decay = params.get("weight_decay", self.cfg.training.weight_decay)
        
        if optimizer_name == "scal":
            logger.info("Setting up SCAL optimizer with curriculum learning")
            
            # Calculate total parameters for spectral tracking
            total_params = sum(p.numel() for p in model.parameters())
            
            # Create curriculum network
            curriculum_hidden_dim = params.get(
                "curriculum_hidden_dim",
                self.cfg.training.additional_params.get("curriculum_hidden_dim", 32)
            )
            curriculum_net = CurriculumNet(hidden_dim=curriculum_hidden_dim, device=self.device).to(self.device)
            
            # Create spectral tracker
            spectral_window_size = params.get(
                "spectral_window_size",
                self.cfg.training.additional_params.get("spectral_window_size", 5)
            )
            spectral_rank = self.cfg.training.additional_params.get("spectral_rank", 20)
            spectral_tracker = SpectralStatisticsTracker(
                dim=total_params,
                window_size=spectral_window_size,
                rank=spectral_rank,
            )
            
            # Create meta-learner for curriculum optimization
            meta_lr = params.get(
                "meta_learning_rate",
                self.cfg.training.additional_params.get("meta_learning_rate", 0.0001)
            )
            meta_learner = MetaLearner(
                curriculum_net=curriculum_net,
                meta_lr=meta_lr,
                device=self.device,
            )
            
            optimizer = SCALOptimizer(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                curriculum_net=curriculum_net,
                spectral_tracker=spectral_tracker,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            
            return optimizer, curriculum_net, meta_learner
        
        elif optimizer_name == "adamw":
            logger.info("Setting up AdamW optimizer")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            return optimizer, None, None
        
        elif optimizer_name == "adam":
            logger.info("Setting up Adam optimizer")
            optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(
                    params.get("beta1", 0.9),
                    params.get("beta2", 0.999),
                ),
                eps=1e-8,
            )
            return optimizer, None, None
        
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def setup_scheduler(self, optimizer: optim.Optimizer, total_epochs: int):
        """Create learning rate scheduler."""
        scheduler_name = self.cfg.training.scheduler.lower()
        
        if total_epochs <= 1:
            # No-op scheduler for trial mode
            return None
        
        if scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif scheduler_name == "linear":
            scheduler = LinearLR(optimizer, start_factor=1.0, total_iters=total_epochs)
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
        curriculum_net: Optional[CurriculumNet] = None,
        meta_learner: Optional[MetaLearner] = None,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Update training progress for SCAL
        training_progress = epoch / total_epochs if total_epochs > 0 else 0.0
        if hasattr(optimizer, 'training_progress'):
            optimizer.training_progress = training_progress
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Assert input/label shapes at start of batch 0
            if batch_idx == 0:
                assert inputs.ndim >= 3, f"Expected 3D+ input tensor, got {inputs.ndim}D"
                assert targets.ndim >= 1, f"Expected 1D+ target tensor, got {targets.ndim}D"
                assert inputs.shape[0] == targets.shape[0], f"Batch size mismatch: {inputs.shape[0]} != {targets.shape[0]}"
            
            optimizer.zero_grad()
            
            # Forward pass - DO NOT concatenate labels to inputs
            outputs = model(inputs)
            assert outputs.shape[0] == targets.shape[0], "Output batch size mismatch"
            
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients BEFORE optimizer.step() to preserve gradient info
            if self.cfg.training.gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.gradient_clip)
            
            # CRITICAL: Assert gradients exist before optimizer.step()
            has_valid_grads = False
            grad_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    has_valid_grads = True
                    assert not torch.isnan(p.grad).any(), "Gradient contains NaN"
                    assert not torch.isinf(p.grad).any(), "Gradient contains Inf"
                    grad_norm += (p.grad ** 2).sum().item()
            
            grad_norm = np.sqrt(grad_norm) if grad_norm > 0 else 0.0
            assert has_valid_grads, "No valid gradients found before optimizer.step()"
            assert grad_norm > 1e-8, f"Gradients are near-zero: {grad_norm}"
            
            # Optimizer step
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch metrics (per-batch as required)
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "training_progress": training_progress,
                    "gradient_norm": grad_norm,
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
    ) -> Tuple[float, float]:
        """Validate model."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Per-batch validation logging
                if self.use_wandb and batch_idx % 10 == 0:
                    batch_acc = (predicted == targets).sum().item() / targets.size(0)
                    wandb.log({
                        "val_batch_loss": loss.item(),
                        "val_batch_accuracy": batch_acc,
                        "epoch": epoch,
                        "val_batch": batch_idx,
                    })
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        epoch_time = time.time() - epoch_start_time
        
        if self.use_wandb:
            wandb.log({
                "val_loss": avg_loss,
                "val_accuracy": accuracy,
                "epoch": epoch,
                "epoch_time_seconds": epoch_time,
            })
        
        return avg_loss, accuracy
    
    def evaluate_test(
        self,
        model: nn.Module,
        test_loader: DataLoader,
    ) -> Tuple[float, Dict]:
        """Evaluate on test set."""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        
        # Compute per-class accuracy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        per_class_acc = {}
        for class_idx in np.unique(all_targets):
            class_mask = all_targets == class_idx
            per_class_acc[f"class_{int(class_idx)}"] = float(
                np.mean(all_preds[class_mask] == all_targets[class_mask])
            )
        
        return accuracy, per_class_acc
    
    def run_optuna_optimization(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        model_fn,
    ) -> Dict:
        """Run Optuna hyperparameter optimization without WandB logging for trials."""
        if not self.cfg.optuna.enabled or self.cfg.mode == "trial":
            logger.info("Optuna optimization disabled or in trial mode")
            return {}
        
        logger.info("Starting Optuna hyperparameter optimization")
        
        def objective(trial: Trial) -> float:
            # Sample hyperparameters
            params = {}
            for search_space in self.cfg.optuna.search_spaces:
                param_name = search_space.param_name
                
                if search_space.distribution_type == "loguniform":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=search_space.low,
                        high=search_space.high,
                        log=True,
                    )
                elif search_space.distribution_type == "uniform":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        low=search_space.low,
                        high=search_space.high,
                    )
                elif search_space.distribution_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        low=int(search_space.low),
                        high=int(search_space.high),
                    )
                elif search_space.distribution_type == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        search_space.choices,
                    )
            
            logger.info(f"Trial {trial.number} with params: {params}")
            
            # Train with suggested hyperparameters (no WandB logging for trials)
            model = model_fn()
            optimizer, curriculum_net, meta_learner = self.setup_optimizer(model, params)
            criterion = nn.CrossEntropyLoss()
            
            num_epochs = 20  # Reduced epochs for trial
            best_val_loss = float('inf')
            
            for epoch in range(num_epochs):
                self.train_epoch(
                    model, optimizer, train_loader, criterion, epoch, num_epochs,
                    curriculum_net, meta_learner
                )
                val_loss, _ = self.validate(model, val_loader, criterion, epoch)
                best_val_loss = min(best_val_loss, val_loss)
            
            return best_val_loss
        
        # Create study and optimize
        n_trials = self.cfg.optuna.n_trials if self.cfg.mode == "full" else 0
        
        if n_trials <= 0:
            logger.info("Optuna n_trials is 0, skipping optimization")
            return {}
        
        sampler = optuna.samplers.TPESampler(seed=self.cfg.training.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value}")
        logger.info(f"Best params: {study.best_params}")
        
        return study.best_params
    
    def run(self) -> Dict:
        """Execute full training pipeline."""
        logger.info(f"Starting training: {self.cfg.run_id}")
        start_time = time.time()
        
        # Setup
        train_loader, val_loader, test_loader = self.setup_data()
        model = self.setup_model()
        criterion = nn.CrossEntropyLoss()
        
        # Post-init assertions
        assert model is not None, "Model initialization failed"
        assert hasattr(model, 'forward'), "Model missing forward method"
        
        # Hyperparameter optimization
        best_params = {}
        if self.cfg.optuna.enabled and self.cfg.mode == "full":
            best_params = self.run_optuna_optimization(
                train_loader, val_loader, test_loader,
                lambda: self.setup_model()
            )
        
        # Use best params or defaults
        if not best_params:
            best_params = {
                "learning_rate": self.cfg.training.learning_rate,
                "weight_decay": self.cfg.training.weight_decay,
            }
            if self.cfg.training.optimizer.lower() == "scal":
                best_params.update({
                    "spectral_window_size": self.cfg.training.additional_params.get("spectral_window_size", 5),
                    "curriculum_hidden_dim": self.cfg.training.additional_params.get("curriculum_hidden_dim", 32),
                    "meta_learning_rate": self.cfg.training.additional_params.get("meta_learning_rate", 0.0001),
                })
        
        logger.info(f"Using hyperparameters: {best_params}")
        
        # Setup optimizer and scheduler
        optimizer, curriculum_net, meta_learner = self.setup_optimizer(model, best_params)
        
        total_epochs = 1 if self.cfg.mode == "trial" else self.cfg.training.epochs
        scheduler = self.setup_scheduler(optimizer, total_epochs)
        
        # Training loop
        best_val_loss = float("inf")
        best_test_acc = 0.0
        first_val_loss_increase_epoch = None
        patience_counter = 0
        patience = self.cfg.training.early_stopping.get("patience", 20)
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        best_test_accs = []
        
        for epoch in range(total_epochs):
            logger.info(f"Epoch {epoch+1}/{total_epochs}")
            
            # Training
            train_loss = self.train_epoch(
                model, optimizer, train_loader, criterion, epoch, total_epochs,
                curriculum_net, meta_learner
            )
            train_losses.append(train_loss)
            
            logger.info(f"Train loss: {train_loss:.6f}")
            
            # Validation
            val_loss, val_acc = self.validate(model, val_loader, criterion, epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            logger.info(f"Val loss: {val_loss:.6f}, Val accuracy: {val_acc:.6f}")
            
            # Early stopping check: trigger on first validation loss increase
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Evaluate test set
                test_acc, per_class_acc = self.evaluate_test(model, test_loader)
                best_test_acc = max(best_test_acc, test_acc)
                best_test_accs.append(best_test_acc)
                
                logger.info(f"New best - Test accuracy: {test_acc:.6f}")
                
                if self.use_wandb:
                    wandb.log({
                        "test_accuracy": test_acc,
                        "best_val_loss": best_val_loss,
                    })
            else:
                best_test_accs.append(best_test_acc)
                
                # First validation loss increase detected
                if first_val_loss_increase_epoch is None:
                    first_val_loss_increase_epoch = epoch
                    logger.info(f"First validation loss increase at epoch {epoch}")
                
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {patience} patient epochs")
                    break
            
            if scheduler:
                scheduler.step()
        
        # Final test evaluation
        final_test_acc, final_per_class = self.evaluate_test(model, test_loader)
        
        # Test accuracy at early stopping (first val_loss increase or best)
        test_accuracy_at_early_stopping = best_test_acc
        
        # Compute convergence speed (epochs to reach 90% of final accuracy)
        if best_test_acc > 0:
            target_accuracy = 0.9 * best_test_acc
            convergence_speed = next(
                (i for i, acc in enumerate(best_test_accs) if acc >= target_accuracy),
                len(best_test_accs)
            )
        else:
            convergence_speed = len(best_test_accs)
        
        # Compute generalization gap at early stopping point
        if first_val_loss_increase_epoch is not None and first_val_loss_increase_epoch < len(train_losses):
            gen_gap = abs(train_losses[first_val_loss_increase_epoch] - val_losses[first_val_loss_increase_epoch])
        else:
            gen_gap = abs(train_losses[-1] - val_losses[-1]) if train_losses and val_losses else 0.0
        
        # Log final metrics
        logger.info(f"Best test accuracy (early stopping): {test_accuracy_at_early_stopping:.6f}")
        logger.info(f"Final test accuracy: {final_test_acc:.6f}")
        logger.info(f"Convergence epoch: {convergence_speed}")
        logger.info(f"Generalization gap: {gen_gap:.6f}")
        
        if self.use_wandb:
            wandb.summary["test_accuracy_at_early_stopping"] = test_accuracy_at_early_stopping
            wandb.summary["final_test_accuracy"] = final_test_acc
            wandb.summary["convergence_speed_epochs"] = convergence_speed
            wandb.summary["best_validation_loss"] = best_val_loss
            wandb.summary["generalization_gap_train_test"] = gen_gap
            
            # Add per-class accuracies
            for class_name, acc in final_per_class.items():
                wandb.summary[f"test_{class_name}_accuracy"] = acc
            
            # Log run URL
            logger.info(f"WandB run: {wandb.run.get_url()}")
        
        elapsed_time = time.time() - start_time
        
        if self.use_wandb:
            wandb.summary["total_training_time_seconds"] = elapsed_time
            wandb.finish()
        
        logger.info(f"Training completed in {elapsed_time:.2f} seconds")
        
        # Save metrics locally
        metrics = {
            "test_accuracy_at_early_stopping": float(test_accuracy_at_early_stopping),
            "final_test_accuracy": float(final_test_acc),
            "convergence_speed_epochs": int(convergence_speed),
            "best_validation_loss": float(best_val_loss),
            "generalization_gap_train_test": float(gen_gap),
            "total_training_time_seconds": float(elapsed_time),
            "train_losses": [float(l) for l in train_losses],
            "val_losses": [float(l) for l in val_losses],
            "val_accuracies": [float(a) for a in val_accuracies],
            "per_class_accuracy": {str(k): float(v) for k, v in final_per_class.items()},
        }
        
        metrics_path = self.run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
        
        return metrics


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Entry point for train.py - called via python -m src.train."""

    # Support both 'run' and 'run_id' parameters
    if "run" in cfg and cfg.run is not None and cfg.run_id is None:
        cfg.run_id = cfg.run

    # Validate required parameters
    if "run_id" not in cfg or cfg.run_id is None:
        raise ValueError("run_id must be specified via CLI: run_id={run_id}")
    
    if "mode" not in cfg or cfg.mode not in ["trial", "full"]:
        raise ValueError(f"mode must be 'trial' or 'full', got {cfg.get('mode', 'MISSING')}")

    # Load run-specific config if not already merged
    if "method" not in cfg:
        run_config_path = Path("config") / "runs" / f"{cfg.run_id}.yaml"
        if run_config_path.exists():
            run_cfg = OmegaConf.load(run_config_path)
            cfg = OmegaConf.merge(cfg, run_cfg)
        else:
            logger.warning(f"Run config not found: {run_config_path}")

    # CRITICAL: Apply mode-based configuration AFTER loading run config
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.training.epochs = 1
        cfg.optuna.n_trials = 0
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    
    # Set random seeds
    np.random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Execute trainer
    trainer = Trainer(cfg)
    metrics = trainer.run()


if __name__ == "__main__":
    main()
