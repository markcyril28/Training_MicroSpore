"""Training loop for the move scorer model."""

import os
import sys
import json
import time
import argparse
import tempfile
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

# Enable TF32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

# Suppress torch.compile inductor warnings
warnings.filterwarnings('ignore', message='.*TensorFloat32.*')
warnings.filterwarnings('ignore', message='.*max_autotune_gemm.*')

from .model import MoveScorerNet, create_model, save_model, load_model
from .replay import ReplayBuffer
from .selfplay import SelfPlayRunner
from .dataset import create_dataloader, prepare_training_data


def parse_duration(duration_str: Optional[str]) -> Optional[datetime]:
    """Parse duration string and return the stop time.
    
    Supported formats:
    - Nd or Ndays (e.g., 2d, 2days) - N days
    - Nh or Nhours (e.g., 4h, 4hours) - N hours  
    - Nm or Nmin (e.g., 30m, 30min) - N minutes
    - Combined (e.g., 1d12h, 2d6h30m) - multiple units
    """
    if not duration_str:
        return None
    
    duration_str = duration_str.strip().lower()
    
    import re
    from datetime import timedelta
    
    total_seconds = 0
    
    # Match patterns like 2d, 4h, 30m
    patterns = [
        (r'(\d+)\s*d(?:ays?)?', 86400),   # days
        (r'(\d+)\s*h(?:ours?)?', 3600),    # hours
        (r'(\d+)\s*m(?:in(?:utes?)?)?', 60),  # minutes
        (r'(\d+)\s*s(?:ec(?:onds?)?)?', 1),   # seconds
    ]
    
    matched_any = False
    for pattern, multiplier in patterns:
        for match in re.finditer(pattern, duration_str):
            total_seconds += int(match.group(1)) * multiplier
            matched_any = True
    
    if not matched_any:
        # Try parsing as plain number (assume hours)
        try:
            hours = float(duration_str)
            total_seconds = int(hours * 3600)
            matched_any = True
        except ValueError:
            pass
    
    if not matched_any or total_seconds <= 0:
        raise ValueError(f"Could not parse duration '{duration_str}'. Use format like: 2d, 4h, 30m, 1d12h, 2days, etc.")
    
    stop_time = datetime.now() + timedelta(seconds=total_seconds)
    return stop_time


@dataclass
class TrainingStats:
    """Training statistics tracking."""
    start_time: str = ""
    end_time: str = ""
    total_steps: int = 0
    epochs_completed: int = 0
    best_loss: float = float('inf')
    best_val_loss: float = float('inf')
    
    # History lists (stored as lists for JSON serialization)
    loss_history: list = None
    val_loss_history: list = None
    lr_history: list = None
    gpu_mem_history: list = None
    step_times: list = None
    test_history: list = None  # Model vs algorithm test results
    
    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.val_loss_history is None:
            self.val_loss_history = []
        if self.lr_history is None:
            self.lr_history = []
        if self.gpu_mem_history is None:
            self.gpu_mem_history = []
        if self.step_times is None:
            self.step_times = []
        if self.test_history is None:
            self.test_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_steps': self.total_steps,
            'epochs_completed': self.epochs_completed,
            'best_loss': self.best_loss,
            'best_val_loss': self.best_val_loss,
            'loss_history': self.loss_history,
            'val_loss_history': self.val_loss_history,
            'lr_history': self.lr_history,
            'gpu_mem_history': self.gpu_mem_history,
            'step_times': self.step_times,
            'test_history': self.test_history,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingStats':
        """Create from dictionary."""
        return cls(
            start_time=data.get('start_time', ''),
            end_time=data.get('end_time', ''),
            total_steps=data.get('total_steps', 0),
            epochs_completed=data.get('epochs_completed', 0),
            best_loss=data.get('best_loss', float('inf')),
            best_val_loss=data.get('best_val_loss', float('inf')),
            loss_history=data.get('loss_history', []),
            val_loss_history=data.get('val_loss_history', []),
            lr_history=data.get('lr_history', []),
            gpu_mem_history=data.get('gpu_mem_history', []),
            step_times=data.get('step_times', []),
            test_history=data.get('test_history', []),
        )


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Device settings
    device: str = 'cuda'
    amp: bool = True
    compile_model: bool = True

    # Self-play settings
    cpu_workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 2)))
    selfplay_games: int = 500
    selfplay_difficulties: list = field(default_factory=lambda: ['medium'])  # List of difficulties to cycle
    selfplay_focus_side: str = 'both'  # white, black, both
    selfplay_opponent_focus: str = 'both'  # ml, algorithm, both
    selfplay_noise_prob: float = 0.1  # Probability of random move for exploration
    selfplay_max_moves: int = 200  # Maximum moves per game

    # Training settings
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5  # Weight decay for regularization
    train_steps: int = 999999999  # Default to essentially indefinite
    checkpoint_every: int = 1000

    # DataLoader settings
    dataloader_workers: int = field(default_factory=lambda: max(2, (os.cpu_count() or 2)))
    pin_memory: bool = True

    # Stability
    grad_clip_norm: Optional[float] = 1.0

    # Model testing settings
    test_vs_algo: bool = True
    test_every: int = 5000  # Run tests every N steps
    test_games: int = 50  # Number of test games per evaluation
    test_difficulty: str = 'medium'

    # Paths
    checkpoint_dir: str = 'models/checkpoints'
    latest_path: str = 'models/latest.pt'
    replay_dir: str = 'data/replay'
    log_dir: str = 'logs'
    stats_file: str = 'models/training_stats.json'

    # Resume
    resume: Optional[str] = None

    # Time-based stopping
    stop_time: Optional[datetime] = None  # Calculated from train_duration


class Trainer:
    """
    Trainer for the move scorer model.

    Supports:
    - Self-play data generation
    - Imitation learning from algorithmic AI
    - Checkpoint saving and resume
    - Mixed precision training
    - IPC control for GUI integration
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Set up device
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("ERROR: CUDA requested but not available.")
            print("\nTroubleshooting:")
            print("  1. Ensure NVIDIA GPU driver is installed on Windows")
            print("  2. Run 'nvidia-smi' to verify GPU access in WSL")
            print("  3. Reinstall PyTorch with CUDA support")
            sys.exit(1)

        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Set thread count for CPU operations
        cpu_threads = max(1, (os.cpu_count() or 1))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(min(4, cpu_threads))
        except Exception:
            pass

        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = create_model()
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scaler = GradScaler() if config.amp else None

        self.replay_buffer = ReplayBuffer(config.replay_dir)

        # Training state
        self.step = 0
        self.best_loss = float('inf')
        self.epoch = 0

        # Control flags for IPC
        self._paused = False
        self._stopped = False

        # Training statistics
        self.stats = TrainingStats()
        self._load_stats()
        
        # Timing for step tracking
        self._last_step_time = None

        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(config.log_dir) / f"train_{timestamp}.jsonl"

        # Resume if specified
        if config.resume:
            self._load_checkpoint(config.resume)

        # Compile after loading checkpoint to avoid state_dict key mismatch
        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                print("Enabled torch.compile for the model")
            except Exception as e:
                print(f"torch.compile unavailable: {e}")

    def _has_non_finite_tensors(self) -> bool:
        """Check if model parameters or buffers contain NaN/Inf."""
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"  Non-finite parameter detected: {name}")
                return True
        for name, buf in self.model.named_buffers():
            if not torch.isfinite(buf).all():
                print(f"  Non-finite buffer detected: {name}")
                return True
        return False

    def _reset_model_state(self, reason: str) -> None:
        """Reset model/optimizer if checkpoint is corrupt."""
        print(f"WARNING: {reason}. Resetting model and optimizer state.")
        self.model = create_model()
        self.model.to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scaler = GradScaler() if self.config.amp else None
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def _load_checkpoint(self, path: str) -> None:
        """Load training state from checkpoint."""
        print(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Handle state_dict from compiled models (torch.compile adds "_orig_mod." prefix)
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint.get('step', 0)
        self.best_loss = checkpoint.get('loss', float('inf'))

        # Restore RNG state if available
        if 'rng_state' in checkpoint:
            rng = checkpoint['rng_state']
            if 'torch' in rng:
                # Ensure RNG state is a ByteTensor
                rng_state = rng['torch']
                if not isinstance(rng_state, torch.ByteTensor):
                    rng_state = rng_state.to(torch.uint8)
                torch.set_rng_state(rng_state.cpu())
            if 'cuda' in rng and torch.cuda.is_available():
                cuda_rng = rng['cuda']
                if not isinstance(cuda_rng, torch.ByteTensor):
                    cuda_rng = cuda_rng.to(torch.uint8)
                torch.cuda.set_rng_state(cuda_rng.cpu())

        print(f"Resumed at step {self.step}")

        if self._has_non_finite_tensors():
            self._reset_model_state("Loaded checkpoint contains NaN/Inf")

    def _load_stats(self) -> None:
        """Load training stats from file if exists, and merge in any missing test history from log files."""
        stats_path = Path(self.config.stats_file)
        if stats_path.exists():
            try:
                with open(stats_path, 'r') as f:
                    data = json.load(f)
                self.stats = TrainingStats.from_dict(data)
                print(f"Loaded training stats: {self.stats.total_steps} previous steps")
            except Exception as e:
                print(f"Could not load stats: {e}")
                self.stats = TrainingStats()
        
        # Merge any missing test results from JSONL log files
        self._merge_test_history_from_logs()

    def _merge_test_history_from_logs(self) -> None:
        """Merge test_vs_algo entries from JSONL log files into stats.test_history."""
        log_dir = Path(self.config.log_dir)
        if not log_dir.exists():
            return
        
        # Build a set of existing steps to avoid duplicates
        existing_steps = {entry.get('step') for entry in self.stats.test_history}
        
        # Find all training log files
        log_files = sorted(log_dir.glob("train_*.jsonl"))
        merged_count = 0
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get('type') == 'test_vs_algo':
                                step = entry.get('step')
                                if step is not None and step not in existing_steps:
                                    # Remove the 'type' field before adding to test_history
                                    entry_copy = {k: v for k, v in entry.items() if k != 'type'}
                                    self.stats.test_history.append(entry_copy)
                                    existing_steps.add(step)
                                    merged_count += 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
        
        if merged_count > 0:
            # Sort by step
            self.stats.test_history.sort(key=lambda x: x.get('step', 0))
            print(f"Merged {merged_count} test entries from log files into stats")

    def _save_stats(self) -> None:
        """Save training stats to file."""
        stats_path = Path(self.config.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.stats.total_steps = self.step
        self.stats.end_time = datetime.now().isoformat()
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)

    def _record_step_stats(self, loss: float, lr: float) -> None:
        """Record statistics for a training step."""
        current_time = time.time()
        
        # Record loss
        self.stats.loss_history.append({
            'step': self.step,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        })
        
        # Record learning rate
        self.stats.lr_history.append({
            'step': self.step,
            'lr': lr
        })
        
        # Record GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6
            self.stats.gpu_mem_history.append({
                'step': self.step,
                'gpu_mem_mb': gpu_mem
            })
        
        # Record step time
        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
            self.stats.step_times.append({
                'step': self.step,
                'time_sec': step_time
            })
        self._last_step_time = current_time
        
        # Update best loss
        if loss < self.stats.best_loss:
            self.stats.best_loss = loss

    def _record_validation_stats(self, val_loss: float) -> None:
        """Record validation statistics."""
        self.stats.val_loss_history.append({
            'step': self.step,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        })
        
        if val_loss < self.stats.best_val_loss:
            self.stats.best_val_loss = val_loss

    def _save_checkpoint(self, loss: float) -> str:
        """Save a checkpoint atomically."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'loss': loss,
            'epoch': self.epoch,
            'rng_state': {
                'torch': torch.get_rng_state(),
            }
        }

        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state()

        # Save to temp file then rename (atomic)
        checkpoint_path = Path(self.config.checkpoint_dir) / f"model_step_{self.step:06d}.pt"

        with tempfile.NamedTemporaryFile(delete=False, dir=self.config.checkpoint_dir) as tmp:
            torch.save(checkpoint, tmp.name)
            tmp_path = tmp.name

        os.replace(tmp_path, checkpoint_path)

        # Update latest.pt
        latest_path = Path(self.config.latest_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(delete=False, dir=latest_path.parent) as tmp:
            torch.save(checkpoint, tmp.name)
            tmp_path = tmp.name

        os.replace(tmp_path, latest_path)

        # Save stats alongside checkpoint
        self._save_stats()

        print(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)

    def _log(self, data: Dict[str, Any]) -> None:
        """Log training metrics."""
        data['timestamp'] = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def run_selfplay(self, num_games: int, callback=None) -> int:
        """Run self-play to generate training data."""
        total_games = max(0, num_games)
        opponent_focus = self.config.selfplay_opponent_focus
        side_focus = self.config.selfplay_focus_side

        if opponent_focus == 'ml':
            ml_self_games = total_games
            vs_algo_games = 0
        elif opponent_focus == 'algorithm':
            ml_self_games = 0
            vs_algo_games = total_games
        else:
            ml_self_games = total_games // 2
            vs_algo_games = total_games - ml_self_games

        if opponent_focus == 'ml':
            focus_desc = "ML self-play"
        elif opponent_focus == 'algorithm':
            focus_desc = "ML vs algorithm"
        else:
            focus_desc = "half ML self-play, half vs algorithm"

        print(
            f"\nGenerating {num_games} self-play games "
            f"({focus_desc}; focus side: {side_focus})..."
        )

        # Save current model for self-play inference
        temp_model_path = Path(self.config.checkpoint_dir) / "temp_selfplay_model.pt"
        temp_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': self.step,
        }, temp_model_path)

        completed_total = 0

        def make_progress(base: int):
            def _progress(completed, _total):
                overall = base + completed
                if callback:
                    callback(overall, total_games)
                if overall % 50 == 0:
                    print(f"  Games: {overall}/{total_games}")
            return _progress

        entries = 0
        difficulties = self.config.selfplay_difficulties
        num_difficulties = len(difficulties)

        def get_difficulty_for_batch(batch_idx: int) -> str:
            """Cycle through difficulties for each batch of games."""
            diff = difficulties[batch_idx % num_difficulties]
            # 'self' means ML self-play, use 'medium' as base difficulty
            return 'medium' if diff == 'self' else diff

        def is_self_play_batch(batch_idx: int) -> bool:
            """Check if this batch should be ML self-play."""
            return difficulties[batch_idx % num_difficulties] == 'self'

        # Log difficulties being used
        print(f"  Cycling through difficulties: {difficulties}")

        if ml_self_games > 0:
            # For ML self-play, use medium difficulty as base
            runner = SelfPlayRunner(
                replay_buffer=self.replay_buffer,
                num_workers=self.config.cpu_workers,
                difficulty='medium',
                max_moves=self.config.selfplay_max_moves,
                noise_prob=self.config.selfplay_noise_prob,
                p1_policy='ml',
                p2_policy='ml',
                model_path=str(temp_model_path),
                device=self.device,
            )
            entries += runner.run_games(ml_self_games, callback=make_progress(completed_total))
            completed_total += ml_self_games

        if vs_algo_games > 0:
            if side_focus == 'white':
                ml_as_p1 = vs_algo_games
                ml_as_p2 = 0
            elif side_focus == 'black':
                ml_as_p1 = 0
                ml_as_p2 = vs_algo_games
            else:
                ml_as_p1 = vs_algo_games // 2
                ml_as_p2 = vs_algo_games - ml_as_p1

            # Split games across difficulties (excluding 'self' which is handled separately)
            algo_difficulties = [d for d in difficulties if d != 'self']
            if not algo_difficulties:
                algo_difficulties = ['medium']

            if ml_as_p1 > 0:
                games_per_diff = ml_as_p1 // len(algo_difficulties)
                remainder = ml_as_p1 % len(algo_difficulties)
                
                for i, diff in enumerate(algo_difficulties):
                    games_this_diff = games_per_diff + (1 if i < remainder else 0)
                    if games_this_diff > 0:
                        runner = SelfPlayRunner(
                            replay_buffer=self.replay_buffer,
                            num_workers=self.config.cpu_workers,
                            difficulty=diff,
                            max_moves=self.config.selfplay_max_moves,
                            noise_prob=self.config.selfplay_noise_prob,
                            p1_policy='ml',
                            p2_policy='algorithmic',
                            model_path=str(temp_model_path),
                            device=self.device,
                        )
                        entries += runner.run_games(games_this_diff, callback=make_progress(completed_total))
                        completed_total += games_this_diff

            if ml_as_p2 > 0:
                games_per_diff = ml_as_p2 // len(algo_difficulties)
                remainder = ml_as_p2 % len(algo_difficulties)
                
                for i, diff in enumerate(algo_difficulties):
                    games_this_diff = games_per_diff + (1 if i < remainder else 0)
                    if games_this_diff > 0:
                        runner = SelfPlayRunner(
                            replay_buffer=self.replay_buffer,
                            num_workers=self.config.cpu_workers,
                            difficulty=diff,
                            max_moves=self.config.selfplay_max_moves,
                            noise_prob=self.config.selfplay_noise_prob,
                            p1_policy='algorithmic',
                            p2_policy='ml',
                            model_path=str(temp_model_path),
                            device=self.device,
                        )
                        entries += runner.run_games(games_this_diff, callback=make_progress(completed_total))
                        completed_total += games_this_diff

        print(f"Generated {entries} training entries")
        return entries

    def run_test_vs_algo(self, num_games: int = None) -> Dict[str, Any]:
        """
        Run test games between current model and algorithmic AI.
        
        Args:
            num_games: Number of test games (defaults to config)
        
        Returns:
            Test statistics dictionary
        """
        if num_games is None:
            num_games = self.config.test_games
        
        print(f"\nRunning {num_games} test games vs algorithm ({self.config.test_difficulty})...")
        
        from .model_vs_algo import ModelVsAlgoTester
        
        # Save current model temporarily for testing
        temp_path = Path(self.config.checkpoint_dir) / "temp_test_model.pt"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'step': self.step,
        }, temp_path)
        
        try:
            tester = ModelVsAlgoTester(
                model_path=str(temp_path),
                algo_difficulty=self.config.test_difficulty,
                num_workers=min(self.config.cpu_workers, 4),
                max_moves=self.config.selfplay_max_moves,
            )
            
            def progress(completed, total, stats):
                if completed % 10 == 0:
                    print(f"  Test games: {completed}/{total} (ML: {stats.ml_win_rate*100:.1f}%)")
            
            stats = tester.run_tests(num_games=num_games, callback=progress)
            
            # Record in training stats
            test_record = {
                'step': self.step,
                'epoch': self.epoch,
                'total_games': stats.total_games,
                'ml_wins': stats.ml_wins,
                'algo_wins': stats.algo_wins,
                'draws': stats.draws,
                'ml_win_rate': stats.ml_win_rate,
                'ml_as_p1_win_rate': stats.ml_as_p1_win_rate,
                'ml_as_p2_win_rate': stats.ml_as_p2_win_rate,
                'avg_game_length': stats.avg_game_length,
                'timestamp': datetime.now().isoformat(),
            }
            self.stats.test_history.append(test_record)
            
            print(f"  ML Win Rate: {stats.ml_win_rate*100:.1f}%")
            print(f"    As P1 (White): {stats.ml_as_p1_win_rate*100:.1f}%")
            print(f"    As P2 (Black): {stats.ml_as_p2_win_rate*100:.1f}%")
            
            # Log to JSONL
            self._log({
                'type': 'test_vs_algo',
                **test_record
            })
            
            # Save stats to JSON file immediately so plot_training.py can read them
            self._save_stats()
            
            return test_record
        
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

    def train_epoch(self, dataloader) -> float:
        """Train for one epoch, returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for boards, move_features, move_counts, targets in dataloader:
            if self._stopped:
                break

            while self._paused:
                time.sleep(0.1)
                if self._stopped:
                    break

            # Move to device
            boards = boards.to(self.device)
            move_features = move_features.to(self.device)
            move_counts = move_counts.to(self.device)
            targets = targets.to(self.device)

            if not torch.isfinite(boards).all() or not torch.isfinite(move_features).all():
                print("  Warning: non-finite inputs detected; skipping batch")
                continue

            self.optimizer.zero_grad()

            if self.config.amp and self.scaler is not None:
                with autocast(device_type='cuda'):
                    scores = self.model(boards, move_features, move_counts)
                    if not torch.isfinite(scores).all():
                        print("  Warning: non-finite scores detected; skipping batch")
                        continue
                    loss = self._compute_loss(scores, move_counts, targets)

                if not torch.isfinite(loss):
                    print("  Warning: non-finite loss detected; skipping batch")
                    continue

                self.scaler.scale(loss).backward()

                if self.config.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                scores = self.model(boards, move_features, move_counts)
                if not torch.isfinite(scores).all():
                    print("  Warning: non-finite scores detected; skipping batch")
                    continue
                loss = self._compute_loss(scores, move_counts, targets)
                if not torch.isfinite(loss):
                    print("  Warning: non-finite loss detected; skipping batch")
                    continue
                loss.backward()
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.step += 1

            # Record step stats every 10 steps (to avoid excessive memory usage)
            if self.step % 10 == 0:
                self._record_step_stats(loss.item(), self.config.learning_rate)

            # Checkpoint
            if self.step % self.config.checkpoint_every == 0:
                avg_loss = total_loss / num_batches
                self._save_checkpoint(avg_loss)

                # Log metrics
                gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                self._log({
                    'step': self.step,
                    'loss': avg_loss,
                    'lr': self.config.learning_rate,
                    'gpu_mem_mb': gpu_mem,
                })

            # Progress
            if self.step % 100 == 0:
                print(f"  Step {self.step}, Loss: {loss.item():.4f}")

            if self.step >= self.config.train_steps:
                break

        return total_loss / max(num_batches, 1)

    def _compute_loss(
        self,
        scores: torch.Tensor,
        move_counts: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for move selection.

        The model outputs one score per move. For each position,
        we want to maximize the score of the chosen move.
        """
        batch_size = move_counts.shape[0]
        total_loss = scores.new_zeros(())
        offset = 0
        valid_count = 0

        for i in range(batch_size):
            num_moves = move_counts[i].item()
            if num_moves <= 0:
                continue
            position_scores = scores[offset:offset + num_moves].float()
            target_idx = int(targets[i].item())

            if target_idx < 0 or target_idx >= num_moves:
                offset += num_moves
                continue

            # Cross-entropy over the legal moves
            log_probs = torch.log_softmax(position_scores, dim=0)
            loss = -log_probs[target_idx]
            total_loss += loss
            valid_count += 1

            offset += num_moves

        if valid_count == 0:
            return scores.sum() * 0

        return total_loss / valid_count

    def train(self) -> None:
        """Run the full training loop."""
        print("\n" + "=" * 50)
        print("Filipino Micro - ML Training")
        print("=" * 50)

        # Set start time for stats
        if not self.stats.start_time:
            self.stats.start_time = datetime.now().isoformat()

        # Generate initial self-play data if needed
        entry_count = self.replay_buffer.count_entries()
        if entry_count < self.config.batch_size * 10:
            print(f"\nInsufficient training data ({entry_count} entries)")
            self.run_selfplay(self.config.selfplay_games)

        # Prepare data
        print("\nPreparing training data...")
        train_entries, val_entries = prepare_training_data(self.replay_buffer)
        print(f"Training: {len(train_entries)}, Validation: {len(val_entries)}")

        if not train_entries:
            print("ERROR: No training data available")
            return

        # Create dataloader
        dataloader = create_dataloader(
            train_entries,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_workers,
            pin_memory=self.config.pin_memory,
        )

        # Training loop
        print(f"\nStarting training from step {self.step}...")
        start_time = time.time()
        
        # Track last test step
        last_test_step = 0
        loss = 0.0  # Initialize loss in case loop doesn't run

        while self.step < self.config.train_steps:
            if self._stopped:
                break
            
            # Check if stop time has been reached
            if self.config.stop_time and datetime.now() >= self.config.stop_time:
                print(f"\nStop time reached ({self.config.stop_time.strftime('%Y-%m-%d %H:%M')}). Saving and exiting...")
                break

            loss = self.train_epoch(dataloader)
            self.epoch += 1
            self.stats.epochs_completed = self.epoch
            print(f"\nEpoch {self.epoch} complete. Avg Loss: {loss:.4f}")

            # Periodic self-play to refresh data
            if self.step % 5000 == 0 and self.step > 0:
                self.run_selfplay(self.config.selfplay_games // 2)
                train_entries, _ = prepare_training_data(self.replay_buffer)
                dataloader = create_dataloader(
                    train_entries,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=self.config.dataloader_workers,
                    pin_memory=self.config.pin_memory,
                )
            
            # Periodic model testing vs algorithm
            if (self.config.test_vs_algo and 
                self.step > 0 and 
                self.step - last_test_step >= self.config.test_every):
                try:
                    self.run_test_vs_algo()
                    last_test_step = self.step
                except Exception as e:
                    print(f"Test vs algorithm failed: {e}")

        # Final checkpoint
        self._save_checkpoint(loss)
        
        # Final test vs algorithm
        if self.config.test_vs_algo:
            try:
                print("\nRunning final model evaluation...")
                self.run_test_vs_algo(num_games=self.config.test_games * 2)
            except Exception as e:
                print(f"Final test failed: {e}")

        elapsed = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"  Total steps: {self.step}")
        print(f"  Epochs: {self.epoch}")
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Final model: {self.config.latest_path}")
        
        # Print test summary
        if self.stats.test_history:
            latest_test = self.stats.test_history[-1]
            print(f"  Final ML Win Rate: {latest_test.get('ml_win_rate', 0)*100:.1f}%")

    def pause(self) -> None:
        """Pause training."""
        self._paused = True

    def resume(self) -> None:
        """Resume training."""
        self._paused = False

    def stop(self) -> None:
        """Stop training."""
        self._stopped = True
        self._paused = False

    @property
    def is_paused(self) -> bool:
        return self._paused

    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        gpu_mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        
        # Get recent loss from history
        recent_loss = None
        if self.stats.loss_history:
            recent_loss = self.stats.loss_history[-1].get('loss')
        
        return {
            'step': self.step,
            'epoch': self.epoch,
            'paused': self._paused,
            'device': str(self.device),
            'gpu_mem_mb': gpu_mem,
            'recent_loss': recent_loss,
            'best_loss': self.stats.best_loss if self.stats.best_loss != float('inf') else None,
        }

    def get_stats(self) -> TrainingStats:
        """Get the full training statistics."""
        return self.stats


def list_checkpoints(checkpoint_dir: str = 'models/checkpoints') -> list:
    """List available checkpoints sorted by step."""
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return []
    
    checkpoints = []
    for f in checkpoint_path.glob('model_step_*.pt'):
        try:
            step = int(f.stem.split('_')[-1])
            checkpoints.append({
                'path': str(f),
                'step': step,
                'name': f.name,
            })
        except ValueError:
            continue
    
    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])
    return checkpoints


def load_training_stats(stats_file: str = 'models/training_stats.json') -> Optional[TrainingStats]:
    """Load training statistics from file."""
    stats_path = Path(stats_file)
    if not stats_path.exists():
        return None
    
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)
        return TrainingStats.from_dict(data)
    except Exception:
        return None


def main():
    """Main entry point for command-line training."""
    parser = argparse.ArgumentParser(description='Train Filipino Micro ML model')

    # Device settings
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to train on')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--compile-model', action='store_true',
                       help='Use torch.compile for faster training')

    # Self-play settings
    parser.add_argument('--cpu-workers', type=int, default=10,
                       help='Number of parallel self-play workers')
    parser.add_argument('--selfplay-games', type=int, default=500,
                       help='Number of self-play games per iteration')
    parser.add_argument('--focus-side', type=str, default='both',
                       choices=['white', 'black', 'both'],
                       help='Which side to focus on during self-play vs algorithm')
    parser.add_argument('--opponent-focus', type=str, default='both',
                       choices=['ml', 'algorithm', 'both'],
                       help='Opponent type to focus on during self-play')
    parser.add_argument('--selfplay-difficulties', type=str, default='medium',
                       help='Comma-separated difficulties to cycle: easy,medium,hard,self')
    parser.add_argument('--noise-prob', type=float, default=0.1,
                       help='Probability of random move for exploration (0.0 to 1.0)')
    parser.add_argument('--max-moves', type=int, default=200,
                       help='Maximum moves per game before declaring draw')

    # Training settings
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay for regularization (0 to disable)')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--train-steps', type=int, default=10000,
                       help='Total training steps')
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                       help='Steps between checkpoints')

    # DataLoader settings
    parser.add_argument('--dataloader-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--pin-memory', action='store_true',
                       help='Pin memory for faster GPU transfer')

    # Model testing settings
    parser.add_argument('--test-vs-algo', action='store_true',
                       help='Enable periodic testing against algorithm')
    parser.add_argument('--test-every', type=int, default=5000,
                       help='Steps between model tests')
    parser.add_argument('--test-games', type=int, default=50,
                       help='Number of test games per evaluation')
    parser.add_argument('--test-difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard'],
                       help='Algorithm difficulty for testing')

    # Resume settings
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--resume-latest', action='store_true',
                       help='Resume from the latest checkpoint in models/checkpoints/')
    parser.add_argument('--train-duration', type=str, default=None,
                       help='Train for this duration (e.g., 2d, 4h, 30m, 1d12h)')

    args = parser.parse_args()

    # Handle --resume-latest
    resume_path = args.resume
    if args.resume_latest:
        import glob
        import re
        checkpoint_dir = 'models/checkpoints'
        pattern = os.path.join(checkpoint_dir, 'model_step_*.pt')
        checkpoints = glob.glob(pattern)
        if checkpoints:
            # Sort by step number to find the latest
            def get_step(path):
                match = re.search(r'model_step_(\d+)\.pt$', path)
                return int(match.group(1)) if match else 0
            checkpoints.sort(key=get_step)
            resume_path = checkpoints[-1]
            print(f'Resuming from latest checkpoint: {resume_path}')
        else:
            print('No checkpoints found in models/checkpoints/, starting fresh.')
            resume_path = None

    # Parse train duration
    stop_time = parse_duration(args.train_duration) if args.train_duration else None
    if stop_time:
        print(f'Training duration: {args.train_duration}')
        print(f'Training will stop at: {stop_time.strftime("%Y-%m-%d %H:%M:%S")}')

    config = TrainingConfig(
        # Device settings
        device=args.device,
        amp=not args.no_amp,
        compile_model=args.compile_model,
        # Self-play settings
        cpu_workers=args.cpu_workers,
        selfplay_games=args.selfplay_games,
        selfplay_focus_side=args.focus_side,
        selfplay_opponent_focus=args.opponent_focus,
        selfplay_difficulties=[d.strip() for d in args.selfplay_difficulties.split(',')],
        selfplay_noise_prob=args.noise_prob,
        selfplay_max_moves=args.max_moves,
        # Training settings
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm if args.grad_clip_norm > 0 else None,
        train_steps=args.train_steps,
        checkpoint_every=args.checkpoint_every,
        # DataLoader settings
        dataloader_workers=args.dataloader_workers,
        pin_memory=args.pin_memory,
        # Model testing settings
        test_vs_algo=args.test_vs_algo,
        test_every=args.test_every,
        test_games=args.test_games,
        test_difficulty=args.test_difficulty,
        # Resume settings
        resume=resume_path,
        stop_time=stop_time,
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
