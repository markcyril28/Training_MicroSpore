"""
Tests for the ML training components.

This module tests:
- Dataset loading and preprocessing
- Move encoding/decoding
- Model architecture
- Training loop components
"""

import pytest
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from micro.types import Player, Move, Position
from micro.game_state import GameState
from micro.ai.ml.move_encoder import (
    encode_board, 
    encode_moves, 
    BOARD_PLANES, 
    MOVE_FEATURE_SIZE
)
from micro.ai.ml.dataset import (
    MicroDataset,
    CachedTensorDataset,
    preprocess_entries_to_tensors,
    collate_batch,
    collate_cached_batch,
    get_available_ram_gb,
    get_total_ram_gb,
)
from micro.ai.ml.replay import ReplayEntry


class TestMoveEncoder:
    """Tests for move encoding."""
    
    def test_encode_board_shape(self):
        """Test that board encoding has correct shape."""
        state = GameState.initial()
        encoded = encode_board(state)
        
        assert encoded.shape == (BOARD_PLANES, 8, 8)
        assert encoded.dtype == np.float32
    
    def test_encode_board_planes(self):
        """Test that board planes contain valid values."""
        state = GameState.initial()
        encoded = encode_board(state)
        
        # Values should be 0 or 1
        assert np.all((encoded >= 0) & (encoded <= 1))
        
        # Last plane (side to move) should be all 1s
        assert np.all(encoded[4] == 1.0)
    
    def test_encode_moves_shape(self):
        """Test that move encoding has correct shape."""
        state = GameState.initial()
        moves = state.legal_moves()
        
        encoded = encode_moves(state, moves)
        
        assert encoded.shape == (len(moves), MOVE_FEATURE_SIZE)
        assert encoded.dtype == np.float32


class TestDataset:
    """Tests for dataset classes."""
    
    @pytest.fixture
    def sample_entries(self):
        """Create sample replay entries for testing."""
        state = GameState.initial()
        moves = state.legal_moves()
        
        entries = []
        for i in range(10):
            entry = ReplayEntry(
                state=state.to_compact(),
                legal_moves=[m.to_dict() for m in moves[:5]],
                chosen_index=0,
                result=1,
            )
            entries.append(entry)
        
        return entries
    
    def test_micro_dataset_length(self, sample_entries):
        """Test MicroDataset length."""
        dataset = MicroDataset(sample_entries)
        assert len(dataset) == len(sample_entries)
    
    def test_micro_dataset_getitem(self, sample_entries):
        """Test MicroDataset item retrieval."""
        dataset = MicroDataset(sample_entries)
        board, move_features, target = dataset[0]
        
        assert isinstance(board, torch.Tensor)
        assert board.shape == (BOARD_PLANES, 8, 8)
        assert isinstance(move_features, torch.Tensor)
        assert isinstance(target, int)
    
    def test_preprocess_entries_to_tensors(self, sample_entries):
        """Test tensor preprocessing."""
        boards, move_features, move_counts, targets = preprocess_entries_to_tensors(
            sample_entries, max_moves_per_sample=64, show_progress=False
        )
        
        assert boards.shape == (len(sample_entries), BOARD_PLANES, 8, 8)
        assert move_features.shape == (len(sample_entries), 64, MOVE_FEATURE_SIZE)
        assert move_counts.shape == (len(sample_entries),)
        assert targets.shape == (len(sample_entries),)
    
    def test_cached_tensor_dataset(self, sample_entries):
        """Test CachedTensorDataset creation."""
        cached_dataset = CachedTensorDataset.from_entries(
            sample_entries, max_moves_per_sample=64, show_progress=False
        )
        
        assert len(cached_dataset) == len(sample_entries)
        
        # Test item retrieval
        board, move_feats, move_count, target = cached_dataset[0]
        assert board.shape == (BOARD_PLANES, 8, 8)
        assert move_feats.shape == (64, MOVE_FEATURE_SIZE)
    
    def test_collate_batch(self, sample_entries):
        """Test batch collation."""
        dataset = MicroDataset(sample_entries)
        batch = [dataset[i] for i in range(3)]
        
        boards, all_moves, move_counts, targets = collate_batch(batch)
        
        assert boards.shape[0] == 3
        assert len(move_counts) == 3
        assert len(targets) == 3
    
    def test_ram_detection(self):
        """Test RAM detection functions."""
        available = get_available_ram_gb()
        total = get_total_ram_gb()
        
        assert available >= 0
        assert total >= available


class TestModel:
    """Tests for the ML model."""
    
    def test_model_creation(self):
        """Test model creation."""
        from micro.ai.ml.model import create_model, MoveScorerNet
        
        model = create_model()
        assert isinstance(model, MoveScorerNet)
    
    def test_model_forward(self):
        """Test model forward pass."""
        from micro.ai.ml.model import create_model
        
        model = create_model()
        model.eval()
        
        batch_size = 4
        num_moves = 10
        
        boards = torch.randn(batch_size, BOARD_PLANES, 8, 8)
        move_features = torch.randn(num_moves, MOVE_FEATURE_SIZE)
        move_counts = torch.tensor([3, 2, 3, 2], dtype=torch.long)
        
        with torch.no_grad():
            scores = model(boards, move_features, move_counts)
        
        assert scores.shape == (num_moves,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
