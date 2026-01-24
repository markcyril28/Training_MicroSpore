"""Dataset for training the move scorer model."""

import random
from typing import List, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from .replay import ReplayBuffer, ReplayEntry
from .move_encoder import encode_board, encode_moves, MOVE_FEATURE_SIZE, BOARD_PLANES


class MicroDataset(Dataset):
    """
    PyTorch dataset for training data.

    Each item returns:
    - board: (BOARD_PLANES, 8, 8) tensor
    - move_features: (num_moves, MOVE_FEATURE_SIZE) tensor
    - target: int (index of chosen move)
    """

    def __init__(self, entries: List[ReplayEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        entry = self.entries[idx]

        # Reconstruct game state
        state = GameState.from_compact(entry.state)

        # Encode board
        board = encode_board(state)

        # Reconstruct moves and encode them
        moves = [Move.from_dict(m) for m in entry.legal_moves]
        move_features = encode_moves(state, moves)

        return (
            torch.from_numpy(board),
            torch.from_numpy(move_features),
            entry.chosen_index,
        )


class StreamingMicroDataset(IterableDataset):
    """
    Streaming dataset that reads from replay files on-the-fly.

    More memory efficient for large datasets.
    """

    def __init__(self, replay_buffer: ReplayBuffer, shuffle: bool = True):
        self.replay_buffer = replay_buffer
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        for entry in self.replay_buffer.iterate_entries(shuffle_files=self.shuffle):
            # Reconstruct game state
            state = GameState.from_compact(entry.state)

            # Encode board
            board = encode_board(state)

            # Reconstruct moves and encode them
            moves = [Move.from_dict(m) for m in entry.legal_moves]
            move_features = encode_moves(state, moves)

            yield (
                torch.from_numpy(board),
                torch.from_numpy(move_features),
                entry.chosen_index,
            )


def collate_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader.

    Returns:
    - boards: (batch_size, BOARD_PLANES, 8, 8)
    - all_move_features: (total_moves, MOVE_FEATURE_SIZE)
    - move_counts: (batch_size,) - number of moves per sample
    - targets: (batch_size,) - index of chosen move for each sample
    """
    boards = []
    all_move_features = []
    move_counts = []
    targets = []

    for board, move_features, target in batch:
        boards.append(board)
        all_move_features.append(move_features)
        move_counts.append(move_features.shape[0])
        targets.append(target)

    return (
        torch.stack(boards),
        torch.cat(all_move_features, dim=0),
        torch.tensor(move_counts, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def create_dataloader(
    entries: List[ReplayEntry],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader from replay entries."""
    dataset = MicroDataset(entries)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def create_streaming_dataloader(
    replay_buffer: ReplayBuffer,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a streaming DataLoader from a replay buffer."""
    dataset = StreamingMicroDataset(replay_buffer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def prepare_training_data(
    replay_buffer: ReplayBuffer,
    max_entries: int = 100000,
    val_split: float = 0.1,
) -> Tuple[List[ReplayEntry], List[ReplayEntry]]:
    """
    Prepare training and validation data from replay buffer.

    Args:
        replay_buffer: Source of training data
        max_entries: Maximum entries to use
        val_split: Fraction for validation

    Returns:
        (train_entries, val_entries)
    """
    # Collect entries
    entries = replay_buffer.sample_entries(max_entries)

    if not entries:
        return [], []

    # Shuffle
    random.shuffle(entries)

    # Split
    val_size = int(len(entries) * val_split)
    val_entries = entries[:val_size]
    train_entries = entries[val_size:]

    return train_entries, val_entries
