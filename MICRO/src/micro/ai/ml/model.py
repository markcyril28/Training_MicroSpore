"""Neural network model for move scoring."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .move_encoder import BOARD_PLANES, MOVE_FEATURE_SIZE


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class BoardEncoder(nn.Module):
    """CNN encoder for the board state."""

    def __init__(self, embedding_size: int = 128, num_blocks: int = 4):
        super().__init__()

        self.input_conv = nn.Conv2d(BOARD_PLANES, 64, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(64)

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(64) for _ in range(num_blocks)
        ])

        # Final layers to produce embedding
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 8 * 8, embedding_size)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        """
        Encode board state.

        Args:
            board: Tensor of shape (batch, BOARD_PLANES, 8, 8)

        Returns:
            Tensor of shape (batch, embedding_size)
        """
        x = F.relu(self.input_bn(self.input_conv(board)))

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.fc(x)
        return x


class MoveScorer(nn.Module):
    """MLP to score a move given board embedding and move features."""

    def __init__(self, embedding_size: int = 128, hidden_size: int = 64):
        super().__init__()

        input_size = embedding_size + MOVE_FEATURE_SIZE

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, board_embedding: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
        """
        Score a move.

        Args:
            board_embedding: Tensor of shape (batch, embedding_size)
            move_features: Tensor of shape (batch, MOVE_FEATURE_SIZE)

        Returns:
            Tensor of shape (batch, 1) - score for each move
        """
        x = torch.cat([board_embedding, move_features], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MoveScorerNet(nn.Module):
    """
    Complete move scoring network.

    Takes a board state and a batch of move features,
    returns a score for each move.
    """

    def __init__(self, embedding_size: int = 128, num_blocks: int = 4, hidden_size: int = 64):
        super().__init__()

        self.board_encoder = BoardEncoder(embedding_size, num_blocks)
        self.move_scorer = MoveScorer(embedding_size, hidden_size)

    def forward(
        self,
        board: torch.Tensor,
        move_features: torch.Tensor,
        move_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Score moves for a batch of positions.

        Args:
            board: Tensor of shape (batch_size, BOARD_PLANES, 8, 8)
            move_features: Tensor of shape (total_moves, MOVE_FEATURE_SIZE)
                           where total_moves = sum(move_counts)
            move_counts: Tensor of shape (batch_size,) - number of moves per position

        Returns:
            Tensor of shape (total_moves,) - score for each move
        """
        batch_size = board.shape[0]

        # Encode all boards
        board_embeddings = self.board_encoder(board)  # (batch_size, embedding_size)

        # Expand board embeddings to match moves
        # Create indices to repeat board embeddings for each move
        indices = torch.repeat_interleave(
            torch.arange(batch_size, device=board.device),
            move_counts
        )
        expanded_embeddings = board_embeddings[indices]  # (total_moves, embedding_size)

        # Score all moves
        scores = self.move_scorer(expanded_embeddings, move_features)  # (total_moves, 1)

        return scores.squeeze(-1)

    def score_single(self, board: torch.Tensor, move_features: torch.Tensor) -> torch.Tensor:
        """
        Score moves for a single position (convenience method).

        Args:
            board: Tensor of shape (1, BOARD_PLANES, 8, 8) or (BOARD_PLANES, 8, 8)
            move_features: Tensor of shape (num_moves, MOVE_FEATURE_SIZE)

        Returns:
            Tensor of shape (num_moves,) - score for each move
        """
        if board.dim() == 3:
            board = board.unsqueeze(0)

        num_moves = move_features.shape[0]

        # Encode board
        board_embedding = self.board_encoder(board)  # (1, embedding_size)

        # Expand for all moves
        expanded_embedding = board_embedding.expand(num_moves, -1)

        # Score moves
        scores = self.move_scorer(expanded_embedding, move_features)

        return scores.squeeze(-1)


def create_model(
    embedding_size: int = 128,
    num_blocks: int = 4,
    hidden_size: int = 64
) -> MoveScorerNet:
    """Create a new model with default parameters."""
    return MoveScorerNet(
        embedding_size=embedding_size,
        num_blocks=num_blocks,
        hidden_size=hidden_size
    )


def load_model(path: str, device: torch.device = None) -> MoveScorerNet:
    """Load a model from a checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(path, map_location=device)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Handle models saved after torch.compile() - strip "_orig_mod." prefix
    state_dict = {
        k.replace("_orig_mod.", ""): v
        for k, v in state_dict.items()
    }

    model = create_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def save_model(model: MoveScorerNet, path: str, **kwargs) -> None:
    """Save a model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        **kwargs
    }
    torch.save(checkpoint, path)
