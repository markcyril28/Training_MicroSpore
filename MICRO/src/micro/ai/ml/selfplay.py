"""Self-play for generating training data."""

import random
from typing import List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import multiprocessing as mp

from ...types import Move, Player
from ...game_state import GameState
from ...board import Board
from ...ai.algorithmic.search import get_best_move

try:
    from .inference import get_ml_move
except Exception:
    get_ml_move = None
from .replay import ReplayEntry, ReplayBuffer


@dataclass
class GameRecord:
    """Record of a single game."""
    entries: List[ReplayEntry]
    winner: Optional[Player]
    num_moves: int


def play_single_game(
    difficulty: str = 'medium',
    max_moves: int = 200,
    noise_prob: float = 0.1,
    start_player: Player = Player.ONE,
    p1_policy: str = 'algorithmic',
    p2_policy: str = 'algorithmic',
    model_path: str = 'models/latest.pt',
    device=None,
) -> GameRecord:
    """
    Play a single self-play game using the algorithmic AI as teacher.

    Args:
        difficulty: AI difficulty level
        max_moves: Maximum moves before declaring draw
        noise_prob: Probability of playing a random move (for exploration)

    Returns:
        GameRecord with all positions and the chosen moves
    """
    start_player = Player(start_player)
    state = GameState(
        board=Board.initial(),
        current_player=start_player,
        move_count=0,
    )
    entries = []
    move_count = 0

    def _select_move(policy: str, legal_moves: List[Move]) -> Move:
        if random.random() < noise_prob and len(legal_moves) > 1:
            return random.choice(legal_moves)
        if policy == 'ml' and get_ml_move is not None:
            try:
                chosen = get_ml_move(state, model_path=model_path, device=device)
                if chosen is not None:
                    return chosen
            except Exception:
                pass
            return random.choice(legal_moves)
        chosen = get_best_move(state, difficulty)
        if chosen is None:
            return random.choice(legal_moves)
        return chosen

    while not state.is_terminal() and move_count < max_moves:
        legal_moves = state.legal_moves()
        if not legal_moves:
            break

        policy = p1_policy if state.current_player == Player.ONE else p2_policy
        chosen_move = _select_move(policy, legal_moves)

        # Find the index of chosen move
        try:
            chosen_index = legal_moves.index(chosen_move)
        except ValueError:
            # If exact match not found, find by path
            for i, m in enumerate(legal_moves):
                if m.path == chosen_move.path:
                    chosen_index = i
                    break
            else:
                chosen_index = 0
                chosen_move = legal_moves[0]

        # Record the position
        entry = ReplayEntry(
            state=state.to_compact(),
            legal_moves=[m.to_dict() for m in legal_moves],
            chosen_index=chosen_index,
            result=0,  # Will be filled after game ends
        )
        entries.append(entry)

        # Apply the move
        state = state.apply_move(chosen_move)
        move_count += 1

    # Determine winner
    winner = state.winner()

    # Update results from each player's perspective
    for i, entry in enumerate(entries):
        turn = entry.state['turn']
        player = Player(turn)

        if winner is None:
            entry.result = 0  # Draw
        elif winner == player:
            entry.result = 1  # Win
        else:
            entry.result = -1  # Loss

    return GameRecord(entries=entries, winner=winner, num_moves=move_count)


def _play_game_worker(args: Tuple[str, int, float, int]) -> List[dict]:
    """Worker function for parallel self-play."""
    difficulty, max_moves, noise_prob, start_player = args
    record = play_single_game(
        difficulty,
        max_moves,
        noise_prob,
        start_player,
        p1_policy='algorithmic',
        p2_policy='algorithmic',
    )
    return [e.to_dict() for e in record.entries]


def _play_game_worker_full(
    args: Tuple[str, int, float, int, str, str, str, object]
) -> List[dict]:
    """Worker function for threaded self-play with optional ML policy."""
    (difficulty, max_moves, noise_prob, start_player, p1_policy,
     p2_policy, model_path, device) = args
    record = play_single_game(
        difficulty=difficulty,
        max_moves=max_moves,
        noise_prob=noise_prob,
        start_player=start_player,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        model_path=model_path,
        device=device,
    )
    return [e.to_dict() for e in record.entries]


class SelfPlayRunner:
    """
    Runs self-play games in parallel to generate training data.
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        num_workers: int = 10,
        difficulty: str = 'medium',
        max_moves: int = 200,
        noise_prob: float = 0.1,
        p1_policy: str = 'algorithmic',
        p2_policy: str = 'algorithmic',
        model_path: str = 'models/latest.pt',
        device=None,
    ):
        self.replay_buffer = replay_buffer
        self.num_workers = num_workers
        self.difficulty = difficulty
        self.max_moves = max_moves
        self.noise_prob = noise_prob
        self.p1_policy = p1_policy
        self.p2_policy = p2_policy
        self.model_path = model_path
        self.device = device

        self._running = False
        self._games_completed = 0

    def run_games(self, num_games: int, callback=None) -> int:
        """
        Run self-play games and store in replay buffer.

        Args:
            num_games: Number of games to play
            callback: Optional callback(games_completed, total_games)

        Returns:
            Total number of training entries generated
        """
        self._running = True
        self._games_completed = 0
        total_entries = 0

        # Start new replay file
        self.replay_buffer.start_new_file()

        # Prepare arguments for workers (balance starting player across games)
        num_p1 = num_games // 2
        num_p2 = num_games - num_p1
        start_players = ([Player.ONE] * num_p1) + ([Player.TWO] * num_p2)
        random.shuffle(start_players)
        args = [
            (self.difficulty, self.max_moves, self.noise_prob, int(start))
            for start in start_players
        ]

        can_parallel = (
            self.p1_policy == 'algorithmic'
            and self.p2_policy == 'algorithmic'
            and self.num_workers > 1
        )

        # Use process pool for parallel games (algorithmic-only)
        # Note: On Windows, this requires __main__ guard
        if can_parallel:
            try:
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(_play_game_worker, a) for a in args]

                    for future in as_completed(futures):
                        if not self._running:
                            break

                        try:
                            entries_data = future.result()
                            entries = [ReplayEntry.from_dict(d) for d in entries_data]
                            self.replay_buffer.add_entries(entries)
                            total_entries += len(entries)

                            self._games_completed += 1
                            if callback:
                                callback(self._games_completed, num_games)

                        except Exception as e:
                            print(f"Self-play error: {e}")

            except Exception as e:
                # Fallback to sequential if multiprocessing fails
                print(f"Parallel self-play failed ({e}), falling back to sequential")
                can_parallel = False

        threaded_failed = False

        if not can_parallel and self.num_workers > 1:
            try:
                args_full = [
                    (
                        self.difficulty,
                        self.max_moves,
                        self.noise_prob,
                        int(start),
                        self.p1_policy,
                        self.p2_policy,
                        self.model_path,
                        self.device,
                    )
                    for start in start_players
                ]

                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = [executor.submit(_play_game_worker_full, a) for a in args_full]

                    for future in as_completed(futures):
                        if not self._running:
                            break

                        try:
                            entries_data = future.result()
                            entries = [ReplayEntry.from_dict(d) for d in entries_data]
                            self.replay_buffer.add_entries(entries)
                            total_entries += len(entries)

                            self._games_completed += 1
                            if callback:
                                callback(self._games_completed, num_games)

                        except Exception as e:
                            print(f"Self-play error: {e}")

            except Exception as e:
                print(f"Threaded self-play failed ({e}), falling back to sequential")
                threaded_failed = True

        if not can_parallel and (self.num_workers <= 1 or threaded_failed):
            for i, start_player in enumerate(start_players):
                if not self._running:
                    break

                record = play_single_game(
                    self.difficulty,
                    self.max_moves,
                    self.noise_prob,
                    start_player,
                    p1_policy=self.p1_policy,
                    p2_policy=self.p2_policy,
                    model_path=self.model_path,
                    device=self.device,
                )
                self.replay_buffer.add_entries(record.entries)
                total_entries += len(record.entries)

                self._games_completed += 1
                if callback:
                    callback(self._games_completed, num_games)

        self.replay_buffer.close()
        return total_entries

    def stop(self) -> None:
        """Stop running games."""
        self._running = False

    @property
    def games_completed(self) -> int:
        return self._games_completed
