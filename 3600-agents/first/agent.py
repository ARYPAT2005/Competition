from collections.abc import Callable
from typing import Tuple
import math
import random

import numpy as np

from game import board, move, enums


class PlayerAgent:
    """
    Entry points that must remain:
    - __init__
    - commentate
    - play
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.T = None
        self.belief = None

        if transition_matrix is not None:
            self.T = np.asarray(transition_matrix, dtype=np.float64)

            # Rat starts at (0,0) and gets 1000 headstart moves.
            start = np.zeros(64, dtype=np.float64)
            start[0] = 1.0
            belief = start
            for _ in range(1000):
                belief = belief @ self.T
            self.belief = self._normalize(belief)
        else:
            self.belief = np.ones(64, dtype=np.float64) / 64.0

        # Debug / commentary info
        self.turn_counter = 0
        self.last_best_prob = 0.0
        self.last_search_target = None
        self.last_eval = 0.0

        self.noise_probs = {
            enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
            enums.Cell.SPACE:   (0.7, 0.15, 0.15),
            enums.Cell.PRIMED:  (0.1, 0.8, 0.1),
            enums.Cell.CARPET:  (0.1, 0.1, 0.8),
        }

    def commentate(self):
        return (
            f"turns={self.turn_counter}, "
            f"last_best_rat_prob={self.last_best_prob:.3f}, "
            f"last_search_target={self.last_search_target}, "
            f"last_eval={self.last_eval:.2f}"
        )

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn_counter += 1

        # If a successful search happened recently, reset to the respawn prior.
        # The engine stores recent search outcomes on the board.
        if board.player_search[1] or board.opponent_search[1]:
            self._reset_belief_after_respawn()

        # Update rat belief using this turn's sensor.
        noise, observed_dist = sensor_data
        self._update_belief(board, noise, observed_dist)

        # Best rat guess
        best_idx = int(np.argmax(self.belief))
        best_prob = float(self.belief[best_idx])
        best_loc = self._idx_to_loc(best_idx)

        self.last_best_prob = best_prob
        self.last_search_target = best_loc

        # Search threshold:
        # EV(search) = 4p - 2(1-p) = 6p - 2  -> positive if p > 1/3
        # Use a slightly more conservative threshold in general.
        turns_left = board.player_worker.turns_left
        threshold = 0.42
        if turns_left <= 10:
            threshold = 0.36
        if turns_left <= 5:
            threshold = 0.34

        # If rat probability is high enough, search.
        if best_prob >= threshold:
            return move.Move.search(best_loc)

        # Otherwise choose a board move with alpha-beta.
        valid_moves = board.get_valid_moves(exclude_search=True)

        # Safety fallback
        if not valid_moves:
            return move.Move.search(best_loc)

        # Move ordering helps alpha-beta.
        valid_moves = self._order_moves(board, valid_moves)

        remaining = time_left()
        if remaining > 40:
            depth = 3
        elif remaining > 12:
            depth = 2
        else:
            depth = 1

        best_move = valid_moves[0]
        best_value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for mv in valid_moves:
            next_board = board.forecast_move(mv)
            if next_board is None:
                continue

            # Reverse perspective so recursive evaluator always sees
            # "current player" as board.player_worker
            next_board.reverse_perspective()

            val = -self._negamax(
                next_board,
                depth - 1,
                -beta,
                -alpha,
                time_left,
            )

            if val > best_value:
                best_value = val
                best_move = mv

            alpha = max(alpha, best_value)

        self.last_eval = best_value
        return best_move

    # ------------------------------------------------------------------
    # Rat belief update
    # ------------------------------------------------------------------

    def _reset_belief_after_respawn(self):
        if self.T is None:
            self.belief = np.ones(64, dtype=np.float64) / 64.0
            return

        start = np.zeros(64, dtype=np.float64)
        start[0] = 1.0
        belief = start
        for _ in range(1000):
            belief = belief @ self.T
        self.belief = self._normalize(belief)

    def _update_belief(self, board_obj: board.Board, noise, observed_dist: int):
        # Prediction step: rat moves before the sensor is generated each turn.
        if self.T is not None:
            self.belief = self.belief @ self.T

        # Correction step using noise + distance observation
        worker_loc = board_obj.player_worker.get_location()
        likelihood = np.zeros(64, dtype=np.float64)

        for idx in range(64):
            loc = self._idx_to_loc(idx)
            cell_type = board_obj.get_cell(loc)

            p_noise = self._noise_likelihood(cell_type, noise)
            actual_dist = self._manhattan(worker_loc, loc)
            p_dist = self._distance_likelihood(actual_dist, observed_dist)

            likelihood[idx] = p_noise * p_dist

        self.belief *= likelihood
        self.belief = self._normalize(self.belief)

    def _noise_likelihood(self, cell_type, noise) -> float:
        probs = self.noise_probs.get(cell_type, self.noise_probs[enums.Cell.SPACE])
        # Noise enum ordering from provided code:
        # SQUEAK=0, SCRATCH=1, SQUEAL=2
        return probs[int(noise)]

    def _distance_likelihood(self, actual: int, observed: int) -> float:
        """
        Distance estimator probabilities:
        one less: 0.12
        correct : 0.70
        one more: 0.12
        two more: 0.06

        Returned distance is clipped at zero.
        """
        prob = 0.0

        # offset -1 with clipping
        if max(actual - 1, 0) == observed:
            prob += 0.12
        # offset 0
        if actual == observed:
            prob += 0.70
        # offset +1
        if actual + 1 == observed:
            prob += 0.12
        # offset +2
        if actual + 2 == observed:
            prob += 0.06

        return prob

    # ------------------------------------------------------------------
    # Negamax + alpha-beta
    # ------------------------------------------------------------------

    def _negamax(self, board_obj: board.Board, depth: int, alpha: float, beta: float, time_left: Callable) -> float:
        # Keep a little time buffer
        if time_left() < 0.03:
            return self._evaluate(board_obj)

        if depth == 0 or board_obj.is_game_over():
            return self._evaluate(board_obj)

        valid_moves = board_obj.get_valid_moves(exclude_search=True)
        if not valid_moves:
            return self._evaluate(board_obj)

        valid_moves = self._order_moves(board_obj, valid_moves)

        best = -float("inf")
        for mv in valid_moves:
            next_board = board_obj.forecast_move(mv)
            if next_board is None:
                continue

            next_board.reverse_perspective()
            score = -self._negamax(next_board, depth - 1, -beta, -alpha, time_left)

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best == -float("inf"):
            return self._evaluate(board_obj)

        return best

    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, board_obj: board.Board) -> float:
        me = board_obj.player_worker
        opp = board_obj.opponent_worker

        score_diff = me.get_points() - opp.get_points()

        my_moves = len(board_obj.get_valid_moves(exclude_search=True))
        opp_moves = len(board_obj.get_valid_moves(enemy=True, exclude_search=True))
        mobility_diff = my_moves - opp_moves

        my_carpet_potential = self._max_carpet_roll_from_here(board_obj, enemy=False)
        opp_carpet_potential = self._max_carpet_roll_from_here(board_obj, enemy=True)
        carpet_potential_diff = my_carpet_potential - opp_carpet_potential

        my_prime_options = self._count_primeable_directions(board_obj, enemy=False)
        opp_prime_options = self._count_primeable_directions(board_obj, enemy=True)
        prime_option_diff = my_prime_options - opp_prime_options

        rat_term = self._rat_position_term(board_obj)

        # Heuristic weights
        value = 0.0
        value += 18.0 * score_diff
        value += 2.0 * mobility_diff
        value += 5.0 * carpet_potential_diff
        value += 1.5 * prime_option_diff
        value += 8.0 * rat_term

        # Slight preference for having more turns remaining if the state is otherwise close
        value += 0.2 * (me.turns_left - opp.turns_left)

        return value

    def _rat_position_term(self, board_obj: board.Board) -> float:
        """
        Reward being closer to likely rat squares.
        """
        my_loc = board_obj.player_worker.get_location()

        # Weighted average negative distance to top rat cells
        top_indices = np.argsort(self.belief)[-8:]
        total = 0.0
        weight_sum = 0.0

        for idx in top_indices:
            p = float(self.belief[idx])
            if p <= 0:
                continue
            loc = self._idx_to_loc(int(idx))
            d = self._manhattan(my_loc, loc)
            total += p * (-d)
            weight_sum += p

        if weight_sum == 0:
            return 0.0
        return total / weight_sum

    # ------------------------------------------------------------------
    # Move ordering and board features
    # ------------------------------------------------------------------

    def _order_moves(self, board_obj: board.Board, moves):
        def move_key(mv):
            score = 0

            if mv.move_type == enums.MoveType.CARPET:
                score += 100 + 20 * mv.roll_length
            elif mv.move_type == enums.MoveType.PRIME:
                score += 40
            elif mv.move_type == enums.MoveType.PLAIN:
                score += 10

            # Prefer moves that bring us closer to likely rat areas
            next_loc = self._resulting_location(board_obj, mv)
            if next_loc is not None:
                rat_bonus = self._proximity_to_belief(next_loc)
                score += rat_bonus

            return score

        return sorted(moves, key=move_key, reverse=True)

    def _resulting_location(self, board_obj: board.Board, mv):
        cur = board_obj.player_worker.get_location()

        if mv.move_type in (enums.MoveType.PLAIN, enums.MoveType.PRIME):
            return enums.loc_after_direction(cur, mv.direction)

        if mv.move_type == enums.MoveType.CARPET:
            loc = cur
            for _ in range(mv.roll_length):
                loc = enums.loc_after_direction(loc, mv.direction)
            return loc

        return None

    def _proximity_to_belief(self, loc) -> float:
        # Higher if closer to high-prob rat cells
        top_indices = np.argsort(self.belief)[-5:]
        score = 0.0
        for idx in top_indices:
            p = float(self.belief[idx])
            rat_loc = self._idx_to_loc(int(idx))
            d = self._manhattan(loc, rat_loc)
            score += p * max(0, 8 - d)
        return score

    def _max_carpet_roll_from_here(self, board_obj: board.Board, enemy: bool = False) -> int:
        worker = board_obj.opponent_worker if enemy else board_obj.player_worker
        start = worker.get_location()

        best = 0
        for direction in (
            enums.Direction.UP,
            enums.Direction.RIGHT,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
        ):
            cur = start
            length = 0
            while True:
                cur = enums.loc_after_direction(cur, direction)
                if not board_obj.is_cell_carpetable(cur):
                    break
                length += 1
            if length > best:
                best = length
        return best

    def _count_primeable_directions(self, board_obj: board.Board, enemy: bool = False) -> int:
        worker = board_obj.opponent_worker if enemy else board_obj.player_worker
        cur = worker.get_location()

        # Cannot prime if current square already primed or carpeted
        if board_obj.get_cell(cur) in (enums.Cell.PRIMED, enums.Cell.CARPET):
            return 0

        count = 0
        for direction in (
            enums.Direction.UP,
            enums.Direction.RIGHT,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
        ):
            nxt = enums.loc_after_direction(cur, direction)
            if not board_obj.is_cell_blocked(nxt):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _normalize(self, arr):
        s = float(np.sum(arr))
        if s <= 0:
            return np.ones_like(arr, dtype=np.float64) / len(arr)
        return arr / s

    def _idx_to_loc(self, idx: int):
        return (idx % 8, idx // 8)

    def _loc_to_idx(self, loc):
        return loc[1] * 8 + loc[0]

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])