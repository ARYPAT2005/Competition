from collections.abc import Callable
from typing import Tuple

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

        # Search behavior tracking
        self.search_miss_streak = 0
        self.turns_since_last_search = 999

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
        self.last_completed_depth = 0
        self.last_search_gain = 0.0
        self.last_board_gain = 0.0

        # Noise probabilities by cell type
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
            f"last_eval={self.last_eval:.2f}, "
            f"last_depth={self.last_completed_depth}, "
            f"search_miss_streak={self.search_miss_streak}, "
            f"search_gain={self.last_search_gain:.2f}, "
            f"board_gain={self.last_board_gain:.2f}"
        )

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn_counter += 1

        # Track result of my previous search
        if board.player_search[0] is not None:
            self.turns_since_last_search = 0
            if board.player_search[1]:
                self.search_miss_streak = 0
            else:
                self.search_miss_streak += 1
                # Strongly use the information that the rat was NOT there
                self._eliminate_missed_square(board.player_search[0])
        else:
            self.turns_since_last_search += 1

        # If someone just found the rat, reset the prior for the respawned rat
        if board.player_search[1] or board.opponent_search[1]:
            self._reset_belief_after_respawn()

        # Update rat belief using this turn's sensor
        noise, observed_dist = sensor_data
        self._update_belief(board, noise, observed_dist)

        # Best rat guess
        best_idx = int(np.argmax(self.belief))
        best_prob = float(self.belief[best_idx])
        best_loc = self._idx_to_loc(best_idx)

        self.last_best_prob = best_prob
        self.last_search_target = best_loc

        turns_left = board.player_worker.turns_left
        score_diff = board.player_worker.get_points() - board.opponent_worker.get_points()

        # Compare top rat square to second-best rat square
        sorted_probs = np.sort(self.belief)
        second_prob = float(sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0
        confidence_gap = best_prob - second_prob

        valid_moves = board.get_valid_moves(exclude_search=True)

        # If there are no non-search moves, search if possible
        if not valid_moves:
            self.last_eval = 18.0 * (6.0 * best_prob - 2.0)
            self.last_completed_depth = 0
            return move.Move.search(best_loc)

        valid_moves = self._order_moves(board, valid_moves)

        # -----------------------------
        # Iterative deepening time budget
        # -----------------------------
        remaining = time_left()
        turns_left_safe = max(1, turns_left)

        reserve = 0.75
        usable_time = max(0.05, remaining - reserve)

        budget = usable_time / turns_left_safe

        # Opening and endgame deserve more thought
        if turns_left >= 30:
            budget *= 1.8
        elif turns_left >= 20:
            budget *= 1.4
        elif turns_left <= 6:
            budget *= 1.4
        elif turns_left <= 12:
            budget *= 1.2

        # Score-based adjustment
        if score_diff <= -4:
            budget *= 1.2
        elif score_diff >= 6:
            budget *= 0.9

        # Close games deserve more thought
        if abs(score_diff) <= 3:
            budget *= 1.25
        elif abs(score_diff) >= 10:
            budget *= 0.9

        # More legal moves = more complexity = more time
        num_moves = len(valid_moves)
        if num_moves >= 12:
            budget *= 1.4
        elif num_moves >= 8:
            budget *= 1.2
        elif num_moves <= 4:
            budget *= 0.85

        # If clock is getting low overall, cap more aggressively
        if remaining < 20:
            budget = min(budget, 1.5)

        budget = max(0.05, min(5.0, budget))

        # -----------------------------
        # Find best board move
        # -----------------------------
        start_remaining = remaining
        best_move = valid_moves[0]
        best_value = -float("inf")
        completed_depth = 0
        depth = 1

        current_eval = self._evaluate(board, self.belief)

        while True:
            elapsed = start_remaining - time_left()
            if elapsed >= budget * 0.92:
                break

            move_at_depth, value_at_depth, completed = self._search_root_to_depth(
                board,
                valid_moves,
                depth,
                time_left,
                budget,
                start_remaining,
                self.belief,
            )

            if not completed:
                break

            best_move = move_at_depth
            best_value = value_at_depth
            completed_depth = depth
            depth += 1

        board_gain = best_value - current_eval

        # -----------------------------
        # Compare search vs best board move
        # -----------------------------
        search_gain = 18.0 * (6.0 * best_prob - 2.0)

        # Confidence handling
        if confidence_gap < 0.03:
            search_gain -= 14.0
        elif confidence_gap < 0.06:
            search_gain -= 7.0

        # Repeated misses and recent search penalty
        search_gain -= 12.0 * self.search_miss_streak
        if self.turns_since_last_search <= 1:
            search_gain -= 10.0
        elif self.turns_since_last_search <= 2:
            search_gain -= 4.0

        # Late-game risk shaping
        if turns_left <= 5:
            if score_diff > 0:
                search_gain -= 16.0
            elif score_diff < 0:
                search_gain += 12.0
        elif turns_left <= 10:
            if score_diff > 0:
                search_gain -= 6.0
            elif score_diff < 0:
                search_gain += 6.0

        # Very low top probability should almost never search
        if best_prob < 0.28:
            search_gain -= 1000.0

        self.last_search_gain = search_gain
        self.last_board_gain = board_gain

        if search_gain > board_gain:
            self.last_eval = search_gain
            self.last_completed_depth = completed_depth
            return move.Move.search(best_loc)

        self.last_eval = best_value
        self.last_completed_depth = completed_depth
        return best_move

    # ------------------------------------------------------------------
    # Root search with iterative deepening support
    # ------------------------------------------------------------------

    def _search_root_to_depth(
        self,
        board_obj: board.Board,
        valid_moves,
        depth: int,
        time_left: Callable,
        budget: float,
        start_remaining: float,
        belief_vec,
    ):
        best_move = valid_moves[0]
        best_value = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        # Opponent's next turn happens after one rat move
        future_belief = self._predict_belief(belief_vec)

        for mv in valid_moves:
            elapsed = start_remaining - time_left()
            if elapsed >= budget * 0.92:
                return best_move, best_value, False

            next_board = board_obj.forecast_move(mv)
            if next_board is None:
                continue

            next_board.reverse_perspective()

            val = -self._negamax(
                next_board,
                depth - 1,
                -beta,
                -alpha,
                time_left,
                budget,
                start_remaining,
                future_belief,
            )

            if val > best_value:
                best_value = val
                best_move = mv

            alpha = max(alpha, best_value)

        return best_move, best_value, True

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

    def _eliminate_missed_square(self, loc):
        if self.belief is None:
            return
        idx = self._loc_to_idx(loc)
        if 0 <= idx < len(self.belief):
            self.belief[idx] = 0.0
            self.belief = self._normalize(self.belief)

    def _predict_belief(self, belief_vec):
        if self.T is None or belief_vec is None:
            return belief_vec
        return self._normalize(belief_vec @ self.T)

    def _update_belief(self, board_obj: board.Board, noise, observed_dist: int):
        # Rat moves before the sensor is generated each turn
        if self.T is not None:
            self.belief = self.belief @ self.T

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
        return probs[int(noise)]

    def _distance_likelihood(self, actual: int, observed: int) -> float:
        prob = 0.0

        if max(actual - 1, 0) == observed:
            prob += 0.12
        if actual == observed:
            prob += 0.70
        if actual + 1 == observed:
            prob += 0.12
        if actual + 2 == observed:
            prob += 0.06

        return prob

    # ------------------------------------------------------------------
    # Negamax + alpha-beta
    # ------------------------------------------------------------------

    def _negamax(
        self,
        board_obj: board.Board,
        depth: int,
        alpha: float,
        beta: float,
        time_left: Callable,
        budget: float,
        start_remaining: float,
        belief_vec,
    ) -> float:
        elapsed = start_remaining - time_left()

        if elapsed >= budget * 0.92 or time_left() < 0.02:
            return self._evaluate(board_obj, belief_vec)

        if depth == 0 or board_obj.is_game_over():
            return self._evaluate(board_obj, belief_vec)

        valid_moves = board_obj.get_valid_moves(exclude_search=True)
        if not valid_moves:
            return self._evaluate(board_obj, belief_vec)

        valid_moves = self._order_moves(board_obj, valid_moves)

        next_belief = self._predict_belief(belief_vec)

        best = -float("inf")
        for mv in valid_moves:
            elapsed = start_remaining - time_left()
            if elapsed >= budget * 0.92:
                return self._evaluate(board_obj, belief_vec)

            next_board = board_obj.forecast_move(mv)
            if next_board is None:
                continue

            next_board.reverse_perspective()
            score = -self._negamax(
                next_board,
                depth - 1,
                -beta,
                -alpha,
                time_left,
                budget,
                start_remaining,
                next_belief,
            )

            if score > best:
                best = score
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best == -float("inf"):
            return self._evaluate(board_obj, belief_vec)

        return best

    def _carpet_value(self, length: int) -> int:
        table = {
            0: 0,
            1: -1,
            2: 2,
            3: 4,
            4: 6,
            5: 10,
            6: 15,
            7: 21,
        }
        return table.get(length, 0)
    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, board_obj: board.Board, belief_vec=None) -> float:
        me = board_obj.player_worker
        opp = board_obj.opponent_worker

        score_diff = me.get_points() - opp.get_points()

        my_moves = len(board_obj.get_valid_moves(exclude_search=True))
        opp_moves = len(board_obj.get_valid_moves(enemy=True, exclude_search=True))
        mobility_diff = my_moves - opp_moves

        my_len = self._max_carpet_roll_from_here(board_obj, enemy=False)
        opp_len = self._max_carpet_roll_from_here(board_obj, enemy=True)

        my_carpet_value = self._carpet_value(my_len)
        opp_carpet_value = self._carpet_value(opp_len)
        carpet_potential_diff = my_carpet_value - opp_carpet_value

        my_prime_options = self._count_primeable_directions(board_obj, enemy=False)
        opp_prime_options = self._count_primeable_directions(board_obj, enemy=True)
        prime_option_diff = my_prime_options - opp_prime_options

        rat_term = self._rat_position_term(board_obj, belief_vec)

        value = 0.0
        value += 18.0 * score_diff
        value += 2.5 * mobility_diff
        value += 10.0 * carpet_potential_diff
        value += 2.0 * prime_option_diff
        value += 5.0 * rat_term

        if me.turns_left <= 8:
            value += 4.0 * score_diff

        return value

    def _rat_position_term(self, board_obj: board.Board, belief_vec=None) -> float:
        if belief_vec is None:
            belief_vec = self.belief

        my_loc = board_obj.player_worker.get_location()

        top_indices = np.argsort(belief_vec)[-8:]
        total = 0.0
        weight_sum = 0.0

        for idx in top_indices:
            p = float(belief_vec[idx])
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
        score = 0.0

        if mv.move_type == enums.MoveType.CARPET:
            score += 35.0 * self._carpet_value(mv.roll_length)

        elif mv.move_type == enums.MoveType.PRIME:
            score += 40.0

        elif mv.move_type == enums.MoveType.PLAIN:
            score += 10.0

        next_loc = self._resulting_location(board_obj, mv)
        if next_loc is not None:
            score += self._proximity_to_belief(next_loc)

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