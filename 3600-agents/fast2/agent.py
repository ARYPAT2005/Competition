import copy
from collections.abc import Callable
from typing import Tuple

import numpy as np

from game import board, move, enums


class PlayerAgent:
    """
    Board-first bot with thresholded expectiminimax for rat search.

    Entry points that must remain:
    - __init__
    - commentate
    - play
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.T = None
        self.belief = None

        self.search_miss_streak = 0
        self.turns_since_last_search = 999

        if transition_matrix is not None:
            self.T = np.asarray(transition_matrix, dtype=np.float64)
            self.belief = self._spawn_prior()
        else:
            self.belief = np.ones(64, dtype=np.float64) / 64.0

        self.turn_counter = 0
        self.last_best_prob = 0.0
        self.last_search_target = None
        self.last_eval = 0.0
        self.last_completed_depth = 0
        self.last_search_gain = 0.0
        self.last_board_gain = 0.0
        self.last_opp_threat = 0.0
        self.last_opp_setup_threat = 0.0
        self.last_search_threshold = 0.0

        self.noise_probs = {
            enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
            enums.Cell.SPACE: (0.7, 0.15, 0.15),
            enums.Cell.PRIMED: (0.1, 0.8, 0.1),
            enums.Cell.CARPET: (0.1, 0.1, 0.8),
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
            f"board_gain={self.last_board_gain:.2f}, "
            f"opp_threat={self.last_opp_threat:.2f}, "
            f"opp_setup_threat={self.last_opp_setup_threat:.2f}, "
            f"search_threshold={self.last_search_threshold:.3f}"
        )

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn_counter += 1

        if board.player_search[0] is not None:
            self.turns_since_last_search = 0
            if board.player_search[1]:
                self.search_miss_streak = 0
            else:
                self.search_miss_streak += 1
                self._eliminate_missed_square(board.player_search[0])
        else:
            self.turns_since_last_search += 1

        if board.opponent_search[0] is not None and not board.opponent_search[1]:
            self._eliminate_missed_square(board.opponent_search[0])

        if board.player_search[1] or board.opponent_search[1]:
            self.belief = self._spawn_prior()

        noise, observed_dist = sensor_data
        self._update_belief(board, noise, observed_dist)

        best_idx = int(np.argmax(self.belief))
        best_prob = float(self.belief[best_idx])
        best_loc = self._idx_to_loc(best_idx)

        self.last_best_prob = best_prob
        self.last_search_target = best_loc

        turns_left = board.player_worker.turns_left
        score_diff = board.player_worker.get_points() - board.opponent_worker.get_points()

        sorted_probs = np.sort(self.belief)
        second_prob = float(sorted_probs[-2]) if len(sorted_probs) >= 2 else 0.0
        confidence_gap = best_prob - second_prob

        valid_moves = board.get_valid_moves(exclude_search=True)
        if not valid_moves:
            self.last_eval = 8.0 * (6.0 * best_prob - 2.0)
            self.last_completed_depth = 0
            return move.Move.search(best_loc)

        valid_moves = self._order_moves(board, valid_moves, self.belief)

        remaining = time_left()
        turns_left_safe = max(1, turns_left)

        reserve = 0.75
        usable_time = max(0.05, remaining - reserve)
        budget = usable_time / turns_left_safe

        if turns_left >= 30:
            budget *= 1.8
        elif turns_left >= 20:
            budget *= 1.4
        elif turns_left <= 6:
            budget *= 1.4
        elif turns_left <= 12:
            budget *= 1.2

        if score_diff <= -4:
            budget *= 1.2
        elif score_diff >= 6:
            budget *= 0.9

        if abs(score_diff) <= 3:
            budget *= 1.25
        elif abs(score_diff) >= 10:
            budget *= 0.9

        num_moves = len(valid_moves)
        if num_moves >= 12:
            budget *= 1.4
        elif num_moves >= 8:
            budget *= 1.2
        elif num_moves <= 4:
            budget *= 0.85

        if remaining < 20:
            budget = min(budget, 1.5)

        budget = max(0.05, min(5.0, budget))

        start_remaining = remaining
        best_move = valid_moves[0]
        current_eval = self._evaluate(board, self.belief)
        best_value = current_eval
        completed_depth = 0
        depth = 1

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

        current_opp_threat = self._best_immediate_points(board, enemy=True)
        opp_reply_after_best = self._opponent_best_reply_points_after_move(board, best_move)
        threat_reduction = current_opp_threat - opp_reply_after_best

        current_opp_setup_threat = self._best_setup_threat(board, enemy=True)
        opp_setup_after_best = self._opponent_setup_threat_after_move(board, best_move)
        setup_reduction = current_opp_setup_threat - opp_setup_after_best

        current_opp_lane = self._max_carpet_roll_from_here(board, enemy=True)
        opp_lane_after_best = self._opponent_best_lane_after_move(board, best_move)
        lane_reduction = current_opp_lane - opp_lane_after_best

        current_territory = self._territory_balance(board)
        territory_after_best = self._territory_after_move(board, best_move)
        territory_gain = territory_after_best - current_territory

        self.last_opp_threat = current_opp_threat
        self.last_opp_setup_threat = current_opp_setup_threat

        board_gain = (
            (best_value - current_eval)
            + 10.0 * threat_reduction
            + 7.0 * setup_reduction
            + 20.0 * lane_reduction
            + 1.5 * territory_gain
        )

        if current_opp_lane >= 7:
            if lane_reduction >= 2:
                board_gain += 60.0
            board_gain -= 6.0 * max(0, opp_lane_after_best - 4)
        elif current_opp_lane >= 6:
            if lane_reduction >= 1:
                board_gain += 35.0
        elif current_opp_lane >= 5:
            if lane_reduction >= 1:
                board_gain += 18.0

        if threat_reduction >= 15:
            board_gain += 30.0
        elif threat_reduction >= 10:
            board_gain += 18.0
        elif threat_reduction >= 6:
            board_gain += 9.0

        if setup_reduction >= 15:
            board_gain += 20.0
        elif setup_reduction >= 10:
            board_gain += 12.0
        elif setup_reduction >= 6:
            board_gain += 6.0

        if turns_left <= 8:
            endgame_penalty = self._endgame_low_value_penalty(
                board,
                best_move,
                threat_reduction,
                setup_reduction,
                lane_reduction,
                territory_gain,
            )
            board_gain -= endgame_penalty

        search_threshold = self._search_expecti_threshold(
            turns_left,
            score_diff,
            current_opp_threat,
            current_opp_setup_threat,
            current_opp_lane,
            confidence_gap,
        )
        self.last_search_threshold = search_threshold

        search_gain = -1e9
        if best_prob >= search_threshold:
            branch_depth = 1 if completed_depth >= 2 and (start_remaining - time_left()) < budget * 0.75 else 0
            search_gain = self._search_expectiminimax_gain(
                board,
                self.belief,
                best_loc,
                current_eval,
                branch_depth,
                time_left,
                budget,
                start_remaining,
            )

            # Risk shaping around the raw expectiminimax value.
            if self.turns_since_last_search <= 1:
                search_gain -= 10.0
            elif self.turns_since_last_search <= 2:
                search_gain -= 4.0

            search_gain -= 10.0 * self.search_miss_streak

            if confidence_gap < 0.03:
                search_gain -= 8.0
            elif confidence_gap < 0.06:
                search_gain -= 4.0

            if turns_left >= 20:
                search_gain -= 4.0
            elif turns_left <= 8 and score_diff < 0:
                search_gain += 4.0

        self.last_search_gain = search_gain
        self.last_board_gain = board_gain

        # If there is an urgent block, take it instead of gambling on search.
        if (
            (current_opp_threat >= 15 and threat_reduction >= 6 and best_prob < 0.68)
            or (current_opp_setup_threat >= 15 and setup_reduction >= 6 and best_prob < 0.68)
            or (current_opp_lane >= 7 and lane_reduction >= 2 and best_prob < 0.75)
            or (current_opp_lane >= 6 and lane_reduction >= 1 and best_prob < 0.72)
            or (current_opp_lane >= 5 and lane_reduction >= 1 and best_prob < 0.68)
        ):
            self.last_eval = best_value
            self.last_completed_depth = completed_depth
            return best_move

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
    # Selective expectiminimax for the final rat decision
    # ------------------------------------------------------------------

    def _search_expecti_threshold(
        self,
        turns_left: int,
        score_diff: int,
        opp_threat: float,
        opp_setup_threat: float,
        opp_lane: int,
        confidence_gap: float,
    ) -> float:
        threshold = 0.42

        if turns_left > 20:
            threshold += 0.16
        elif turns_left > 10:
            threshold += 0.08
        else:
            if score_diff < 0:
                threshold -= 0.04
            elif score_diff >= 0:
                threshold += 0.05

        if opp_threat >= 15:
            threshold += 0.12
        elif opp_threat >= 10:
            threshold += 0.06
        elif opp_threat >= 6:
            threshold += 0.03

        if opp_setup_threat >= 15:
            threshold += 0.08
        elif opp_setup_threat >= 10:
            threshold += 0.04

        if opp_lane >= 7:
            threshold += 0.15
        elif opp_lane >= 6:
            threshold += 0.10
        elif opp_lane >= 5:
            threshold += 0.05

        if self.search_miss_streak > 0:
            threshold += 0.03 * self.search_miss_streak
        if self.turns_since_last_search <= 1:
            threshold += 0.08
        elif self.turns_since_last_search <= 2:
            threshold += 0.04

        if confidence_gap < 0.03:
            threshold += 0.06
        elif confidence_gap < 0.06:
            threshold += 0.03

        return max(0.35, min(0.85, threshold))

    def _search_expectiminimax_gain(
        self,
        board_obj: board.Board,
        belief_vec,
        search_loc,
        current_eval: float,
        branch_depth: int,
        time_left: Callable,
        budget: float,
        start_remaining: float,
    ) -> float:
        idx = self._loc_to_idx(search_loc)
        p_hit = float(belief_vec[idx])

        hit_value = self._search_branch_value(
            board_obj,
            belief_vec,
            search_loc,
            hit=True,
            branch_depth=branch_depth,
            time_left=time_left,
            budget=budget,
            start_remaining=start_remaining,
        )
        miss_value = self._search_branch_value(
            board_obj,
            belief_vec,
            search_loc,
            hit=False,
            branch_depth=branch_depth,
            time_left=time_left,
            budget=budget,
            start_remaining=start_remaining,
        )

        expected_value = p_hit * hit_value + (1.0 - p_hit) * miss_value
        return expected_value - current_eval

    def _search_branch_value(
        self,
        board_obj: board.Board,
        belief_vec,
        search_loc,
        hit: bool,
        branch_depth: int,
        time_left: Callable,
        budget: float,
        start_remaining: float,
    ) -> float:
        branch_board = board_obj.get_copy()

        if hit:
            branch_board.player_worker.increment_points(enums.RAT_BONUS)
            branch_board.player_search = (search_loc, True)
            branch_belief = self._spawn_prior()
        else:
            branch_board.player_worker.decrement_points(enums.RAT_PENALTY)
            branch_board.player_search = (search_loc, False)
            branch_belief = np.array(belief_vec, dtype=np.float64, copy=True)
            idx = self._loc_to_idx(search_loc)
            branch_belief[idx] = 0.0
            branch_belief = self._normalize(branch_belief)
            branch_belief = self._predict_belief(branch_belief)

        branch_board.end_turn(0)
        branch_board.reverse_perspective()

        elapsed = start_remaining - time_left()
        if branch_depth <= 0 or elapsed >= budget * 0.90 or time_left() < 0.02:
            return -self._evaluate(branch_board, branch_belief)

        return -self._negamax(
            branch_board,
            branch_depth,
            -float("inf"),
            float("inf"),
            time_left,
            budget,
            start_remaining,
            branch_belief,
        )

    # ------------------------------------------------------------------
    # Rat belief update
    # ------------------------------------------------------------------

    def _spawn_prior(self):
        if self.T is None:
            return np.ones(64, dtype=np.float64) / 64.0

        start = np.zeros(64, dtype=np.float64)
        start[0] = 1.0
        belief = start
        for _ in range(1000):
            belief = belief @ self.T
        return self._normalize(belief)

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

        valid_moves = self._order_moves(board_obj, valid_moves, belief_vec)
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

    # ------------------------------------------------------------------
    # Heuristic evaluation
    # ------------------------------------------------------------------

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

    def _lane_threat_value(self, length: int) -> float:
        if length <= 1:
            return 0.0
        if length == 2:
            return 0.5
        if length == 3:
            return 1.5
        if length == 4:
            return 3.0
        if length == 5:
            return 9.0
        if length == 6:
            return 16.0
        return 26.0 + 6.0 * (length - 7)

    def _evaluate(self, board_obj: board.Board, belief_vec=None) -> float:
        me = board_obj.player_worker
        opp = board_obj.opponent_worker

        score_diff = me.get_points() - opp.get_points()

        my_moves = len(board_obj.get_valid_moves(exclude_search=True))
        opp_moves = len(board_obj.get_valid_moves(enemy=True, exclude_search=True))
        mobility_diff = my_moves - opp_moves

        my_lane = self._max_carpet_roll_from_here(board_obj, enemy=False)
        opp_lane = self._max_carpet_roll_from_here(board_obj, enemy=True)

        my_roll = self._carpet_value(my_lane)
        opp_roll = self._carpet_value(opp_lane)
        carpet_threat_diff = my_roll - opp_roll

        lane_pressure_diff = self._lane_threat_value(my_lane) - self._lane_threat_value(opp_lane)

        my_immediate = self._best_immediate_points(board_obj, enemy=False)
        opp_immediate = self._best_immediate_points(board_obj, enemy=True)
        immediate_diff = my_immediate - opp_immediate

        my_setup = self._best_setup_threat(board_obj, enemy=False)
        opp_setup = self._best_setup_threat(board_obj, enemy=True)
        setup_diff = my_setup - opp_setup

        territory_diff = self._territory_balance(board_obj)
        rat_race = self._rat_race_term(board_obj, belief_vec)

        value = 0.0
        value += 18.0 * score_diff
        value += 3.0 * mobility_diff
        value += 7.0 * immediate_diff
        value += 5.0 * carpet_threat_diff
        value += 4.5 * lane_pressure_diff
        value += 4.0 * setup_diff
        value += 1.2 * territory_diff
        value += 2.5 * rat_race

        if me.turns_left <= 8:
            value += 8.0 * score_diff
            value += 8.0 * immediate_diff
            value += 5.0 * lane_pressure_diff
            value += 3.0 * setup_diff

        if opp_lane >= 7:
            value -= 75.0
        elif opp_lane >= 6:
            value -= 42.0
        elif opp_lane >= 5:
            value -= 18.0

        if opp_roll >= 21:
            value -= 35.0
        elif opp_roll >= 15:
            value -= 18.0

        if opp_setup >= 21:
            value -= 22.0
        elif opp_setup >= 15:
            value -= 12.0

        return value

    def _rat_race_term(self, board_obj: board.Board, belief_vec=None) -> float:
        if belief_vec is None:
            belief_vec = self.belief

        my_loc = board_obj.player_worker.get_location()
        opp_loc = board_obj.opponent_worker.get_location()

        top_indices = np.argsort(belief_vec)[-8:]
        total = 0.0
        weight_sum = 0.0

        for idx in top_indices:
            p = float(belief_vec[idx])
            if p <= 0:
                continue
            loc = self._idx_to_loc(int(idx))
            my_d = self._manhattan(my_loc, loc)
            opp_d = self._manhattan(opp_loc, loc)
            total += p * (opp_d - my_d)
            weight_sum += p

        if weight_sum == 0:
            return 0.0
        return total / weight_sum

    # ------------------------------------------------------------------
    # Move ordering and board features
    # ------------------------------------------------------------------

    def _order_moves(self, board_obj: board.Board, moves, belief_vec=None):
        current_opp_lane = self._max_carpet_roll_from_here(board_obj, enemy=True)
        current_territory = self._territory_balance(board_obj)
        turns_left = board_obj.player_worker.turns_left

        def move_key(mv):
            score = 0.0

            immediate_points = self._static_move_points(mv)
            score += 28.0 * immediate_points

            if mv.move_type == enums.MoveType.CARPET:
                score += 14.0 * self._carpet_value(mv.roll_length)
                if mv.roll_length == 1:
                    score -= 28.0

            elif mv.move_type == enums.MoveType.PRIME:
                score += 10.0

            elif mv.move_type == enums.MoveType.PLAIN:
                score += 4.0

            next_loc = self._resulting_location(board_obj, mv)
            if next_loc is not None:
                score += 0.55 * self._proximity_to_belief(next_loc, belief_vec)
                score += 0.35 * self._local_space_score(board_obj, next_loc)

            next_board = board_obj.forecast_move(mv)
            if next_board is None:
                return score - 1000.0

            my_lane_after = self._max_carpet_roll_from_here(next_board, enemy=False)
            territory_after = self._territory_balance(next_board)
            territory_gain = territory_after - current_territory

            score += 5.0 * self._lane_threat_value(my_lane_after)
            score += 0.9 * territory_gain

            next_board.reverse_perspective()

            opp_reply = self._best_immediate_points(next_board, enemy=False)
            opp_setup_reply = self._best_setup_threat(next_board, enemy=False)
            opp_lane_after = self._max_carpet_roll_from_here(next_board, enemy=False)
            lane_cut = current_opp_lane - opp_lane_after

            score -= 10.0 * opp_reply
            score -= 6.0 * opp_setup_reply
            score -= 18.0 * self._lane_threat_value(opp_lane_after)

            if lane_cut > 0:
                score += 22.0 * lane_cut

            if opp_lane_after >= 7:
                score -= 140.0
            elif opp_lane_after >= 6:
                score -= 90.0
            elif opp_lane_after >= 5:
                score -= 45.0

            if turns_left <= 8:
                low_value = (
                    immediate_points <= 1.0
                    and lane_cut <= 0
                    and opp_reply >= immediate_points
                    and my_lane_after < 5
                    and territory_gain <= 1.0
                )
                if low_value:
                    score -= 35.0 if turns_left > 5 else 55.0

                if immediate_points >= 4.0:
                    score += 10.0
                if lane_cut >= 1:
                    score += 25.0

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

    def _proximity_to_belief(self, loc, belief_vec=None) -> float:
        if belief_vec is None:
            belief_vec = self.belief

        top_indices = np.argsort(belief_vec)[-5:]
        score = 0.0
        for idx in top_indices:
            p = float(belief_vec[idx])
            rat_loc = self._idx_to_loc(int(idx))
            d = self._manhattan(loc, rat_loc)
            score += p * max(0, 8 - d)
        return score

    def _max_carpet_roll_from_here(self, board_obj: board.Board, enemy: bool = False) -> int:
        worker = board_obj.opponent_worker if enemy else board_obj.player_worker
        start = worker.get_location()
        return self._max_lane_from_loc(board_obj, start)

    def _max_lane_from_loc(self, board_obj: board.Board, start) -> int:
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
                if not self._in_bounds(cur) or not board_obj.is_cell_carpetable(cur):
                    break
                length += 1
            if length > best:
                best = length
        return best

    def _static_move_points(self, mv) -> float:
        if mv.move_type == enums.MoveType.PRIME:
            return 1.0
        if mv.move_type == enums.MoveType.CARPET:
            return float(self._carpet_value(mv.roll_length))
        return 0.0

    def _best_immediate_points(self, board_obj: board.Board, enemy: bool = False) -> float:
        valid_moves = board_obj.get_valid_moves(enemy=enemy, exclude_search=True)
        best = 0.0
        for mv in valid_moves:
            pts = self._static_move_points(mv)
            if pts > best:
                best = pts
        return best

    def _opponent_best_reply_points_after_move(self, board_obj: board.Board, mv) -> float:
        next_board = board_obj.forecast_move(mv)
        if next_board is None:
            return 0.0
        next_board.reverse_perspective()
        return self._best_immediate_points(next_board, enemy=False)

    def _opponent_best_lane_after_move(self, board_obj: board.Board, mv) -> int:
        next_board = board_obj.forecast_move(mv)
        if next_board is None:
            return 0
        next_board.reverse_perspective()
        return self._max_carpet_roll_from_here(next_board, enemy=False)

    def _territory_after_move(self, board_obj: board.Board, mv) -> float:
        next_board = board_obj.forecast_move(mv)
        if next_board is None:
            return self._territory_balance(board_obj)
        return self._territory_balance(next_board)

    def _best_setup_threat(self, board_obj: board.Board, enemy: bool = False) -> float:
        try:
            pov = copy.deepcopy(board_obj)
        except Exception:
            return 0.0

        if enemy:
            pov.reverse_perspective()

        valid_moves = pov.get_valid_moves(exclude_search=True)
        best = 0.0

        for mv in valid_moves:
            after = pov.forecast_move(mv)
            if after is None:
                continue

            immediate = self._static_move_points(mv)

            after.reverse_perspective()
            future_lane = self._max_carpet_roll_from_here(after, enemy=enemy)
            future_roll = self._carpet_value(future_lane)
            future_immediate = self._best_immediate_points(after, enemy=enemy)

            plan_value = (
                immediate
                + 0.8 * future_roll
                + 0.35 * future_immediate
                + 0.8 * self._lane_threat_value(future_lane)
            )
            if plan_value > best:
                best = plan_value

        return best

    def _opponent_setup_threat_after_move(self, board_obj: board.Board, mv) -> float:
        next_board = board_obj.forecast_move(mv)
        if next_board is None:
            return 0.0
        next_board.reverse_perspective()
        return self._best_setup_threat(next_board, enemy=False)

    def _territory_balance(self, board_obj: board.Board) -> float:
        my_loc = board_obj.player_worker.get_location()
        opp_loc = board_obj.opponent_worker.get_location()

        balance = 0.0
        for y in range(8):
            for x in range(8):
                loc = (x, y)
                if not self._is_walkable_cell(board_obj, loc):
                    continue
                my_d = self._manhattan(my_loc, loc)
                opp_d = self._manhattan(opp_loc, loc)

                if my_d < opp_d:
                    balance += 1.0
                elif opp_d < my_d:
                    balance -= 1.0

        balance += 0.5 * (
            self._open_neighbors(board_obj, my_loc)
            - self._open_neighbors(board_obj, opp_loc)
        )
        return balance

    def _local_space_score(self, board_obj: board.Board, loc) -> float:
        score = 0.0
        for y in range(8):
            for x in range(8):
                cell = (x, y)
                if not self._is_walkable_cell(board_obj, cell):
                    continue
                d = self._manhattan(loc, cell)
                if d <= 5:
                    score += max(0, 6 - d)

        score += 4.0 * self._max_lane_from_loc(board_obj, loc)
        score += 2.0 * self._open_neighbors(board_obj, loc)
        return score

    def _open_neighbors(self, board_obj: board.Board, loc) -> int:
        count = 0
        for direction in (
            enums.Direction.UP,
            enums.Direction.RIGHT,
            enums.Direction.DOWN,
            enums.Direction.LEFT,
        ):
            nxt = enums.loc_after_direction(loc, direction)
            if self._is_walkable_cell(board_obj, nxt):
                count += 1
        return count

    def _is_walkable_cell(self, board_obj: board.Board, loc) -> bool:
        if not self._in_bounds(loc):
            return False
        try:
            return board_obj.get_cell(loc) != enums.Cell.BLOCKED
        except Exception:
            return False

    def _endgame_low_value_penalty(
        self,
        board_obj: board.Board,
        mv,
        threat_reduction: float,
        setup_reduction: float,
        lane_reduction: float,
        territory_gain: float,
    ) -> float:
        turns_left = board_obj.player_worker.turns_left
        if turns_left > 8:
            return 0.0

        immediate = self._static_move_points(mv)
        if immediate >= 4.0:
            return 0.0
        if threat_reduction >= 4.0 or setup_reduction >= 4.0 or lane_reduction >= 1.0:
            return 0.0
        if territory_gain >= 2.0:
            return 0.0

        penalty = 12.0
        if immediate <= 1.0:
            penalty += 10.0
        if turns_left <= 6:
            penalty += 10.0
        return penalty

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

    def _in_bounds(self, loc) -> bool:
        return 0 <= loc[0] < 8 and 0 <= loc[1] < 8
