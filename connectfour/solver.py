# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: cpow=True

from math import gcd

from cython import (  # type: ignore
    compiled,
    bint,
    int as cint,
    cast,
    cfunc,
    ccall,
    inline,
    exceptval,
    cclass,
)
from cython.cimports.libc.stdint import (  # type: ignore
    uint8_t,
    uint32_t,
    uint64_t,
)


@cfunc
@inline
@exceptval(check=False)  # type: ignore
def cdiv(n: cint, d: cint) -> cint:
    return cast(cint, n / d)


@cfunc
@inline
@exceptval(check=False)  # type: ignore
def bit_count(i: uint64_t) -> cint:
    n: cint

    n = 0
    while i:
        i &= i - 1
        n += 1
    return n


@cclass
class ConnectFourSolver:  # https://github.com/PascalPons/connect4
    n_rows: cint
    n_cols: cint
    bottom_cells: uint64_t[7]  # n_cols
    top_cells: uint64_t[7]  # n_cols
    cols: uint64_t[7]  # n_cols
    ext_cols: uint64_t[7]  # n_cols
    bottom_row: uint64_t
    board: uint64_t
    move_order: cint[7]  # n_cols # type: ignore
    transpos_tab_size: cint
    transpos_tab_keys: uint32_t[(1 << 23) + 9]  # transpos_tab_size
    transpos_tab_vals: uint8_t[(1 << 23) + 9]  # transpos_tab_size
    opening_tab_size: cint
    opening_tab_keys: uint32_t[(1 << 23) + 9]  # opening_tab_size
    opening_tab_vals: uint8_t[(1 << 23) + 9]  # opening_tab_size
    opening_tab_depth: cint
    n_cells: cint
    stride: cint
    min_score: cint
    max_score: cint
    invalid_score: cint
    score_shift: cint

    def __cinit__(self, opening_file: str = "") -> None:
        one: uint64_t
        n_ext_cells: cint
        tab_size_coprime: uint64_t
        tab_size_lo_bound: cint
        i: cint
        bottom_cell: uint64_t
        top_cell: uint64_t
        col: uint64_t

        N_ROWS: cint = 6
        N_COLS: cint = 7
        TAB_SIZE: cint = (1 << 23) + 9

        one = 1
        self.n_rows = N_ROWS
        self.n_cols = N_COLS
        if self.n_rows < 4 or self.n_cols < 4:
            raise ValueError("invalid board size")
        if self.n_cols > 9:  # necessary for `move_str`
            raise ValueError("board wider than 9 columns")
        self.n_cells = self.n_rows * self.n_cols
        n_ext_cells = self.n_cells + self.n_cols
        if n_ext_cells > 64:
            raise ValueError("board too large")
        self.stride = self.n_rows + 1

        tab_size_coprime = one << 32
        tab_size_lo_bound = one << (n_ext_cells - 32)
        self.transpos_tab_size = TAB_SIZE
        if gcd(tab_size_coprime, self.transpos_tab_size) > 1:
            raise ValueError("transposition table size not coprime with 2^32")
        if self.transpos_tab_size <= tab_size_lo_bound:
            raise ValueError("transposition table too small")
        self.opening_tab_size = TAB_SIZE
        if gcd(tab_size_coprime, self.opening_tab_size) > 1:
            raise ValueError("opening table size not coprime with 2^32")
        if self.opening_tab_size <= tab_size_lo_bound:
            raise ValueError("opening table too small")
        self.opening_tab_depth = -1

        self.min_score = -cdiv(self.n_cells, 2) + 3
        self.max_score = cdiv(self.n_cells + 1, 2) - 3
        self.invalid_score = self.min_score - 1
        self.score_shift = self.max_score - self.invalid_score

        if not compiled:
            self.bottom_cells = [0] * self.n_cols
            self.top_cells = [0] * self.n_cols
            self.cols = [0] * self.n_cols
            self.ext_cols = [0] * self.n_cols
            self.move_order = [0] * self.n_cols  # type: ignore
            self.transpos_tab_keys = [0] * self.transpos_tab_size
            self.transpos_tab_vals = [0] * self.transpos_tab_size
            self.opening_tab_keys = [0] * self.opening_tab_size
            self.opening_tab_vals = [0] * self.opening_tab_size

        self.bottom_row = 0
        self.board = 0
        for i in range(self.n_cols):
            bottom_cell = one << (self.stride * i)
            top_cell = one << (self.stride * (i + 1) - 2)
            col = (top_cell << 1) - bottom_cell
            self.bottom_cells[i] = bottom_cell
            self.top_cells[i] = top_cell
            self.cols[i] = col
            self.ext_cols[i] = (top_cell << 2) - bottom_cell
            self.bottom_row |= bottom_cell
            self.board |= col
            self.move_order[i] = (  # type: ignore
                cdiv(self.n_cols, 2) + cdiv((1 - 2 * (i % 2)) * (i + 1), 2)
            )

        for i in range(self.transpos_tab_size):
            self.transpos_tab_keys[i] = 0
            self.transpos_tab_vals[i] = 0

        for i in range(self.opening_tab_size):
            self.opening_tab_keys[i] = 0
            self.opening_tab_vals[i] = 0

        if len(opening_file) > 0:
            self.init_opening(opening_file)

    def __init__(self, opening_file: str = "") -> None:
        if not compiled:
            self.__cinit__(opening_file)

    @ccall
    def size(self) -> tuple[cint, ...]:
        return self.n_cols, self.n_rows

    @ccall
    def nil_score(self) -> cint:
        return self.invalid_score

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def unique_key(self, key: uint64_t) -> uint64_t:
        step: cint
        i_col: cint
        mirror_key: uint64_t
        shift: cint
        up_col: cint

        step = self.stride << 1
        i_col = self.n_cols >> 1
        if self.n_cols & 1:
            mirror_key = key & self.ext_cols[i_col]
            shift = 0
        else:
            mirror_key = 0
            shift = -self.stride
        up_col = self.n_cols - 1
        for i_col in range(i_col - 1, -1, -1):
            shift += step
            mirror_key |= (key & self.ext_cols[i_col]) << shift
            mirror_key |= (key & self.ext_cols[up_col - i_col]) >> shift
        return min(key, mirror_key)

    @ccall
    def free_col(self, occupied: uint64_t, i_col: cint) -> bint:
        return occupied & self.top_cells[i_col] == 0

    @ccall
    def winning_position(self, position: uint64_t) -> bint:
        stride: cint
        overlap: uint64_t

        stride = 1
        overlap = position & (position >> stride)
        if overlap & (overlap >> (stride << 1)):
            return True
        stride = self.stride
        overlap = position & (position >> stride)
        if overlap & (overlap >> (stride << 1)):
            return True
        stride = self.stride - 1
        overlap = position & (position >> stride)
        if overlap & (overlap >> (stride << 1)):
            return True
        stride = self.stride + 1
        overlap = position & (position >> stride)
        if overlap & (overlap >> (stride << 1)):
            return True
        return False

    @ccall
    def play_moves(self, move_str: str) -> tuple[uint64_t, ...]:
        occupied: uint64_t
        position: uint64_t
        move_char: str
        i_col: cint
        mod_occupied: uint64_t

        occupied = 0
        position = 0
        for move_char in move_str:
            i_col = ord(move_char) - ord("1")
            if (
                i_col < 0
                or i_col >= self.n_cols
                or not self.free_col(occupied, i_col)
            ):
                break
            mod_occupied = occupied + self.bottom_cells[i_col]
            win = self.winning_position(
                position | (mod_occupied & self.cols[i_col])
            )
            position ^= occupied
            occupied |= mod_occupied
            if win:
                break
        return occupied, position

    @cfunc
    @inline
    def init_opening(self, opening_file: str):
        header: str
        line: str
        tokens: list[str]
        move_str: str
        score_str: str
        score: cint
        n_moves: cint
        occupied: uint64_t
        position: uint64_t
        full_unique_key: uint64_t
        partial_unique_key: uint32_t
        unique_idx: cint
        saved_score: cint

        with open(opening_file, "r") as file:
            header = next(file)
            if header.split() != [str(self.n_cols), str(self.n_rows)]:
                raise ValueError("invalid opening file header")
            for line in file:
                if len(line.strip()) == 0:
                    continue
                tokens = line.split()
                if len(tokens) < 2:
                    tokens.insert(0, "")
                move_str, score_str = tokens
                score = cast(cint, int(score_str))
                n_moves = len(move_str)
                if n_moves > self.opening_tab_depth:
                    self.opening_tab_depth = n_moves
                occupied, position = self.play_moves(move_str)
                full_unique_key = self.unique_key(occupied + position)
                partial_unique_key = cast(uint32_t, full_unique_key)
                unique_idx = full_unique_key % self.opening_tab_size
                saved_score = cast(cint, self.opening_tab_vals[unique_idx])
                if saved_score == 0 or abs(score) < abs(
                    saved_score + self.invalid_score
                ):
                    self.opening_tab_keys[unique_idx] = partial_unique_key
                    self.opening_tab_vals[unique_idx] = cast(
                        uint8_t, score - self.invalid_score
                    )

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def possible_moves(self, occupied: uint64_t) -> uint64_t:
        return (occupied + self.bottom_row) & self.board

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def winning_moves(
        self, occupied: uint64_t, position: uint64_t
    ) -> uint64_t:
        stride1: cint
        stride2: cint
        stride3: cint
        overlap: uint64_t
        winning: uint64_t

        winning = (position << 1) & (position << 2) & (position << 3)

        stride1 = self.stride
        stride2 = stride1 << 1
        stride3 = stride2 + stride1
        overlap = (position << stride1) & (position << stride2)
        winning |= overlap & (position << stride3)
        winning |= overlap & (position >> stride1)
        overlap >>= stride3
        winning |= overlap & (position << stride1)
        winning |= overlap & (position >> stride3)

        stride1 = self.stride - 1
        stride2 = stride1 << 1
        stride3 = stride2 + stride1
        overlap = (position << stride1) & (position << stride2)
        winning |= overlap & (position << stride3)
        winning |= overlap & (position >> stride1)
        overlap >>= stride3
        winning |= overlap & (position << stride1)
        winning |= overlap & (position >> stride3)

        stride1 = self.stride + 1
        stride2 = stride1 << 1
        stride3 = stride2 + stride1
        overlap = (position << stride1) & (position << stride2)
        winning |= overlap & (position << stride3)
        winning |= overlap & (position >> stride1)
        overlap >>= stride3
        winning |= overlap & (position << stride1)
        winning |= overlap & (position >> stride3)

        return winning & (self.board ^ occupied)

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def good_moves(self, occupied: uint64_t, position: uint64_t) -> uint64_t:
        possible_moves: uint64_t
        non_losing_moves: uint64_t
        forced_moves: uint64_t

        possible_moves = self.possible_moves(occupied)
        non_losing_moves = self.winning_moves(occupied, position ^ occupied)
        forced_moves = possible_moves & non_losing_moves
        if forced_moves:
            if forced_moves & (forced_moves - 1):
                # bit_count(forced_moves) > 1
                return 0
            else:
                possible_moves = forced_moves
        return possible_moves & ~(non_losing_moves >> 1)

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def position_score(self, occupied: uint64_t, position: uint64_t) -> cint:
        return bit_count(self.winning_moves(occupied, position))

    @cfunc
    @exceptval(check=False)  # type: ignore
    def negamax(
        self,
        occupied: uint64_t,
        position: uint64_t,
        depth: cint,
        alpha: cint,
        beta: cint,
    ) -> cint:
        good_moves: uint64_t
        min_score: cint
        max_score: cint
        score: cint
        full_key: uint64_t
        partial_key: uint32_t
        idx: cint
        full_unique_key: uint64_t
        partial_unique_key: uint32_t
        unique_idx: cint
        i_col: cint
        n_moves: cint
        i_move: cint
        move: uint64_t
        moves: uint64_t[7]  # n_cols
        scores: cint[7]  # n_cols # type: ignore
        new_occupied: uint64_t
        new_position: uint64_t

        if not compiled:
            moves = [0] * self.n_cols
            scores = [0] * self.n_cols  # type: ignore

        good_moves = self.good_moves(occupied, position)
        if good_moves == 0:
            return -(depth >> 1)
        if depth <= 2:
            return 0

        min_score = -((depth - 2) >> 1)
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha
        max_score = (depth - 1) >> 1
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta

        # full_key = (self.bottom_row + occupied) | position
        full_key = occupied + position
        partial_key = cast(uint32_t, full_key)
        idx = full_key % self.transpos_tab_size
        if partial_key == self.transpos_tab_keys[idx]:
            score = cast(cint, self.transpos_tab_vals[idx])
            if score > self.score_shift:
                min_score = score + self.invalid_score - self.score_shift
                if alpha < min_score:
                    alpha = min_score
                    if alpha >= beta:
                        return alpha
            elif score > 0:
                max_score = score + self.invalid_score
                if beta > max_score:
                    beta = max_score
                    if alpha >= beta:
                        return beta

        if self.n_cells - depth <= self.opening_tab_depth:
            full_unique_key = self.unique_key(full_key)
            partial_unique_key = cast(uint32_t, full_unique_key)
            unique_idx = full_unique_key % self.opening_tab_size
            if partial_unique_key == self.opening_tab_keys[unique_idx]:
                score = cast(cint, self.opening_tab_vals[unique_idx])
                if score > 0:
                    score += self.invalid_score
                    return score

        n_moves = 0
        for i_col in self.move_order:  # type: ignore
            move = good_moves & self.cols[i_col]
            if move:
                # score = self.position_score(occupied | move, position | move)
                score = self.position_score(occupied, position | move)
                i_move = n_moves
                n_moves += 1
                while i_move and scores[i_move - 1] < score:  # type: ignore
                    moves[i_move] = moves[i_move - 1]
                    scores[i_move] = scores[i_move - 1]  # type: ignore
                    i_move -= 1
                moves[i_move] = move
                scores[i_move] = score  # type: ignore

        new_position = position ^ occupied
        depth -= 1
        for i_move in range(n_moves):
            new_occupied = occupied | moves[i_move]
            score = -self.negamax(
                new_occupied, new_position, depth, -beta, -alpha
            )
            if score >= beta:
                self.transpos_tab_keys[idx] = partial_key
                self.transpos_tab_vals[idx] = cast(
                    uint8_t, score - self.invalid_score + self.score_shift
                )
                return score
            if score > alpha:
                alpha = score

        score = alpha
        self.transpos_tab_keys[idx] = partial_key
        self.transpos_tab_vals[idx] = cast(uint8_t, score - self.invalid_score)
        return score

    @ccall
    def solve(
        self, occupied: uint64_t, position: uint64_t, weak: bint = False
    ) -> cint:
        depth: cint
        min_score: cint
        max_score: cint
        med_score: cint
        score: cint

        depth = self.n_cells - bit_count(occupied)
        if self.possible_moves(occupied) & self.winning_moves(
            occupied, position
        ):
            return cdiv(depth + 1, 2)
        if weak:
            min_score = -1
            max_score = 1
        else:
            min_score = -cdiv(depth, 2)
            max_score = cdiv(depth + 1, 2)
        while min_score < max_score:
            med_score = min_score + cdiv(max_score - min_score, 2)
            if med_score <= 0:
                med_score = min(med_score, cdiv(min_score, 2))
            elif med_score >= 0:
                med_score = max(med_score, cdiv(max_score, 2))
            score = self.negamax(
                occupied, position, depth, med_score, med_score + 1
            )
            if score <= med_score:
                max_score = score
            else:
                min_score = score
        return min_score

    @ccall
    def analyze(
        self,
        occupied: uint64_t,
        position: uint64_t,
        weak: bint = False,
    ) -> tuple[cint, ...]:
        scores: list[cint]
        mod_occupied: uint64_t
        new_occupied: uint64_t
        new_position: uint64_t
        i_col: cint
        score: cint

        scores = [self.invalid_score] * self.n_cols
        new_position = position ^ occupied
        for i_col in range(self.n_cols):
            if self.free_col(occupied, i_col):
                mod_occupied = occupied + self.bottom_cells[i_col]
                if self.winning_position(
                    position | (mod_occupied & self.cols[i_col])
                ):
                    score = cdiv(self.n_cells - bit_count(occupied) + 1, 2)
                else:
                    new_occupied = occupied | mod_occupied
                    score = -self.solve(new_occupied, new_position, weak)
                scores[i_col] = score
        return tuple(scores)

    @cfunc
    def explore_moves(
        self,
        unique_keys: set[uint64_t],
        score_dict: dict[str, cint],
        move_str: str,
        depth: cint,
    ):
        occupied: uint64_t
        position: uint64_t
        unique_key: uint64_t
        good_moves: uint64_t
        score: cint
        i_col: cint
        n_moves: cint
        i_move: cint
        move: uint64_t
        moves: cint[7]  # n_cols # type: ignore
        scores: cint[7]  # n_cols # type: ignore

        if not compiled:
            moves = [0] * self.n_cols  # type: ignore
            scores = [0] * self.n_cols  # type: ignore

        if depth <= 0:
            return
        occupied, position = self.play_moves(move_str)
        unique_key = self.unique_key(occupied + position)
        if unique_key in unique_keys:
            return
        good_moves = self.good_moves(occupied, position)
        if good_moves == 0:
            return
        if self.n_cells - bit_count(occupied) <= 2:
            return

        unique_keys.add(unique_key)
        score_dict[move_str] = self.solve(occupied, position)

        n_moves = 0
        for i_col in self.move_order:  # type: ignore
            move = good_moves & self.cols[i_col]
            if move:
                # score = self.position_score(occupied | move, position | move)
                score = self.position_score(occupied, position | move)
                i_move = n_moves
                n_moves += 1
                while i_move and scores[i_move - 1] < score:  # type: ignore
                    moves[i_move] = moves[i_move - 1]  # type: ignore
                    scores[i_move] = scores[i_move - 1]  # type: ignore
                    i_move -= 1
                moves[i_move] = i_col  # type: ignore
                scores[i_move] = score  # type: ignore

        depth -= 1
        for i_move in range(n_moves):
            i_col = moves[i_move]  # type: ignore
            self.explore_moves(
                unique_key, score_dict, move_str + str(i_col + 1), depth
            )

    @ccall
    def gen_openings(self, depth: cint) -> dict[str, cint]:
        unique_keys: set[uint64_t]
        score_dict: dict[str, cint]

        unique_keys = set()
        score_dict = {}
        self.explore_moves(unique_keys, score_dict, "", depth + 1)
        return score_dict
