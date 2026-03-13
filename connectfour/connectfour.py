# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: overflowcheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: cpow=True

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
class ConnectFour:  # https://github.com/PascalPons/connect4
    n_rows: cint
    n_cols: cint
    bottom_cells: uint64_t[7]
    top_cells: uint64_t[7]
    cols: uint64_t[7]
    bottom_row: uint64_t
    board: uint64_t
    move_order: cint[7]  # type: ignore
    transpos_tab_size: cint
    transpos_tab_keys: uint32_t[(1 << 23) + 9]
    transpos_tab_vals: uint8_t[(1 << 23) + 9]
    n_cells: cint
    stride: cint
    min_score: cint
    max_score: cint
    invalid_score: cint
    score_shift: cint

    def __cinit__(self) -> None:
        one: uint64_t
        i_col: cint
        bottom_cell: uint64_t
        top_cell: uint64_t
        col: uint64_t

        one = 1
        self.n_rows = 6
        self.n_cols = 7
        self.transpos_tab_size = (1 << 23) + 9
        self.n_rows = max(self.n_rows, 0)
        self.n_cols = max(self.n_cols, 0)
        self.transpos_tab_size = max(self.transpos_tab_size, 0)
        self.n_cells = self.n_rows * self.n_cols
        self.stride = self.n_rows + 1
        if self.n_cols > 9:
            raise ValueError("board wider than 9 columns")
        if self.n_cells + self.n_cols > 64:
            raise ValueError("board too large")
        self.min_score = -cdiv(self.n_cells, 2) + 3
        self.max_score = cdiv(self.n_cells + 1, 2) - 3
        self.invalid_score = self.min_score - 1
        self.score_shift = self.max_score - self.invalid_score
        if not compiled:
            self.bottom_cells = [0] * self.n_cols
            self.top_cells = [0] * self.n_cols
            self.cols = [0] * self.n_cols
            self.move_order = [0] * self.n_cols  # type: ignore
            self.transpos_tab_keys = [0] * self.transpos_tab_size
            self.transpos_tab_vals = [0] * self.transpos_tab_size
        self.bottom_row = 0
        self.board = 0
        for i_col in range(self.n_cols):
            bottom_cell = one << (self.stride * i_col)
            top_cell = one << (self.stride * (i_col + 1) - 2)
            col = (top_cell << 1) - bottom_cell
            self.bottom_cells[i_col] = bottom_cell
            self.top_cells[i_col] = top_cell
            self.cols[i_col] = col
            self.bottom_row |= bottom_cell
            self.board |= col
            self.move_order[i_col] = (  # type: ignore
                cdiv(self.n_cols, 2)
                + cdiv((1 - 2 * (i_col % 2)) * (i_col + 1), 2)
            )
        for i_col in range(self.transpos_tab_size):
            self.transpos_tab_keys[i_col] = 0
            self.transpos_tab_vals[i_col] = 0

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def free(self, occupied: uint64_t, i_col: cint) -> bint:
        return occupied & self.top_cells[i_col] == 0

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def win(self, position: uint64_t) -> bint:
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

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def possible(self, occupied: uint64_t) -> uint64_t:
        return (occupied + self.bottom_row) & self.board

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def winning(self, occupied: uint64_t, position: uint64_t) -> uint64_t:
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
    def good(self, occupied: uint64_t, position: uint64_t) -> uint64_t:
        possible: uint64_t
        losing: uint64_t
        forced: uint64_t

        possible = self.possible(occupied)
        losing = self.winning(occupied, position ^ occupied)
        forced = possible & losing
        if forced:
            if forced & (forced - 1):  # bit_count(forced) > 1
                return 0
            else:
                possible = forced
        return possible & ~(losing >> 1)

    @cfunc
    @inline
    @exceptval(check=False)  # type: ignore
    def score(self, occupied: uint64_t, position: uint64_t) -> cint:
        return bit_count(self.winning(occupied, position))

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
        good: uint64_t
        min_score: cint
        max_score: cint
        key_: uint64_t
        key: uint32_t
        idx: cint
        score: cint
        i_col: cint
        n_moves: cint
        i_move: cint
        move: uint64_t
        moves: uint64_t[7]
        scores: cint[7]  # type: ignore
        new_occupied: uint64_t
        new_position: uint64_t

        if not compiled:
            moves = [0] * self.n_cols
            scores = [0] * self.n_cols  # type: ignore

        good = self.good(occupied, position)
        if good == 0:
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

        # key_ = (self.bottom_row + occupied) | position
        key_ = occupied + position
        key = cast(uint32_t, key_)
        idx = key_ % self.transpos_tab_size
        if key == self.transpos_tab_keys[idx]:
            score = cast(cint, self.transpos_tab_vals[idx])
            if score > self.score_shift:
                min_score = score + self.invalid_score - self.score_shift
                if alpha < min_score:
                    alpha = min_score
                    if alpha >= beta:
                        return alpha
            else:
                max_score = score + self.invalid_score
                if beta > max_score:
                    beta = max_score
                    if alpha >= beta:
                        return beta

        n_moves = 0
        for i_col in self.move_order:  # type: ignore
            move = good & self.cols[i_col]
            if move:
                # score = self.score(occupied | move, position | move)
                score = self.score(occupied, position | move)
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
                self.transpos_tab_keys[idx] = key
                self.transpos_tab_vals[idx] = cast(
                    uint8_t, score - self.invalid_score + self.score_shift
                )
                return score
            if score > alpha:
                alpha = score

        score = alpha
        self.transpos_tab_keys[idx] = key
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
        if self.possible(occupied) & self.winning(occupied, position):
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
        new_occupied: uint64_t
        new_position: uint64_t
        i_col: cint
        score: cint
        scores: list[cint]

        scores = [self.invalid_score] * self.n_cols
        new_position = position ^ occupied
        for i_col in range(self.n_cols):
            if self.free(occupied, i_col):
                mod_occupied = occupied + self.bottom_cells[i_col]
                if self.win(position | (mod_occupied & self.cols[i_col])):
                    score = cdiv(self.n_cells - bit_count(occupied) + 1, 2)
                else:
                    new_occupied = occupied | mod_occupied
                    score = -self.solve(new_occupied, new_position, weak)
                scores[i_col] = score
        return tuple(scores)

    @ccall
    def play(self, moves: str) -> tuple[uint64_t, ...]:
        occupied: uint64_t
        position: uint64_t
        i_col: cint
        mod_occupied: uint64_t

        occupied = 0
        position = 0
        for move in moves:
            i_col = ord(move) - ord("1")
            if (
                i_col < 0
                or i_col >= self.n_cols
                or not self.free(occupied, i_col)
            ):
                break
            mod_occupied = occupied + self.bottom_cells[i_col]
            if self.win(position | (mod_occupied & self.cols[i_col])):
                break
            position ^= occupied
            occupied |= mod_occupied
        return occupied, position

    @ccall
    def display(self, occupied: uint64_t, position: uint64_t):
        one: uint64_t
        i_row: cint
        _: cint
        cell: uint64_t

        one = 1
        string = ""
        color1 = "31"
        color2 = "33"
        if bit_count(occupied) % 2 == 1:
            color1, color2 = color2, color1
        for i_row in range(self.n_rows):
            cell = one << i_row
            line = ""
            for _ in range(self.n_cols):
                if occupied & cell:
                    color = color1 if position & cell else color2
                    disc = f"\x1b[{color}m\u25cf\x1b[0m "  # \u2b24
                else:
                    disc = "\u25cb "  # \u25ef
                line += disc
                cell <<= self.stride
            string = line[: len(line) - 1] + "\n" + string
        print(string[: len(string) - 1])
