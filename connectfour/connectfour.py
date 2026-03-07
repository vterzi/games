# https://github.com/PascalPons/connect4

from cython import (  # type: ignore
    compiled,
    bint,
    uint,
    int as cint,
    ulonglong,
    cfunc,
    inline,
    ccall,
    cclass,
    cast,
)
from cython.cimports.libc.stdint import (  # type: ignore
    uint8_t,
    uint32_t,
)


@cfunc
def bit_count(i: ulonglong) -> uint:
    n: uint

    n = 0
    while i:
        i &= i - 1
        n += 1
    return n


@cclass
class ConnectFour:
    n_rows: uint
    n_cols: uint
    min_score: cint
    max_score: cint
    bottom_cells: ulonglong[7]  # type: ignore
    top_cells: ulonglong[7]  # type: ignore
    cols: ulonglong[7]  # type: ignore
    bottom_row: ulonglong
    board: ulonglong
    move_order: uint[7]  # type: ignore
    transpos_tab_keys: uint32_t[8388617]  # type: ignore
    transpos_tab_vals: uint8_t[8388617]  # type: ignore

    def __cinit__(self) -> None:
        one: ulonglong = 1
        i_col: uint
        bottom_cell: ulonglong
        top_cell: ulonglong
        col: ulonglong

        self.n_rows = 6
        self.n_cols = 7
        if self.n_cols > 9:
            raise ValueError("board wider than 9 columns")
        if (self.n_rows + 1) * self.n_cols > 64:
            raise ValueError("board too large")
        self.min_score = -(self.n_rows * self.n_cols) // 2 + 3
        self.max_score = (self.n_rows * self.n_cols + 1) // 2 - 3
        if compiled:
            self.bottom_row = 0
            self.board = 0
            for i_col in range(7):
                bottom_cell = one << (7 * i_col)
                top_cell = one << (7 * i_col + 5)
                col = (top_cell << 1) - bottom_cell
                self.bottom_cells[i_col] = bottom_cell  # type: ignore
                self.top_cells[i_col] = top_cell  # type: ignore
                self.cols[i_col] = col  # type: ignore
                self.bottom_row |= bottom_cell
                self.board |= col
                self.move_order[i_col] = (  # type: ignore
                    self.n_cols // 2 + (1 - 2 * (i_col % 2)) * (i_col + 1) // 2
                )
            for i_col in range(8388593):
                self.transpos_tab_keys[i_col] = 0  # type: ignore
                self.transpos_tab_vals[i_col] = 0  # type: ignore
        else:
            self.bottom_cells = tuple(  # type: ignore
                1 << (7 * i_col) for i_col in range(7)
            )
            self.top_cells = tuple(  # type: ignore
                1 << (7 * i_col + 5) for i_col in range(7)
            )
            self.cols = tuple(  # type: ignore
                (top_cell << 1) - bottom_cell
                for bottom_cell, top_cell in zip(
                    self.bottom_cells, self.top_cells
                )
            )
            self.bottom_row = sum(self.bottom_cells)
            self.board = sum(self.cols)
            self.move_order = tuple(  # type: ignore
                self.n_cols // 2 + (1 - 2 * (i_col % 2)) * (i_col + 1) // 2
                for i_col in range(7)
            )
            self.transpos_tab_keys = [0] * 8388593  # type: ignore
            self.transpos_tab_vals = [0] * 8388593  # type: ignore

    @cfunc
    @inline
    def free(self, occupied: ulonglong, i_col: uint) -> bint:
        return occupied & self.top_cells[i_col] == 0  # type: ignore

    @cfunc
    @inline
    def move(self, occupied: ulonglong, i_col: uint) -> ulonglong:
        return (occupied + self.bottom_cells[i_col]) | occupied  # type: ignore

    @cfunc
    @inline
    def win(self, position: ulonglong) -> bint:
        overlap: ulonglong

        overlap = position & (position >> 1)
        if overlap & (overlap >> 2):
            return True
        overlap = position & (position >> 7)
        if overlap & (overlap >> 14):
            return True
        overlap = position & (position >> 6)
        if overlap & (overlap >> 12):
            return True
        overlap = position & (position >> 8)
        if overlap & (overlap >> 16):
            return True
        return False

    @cfunc
    @inline
    def possible(self, occupied: ulonglong) -> ulonglong:
        return (occupied + self.bottom_row) & self.board

    @cfunc
    @inline
    def winning(self, occupied: ulonglong, position: ulonglong) -> ulonglong:
        winning: ulonglong
        overlap: ulonglong

        winning = (position << 1) & (position << 2) & (position << 3)

        overlap = (position << 7) & (position << 14)
        winning |= overlap & (position << 21)
        winning |= overlap & (position >> 7)
        overlap >>= 21
        winning |= overlap & (position << 7)
        winning |= overlap & (position >> 21)

        overlap = (position << 6) & (position << 12)
        winning |= overlap & (position << 18)
        winning |= overlap & (position >> 6)
        overlap >>= 18
        winning |= overlap & (position << 6)
        winning |= overlap & (position >> 18)

        overlap = (position << 8) & (position << 16)
        winning |= overlap & (position << 24)
        winning |= overlap & (position >> 8)
        overlap >>= 24
        winning |= overlap & (position << 8)
        winning |= overlap & (position >> 24)

        return winning & (self.board ^ occupied)

    @cfunc
    @inline
    def good(self, occupied: ulonglong, position: ulonglong) -> ulonglong:
        possible: ulonglong
        losing: ulonglong
        forced: ulonglong
        zero: ulonglong

        zero = 0
        possible = self.possible(occupied)
        losing = self.winning(occupied, position ^ occupied)
        forced = possible & losing
        if forced:
            if bit_count(forced) > 1:
                return zero
            else:
                possible = forced
        return possible & ~(losing >> 1)

    @cfunc
    @inline
    def score(self, occupied: ulonglong, position: ulonglong) -> uint:
        return bit_count(self.winning(occupied, position))

    @cfunc
    def negamax(
        self,
        occupied: ulonglong,
        position: ulonglong,
        depth: cint,
        alpha: cint,
        beta: cint,
    ) -> cint:
        key: uint32_t
        idx: uint
        score: cint
        i_col: uint
        moves: ulonglong[7]  # type: ignore
        scores: cint[7]  # type: ignore
        n_moves: uint
        i_move: uint
        new_occupied: ulonglong
        new_position: ulonglong

        if not compiled:
            moves = [0] * 7  # type: ignore
            scores = [0] * 7  # type: ignore

        good = self.good(occupied, position)
        if good == 0:
            return -depth // 2
        if depth <= 2:
            return 0
        min_score = -(depth - 2) // 2
        if alpha < min_score:
            alpha = min_score
            if alpha >= beta:
                return alpha
        max_score = (depth - 1) // 2
        if beta > max_score:
            beta = max_score
            if alpha >= beta:
                return beta
        # key = (self.bottom_row + occupied) | position
        key = occupied + position
        idx = key % 8388617
        if key == self.transpos_tab_keys[idx]:  # type: ignore
            score = cast(cint, self.transpos_tab_vals[idx])  # type: ignore
            if score > self.max_score - self.min_score + 1:
                min_score = score + 2 * self.min_score - self.max_score - 2
                if alpha < min_score:
                    alpha = min_score
                    if alpha >= beta:
                        return alpha
            else:
                max_score = score + self.min_score - 1
                if beta > max_score:
                    beta = max_score
                    if alpha >= beta:
                        return beta

        n_moves = 0
        for i_col in self.move_order:  # type: ignore
            move = good & self.cols[i_col]  # type: ignore
            if move:
                score = self.score(occupied | move, position | move)
                i_move = n_moves
                n_moves += 1
                while (
                    i_move > 0 and scores[i_move - 1] < score  # type: ignore
                ):
                    moves[i_move] = moves[i_move - 1]  # type: ignore
                    scores[i_move] = scores[i_move - 1]  # type: ignore
                    i_move -= 1
                moves[i_move] = move  # type: ignore
                scores[i_move] = score  # type: ignore

        score = -self.min_score
        new_position = position ^ occupied
        for i_move in range(n_moves):
            new_occupied = occupied | moves[i_move]  # type: ignore
            score = max(
                score,
                -self.negamax(
                    new_occupied, new_position, depth - 1, -beta, -alpha
                ),
            )
            if score >= beta:
                self.transpos_tab_keys[idx] = key  # type: ignore
                self.transpos_tab_vals[idx] = cast(  # type: ignore
                    uint8_t, score + self.max_score - 2 * self.min_score + 2
                )
                return score
            if score > alpha:
                alpha = score
        self.transpos_tab_keys[idx] = key  # type: ignore
        self.transpos_tab_vals[idx] = cast(  # type: ignore
            uint8_t, alpha - self.min_score + 1
        )
        return score

    @ccall
    def solve(
        self, occupied: ulonglong, position: ulonglong, weak: bint = False
    ) -> cint:
        depth: cint
        min_score: cint
        max_score: cint
        med_score: cint
        score: cint

        depth = 42 - bit_count(occupied)
        if self.possible(occupied) & self.winning(occupied, position) > 0:
            return (depth + 1) // 2
        if weak:
            min_score = -1
            max_score = 1
        else:
            min_score = -depth // 2
            max_score = (depth + 1) // 2
        while min_score < max_score:
            med_score = min_score + (max_score - min_score) // 2
            if med_score <= 0:
                med_score = min(med_score, min_score // 2)
            elif med_score >= 0:
                med_score = max(med_score, max_score // 2)
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
        occupied: ulonglong,
        position: ulonglong,
        weak: bint = False,
    ) -> tuple[cint, ...]:
        new_occupied: ulonglong
        new_position: ulonglong
        i_col: uint
        score: cint
        scores: list[cint]

        scores = [-self.min_score - 1] * 7
        new_position = position ^ occupied
        for i_col in range(7):
            if self.free(occupied, i_col):
                mod_occupied = (
                    occupied + self.bottom_cells[i_col]  # type: ignore
                )
                if self.win(
                    position
                    | (mod_occupied & self.cols[i_col])  # type: ignore
                ):
                    score = (43 - bit_count(occupied)) // 2
                else:
                    new_occupied = self.move(occupied, i_col)
                    score = -self.solve(new_occupied, new_position, weak)
                scores[i_col] = score
        return tuple(scores)

    @ccall
    def play(self, moves: str) -> tuple[ulonglong, ulonglong]:
        occupied: ulonglong
        position: ulonglong
        i_col: uint
        mod_occupied: ulonglong

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
            mod_occupied = occupied + self.bottom_cells[i_col]  # type: ignore
            if self.win(
                position | (mod_occupied & self.cols[i_col])  # type: ignore
            ):
                break
            position ^= occupied
            occupied |= mod_occupied
        return occupied, position

    @ccall
    def display(self, occupied: ulonglong, position: ulonglong) -> None:
        one: ulonglong
        stride: uint
        row: uint
        _: uint
        cell: ulonglong

        one = 1
        stride = self.n_rows + 1
        string = ""
        color1 = "31"
        color2 = "33"
        if bit_count(occupied) % 2 == 1:
            color1, color2 = color2, color1
        for row in range(self.n_rows):
            cell = one << row
            line = ""
            for _ in range(self.n_cols):
                if occupied & cell:
                    color = color1 if position & cell else color2
                    disc = f"\x1b[{color}m\u25cf\x1b[0m "  # \u2b24
                else:
                    disc = "\u25cb "  # \u25ef
                line += disc
                cell <<= stride
            string = line[: len(line) - 1] + "\n" + string
        print(string[: len(string) - 1])
