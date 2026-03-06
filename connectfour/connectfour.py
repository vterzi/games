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
    move_order: uint[7]  # type: ignore
    bottom_cells: ulonglong[7]  # type: ignore
    top_cells: ulonglong[7]  # type: ignore
    cols: ulonglong[7]  # type: ignore
    bottom_row: ulonglong
    board: ulonglong
    transpos_table: ulonglong[8388593]  # type: ignore

    def __cinit__(self) -> None:
        one: ulonglong = 1
        i_col: uint
        bottom_cell: ulonglong
        top_cell: ulonglong
        col: ulonglong

        self.n_rows = 6
        self.n_cols = 7
        if compiled:
            self.bottom_row = 0
            self.board = 0
            for i_col in range(7):
                self.move_order[i_col] = (  # type: ignore
                    self.n_cols // 2 + (1 - 2 * (i_col % 2)) * (i_col + 1) // 2
                )
                bottom_cell = one << (7 * i_col)
                top_cell = one << (7 * i_col + 5)
                col = (top_cell << 1) - bottom_cell
                self.bottom_cells[i_col] = bottom_cell  # type: ignore
                self.top_cells[i_col] = top_cell  # type: ignore
                self.cols[i_col] = col  # type: ignore
                self.bottom_row |= bottom_cell
                self.board |= col
            for i_col in range(8388593):
                self.transpos_table[i_col] = 0  # type: ignore
        else:
            self.move_order = tuple(  # type: ignore
                self.n_cols // 2 + (1 - 2 * (i_col % 2)) * (i_col + 1) // 2
                for i_col in range(7)
            )
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
            self.transpos_table = [0] * 8388593  # type: ignore

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
    def negamax(
        self,
        mask: ulonglong,
        position: ulonglong,
        depth: cint,
        alpha: cint,
        beta: cint,
    ) -> cint:
        one: ulonglong
        key: ulonglong
        idx: uint
        entry: ulonglong
        score: cint
        flag: uint
        i_col: uint
        new_mask: ulonglong
        new_position: ulonglong
        alpha_: cint

        one = 1
        good = self.good(mask, position)
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
        key = (self.bottom_row + mask) | position
        idx = key % 8388593
        entry = self.transpos_table[idx]  # type: ignore
        if entry & ((one << 49) - 1) == key:
            score = ((entry >> 49) & ((one << 8) - 1)) - 21
            flag = entry >> (49 + 8)
            if flag == 0:
                return score
            elif flag == 1 and score >= alpha:
                alpha = score
            elif flag == 2 and score <= beta:
                beta = score
            if alpha >= beta:
                return score
        score = -21
        alpha_ = alpha
        for i_col in self.move_order:  # type: ignore
            if good & self.cols[i_col]:  # type: ignore
                new_mask = self.move(mask, i_col)
                new_position = position ^ mask
                score = max(
                    score,
                    -self.negamax(
                        new_mask, new_position, depth - 1, -beta, -alpha
                    ),
                )
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
        if score <= alpha_:
            flag = 2
        elif score >= beta:
            flag = 1
        else:
            flag = 0
        entry = flag
        entry <<= 8
        entry |= score + 21
        entry = key | (entry << 49)
        self.transpos_table[idx] = entry  # type: ignore
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
