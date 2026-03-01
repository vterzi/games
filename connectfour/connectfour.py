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
def bit_count(i: ulonglong):
    n: uint

    n = 0
    while i:
        i &= i - 1
        n += 1
    return n


@cclass
class ConnectFour:  # TODO generalize
    rows: uint
    cols: uint
    move_order: uint[7]  # type: ignore
    bottom_cells: ulonglong[7]  # type: ignore
    top_cells: ulonglong[7]  # type: ignore
    bottom_row: ulonglong
    board: ulonglong
    transpos_table: ulonglong[8388593]  # type: ignore

    def __cinit__(self) -> None:
        one: ulonglong = 1
        col: uint
        bottom_cell: ulonglong
        top_cell: ulonglong

        self.rows = 6
        self.cols = 7
        if compiled:
            self.move_order[0] = 3  # type: ignore
            self.move_order[1] = 2  # type: ignore
            self.move_order[2] = 4  # type: ignore
            self.move_order[3] = 1  # type: ignore
            self.move_order[4] = 5  # type: ignore
            self.move_order[5] = 0  # type: ignore
            self.move_order[6] = 6  # type: ignore
            self.bottom_row = 0
            self.board = (one << 49) - 1
            for col in range(7):
                bottom_cell = one << (7 * col)
                top_cell = one << (7 * col + 5)
                self.bottom_cells[col] = bottom_cell  # type: ignore
                self.top_cells[col] = top_cell  # type: ignore
                self.bottom_row |= bottom_cell
                self.board ^= top_cell << 1
            for col in range(8388593):
                self.transpos_table[col] = 0  # type: ignore
        else:
            self.move_order = (3, 2, 4, 1, 5, 0, 6)  # type: ignore
            self.bottom_cells = tuple(  # type: ignore
                1 << (7 * col) for col in range(7)
            )
            self.top_cells = tuple(  # type: ignore
                1 << (7 * col + 5) for col in range(7)
            )
            self.bottom_row = sum(self.bottom_cells)
            self.board = ((1 << 49) - 1) ^ (sum(self.top_cells) << 1)
            self.transpos_table = [0] * 8388593  # type: ignore

    @cfunc
    @inline
    def free(self, mask: ulonglong, col: uint) -> bint:
        return mask & self.top_cells[col] == 0  # type: ignore

    @cfunc
    @inline
    def move(self, mask: ulonglong, col: uint) -> ulonglong:
        return (mask + self.bottom_cells[col]) | mask  # type: ignore

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
    def negamax(
        self,
        mask: ulonglong,
        position: ulonglong,
        depth: uint,
        alpha: cint,
        beta: cint,
    ) -> cint:
        one: ulonglong
        key: ulonglong
        idx: uint
        entry: ulonglong
        value: cint
        flag: uint
        col: uint
        new_mask: ulonglong
        new_position: ulonglong
        alpha_: cint

        one = 1
        key = (self.bottom_row + mask) | position
        idx = key % 8388593
        entry = self.transpos_table[idx]  # type: ignore
        if entry & ((one << 49) - 1) == key:
            value = ((entry >> 49) & ((one << 8) - 1)) - 21
            flag = entry >> (49 + 8)
            if flag == 0:
                return value
            elif flag == 1 and value >= alpha:
                alpha = value
            elif flag == 2 and value <= beta:
                beta = value
            if alpha >= beta:
                return value
        if depth == 0:
            return 0
        for col in self.move_order:  # type: ignore
            if self.free(mask, col):
                new_mask = self.move(mask, col)
                if self.win(position | (new_mask ^ mask)):
                    value = (depth + 1) // 2
                    entry = value + 21
                    entry = key | (entry << 49)
                    self.transpos_table[idx] = entry  # type: ignore
                    return value
        value = -21
        alpha_ = alpha
        for col in self.move_order:  # type: ignore
            if self.free(mask, col):
                new_mask = self.move(mask, col)
                new_position = position ^ mask
                value = max(
                    value,
                    -self.negamax(
                        new_mask, new_position, depth - 1, -beta, -alpha
                    ),
                )
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        if value <= alpha_:
            flag = 2
        elif value >= beta:
            flag = 1
        else:
            flag = 0
        entry = flag
        entry <<= 8
        entry |= value + 21
        entry = key | (entry << 49)
        self.transpos_table[idx] = entry  # type: ignore
        return value

    @ccall
    def solve(
        self, mask: ulonglong, position: ulonglong, weak: bint = False
    ) -> cint:
        depth: cint
        min_score: cint
        max_score: cint
        med_score: cint
        score: cint

        depth = 42 - bit_count(mask)
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
                mask, position, depth, med_score, med_score + 1
            )
            if score <= med_score:
                max_score = score
            else:
                min_score = score
        return min_score

    @ccall
    def display(self, mask: ulonglong, position: ulonglong) -> None:
        one: ulonglong
        stride: uint
        row: uint
        _: uint
        cell: ulonglong

        one = 1
        stride = self.rows + 1
        string = ""
        color1 = "31"
        color2 = "33"
        if bit_count(mask) % 2 == 1:
            color1, color2 = color2, color1
        for row in range(self.rows):
            cell = one << row
            line = ""
            for _ in range(self.cols):
                if mask & cell:
                    color = color1 if position & cell else color2
                    disc = f"\x1b[{color}m\u25cf\x1b[0m "  # \u2b24
                else:
                    disc = "\u25cb "  # \u25ef
                line += disc
                cell <<= stride
            string = line[: len(line) - 1] + "\n" + string
        print(string[: len(string) - 1])
