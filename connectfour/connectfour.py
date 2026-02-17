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


@cclass
class ConnectFour:  # TODO generalize
    rows = 6
    cols = 7
    order: uint[7]
    bottom: ulonglong[7]
    top: ulonglong[7]
    table: dict[tuple[ulonglong, ulonglong], tuple[cint, cint]] = {}

    def __cinit__(self) -> None:
        col: uint
        one: ulonglong

        if compiled:
            self.order[0] = 3
            self.order[1] = 2
            self.order[2] = 4
            self.order[3] = 1
            self.order[4] = 5
            self.order[5] = 0
            self.order[6] = 6
            one = 1
            for col in range(7):
                self.bottom[col] = one << (7 * col)
                self.top[col] = one << (7 * col + 5)
        else:
            self.order = (3, 2, 4, 1, 5, 0, 6)
            self.bottom = tuple(1 << (7 * col) for col in range(7))
            self.top = tuple(1 << (7 * col + 5) for col in range(7))

    @cfunc
    @inline
    def free(self, mask: ulonglong, col: uint) -> bint:
        return mask & self.top[col] == 0

    @cfunc
    @inline
    def move(self, mask: ulonglong, col: uint) -> ulonglong:
        return (mask + self.bottom[col]) | mask

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
        key = (mask, position)
        if key in self.table:
            value, flag = self.table[key]
            if flag == 0:
                return value
            elif flag == -1 and value >= beta:
                return value
            elif flag == 1 and value <= alpha:
                return value
        if depth == 0:
            return 0
        for col in self.order:
            if self.free(mask, col):
                new_mask = self.move(mask, col)
                if self.win(position | (new_mask ^ mask)):
                    value = (depth + 1) // 2
                    self.table[key] = (value, 0)
                    return value
        value = -21
        alpha_ = alpha
        for col in self.order:
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
            flag = 1
        elif value >= beta:
            flag = -1
        else:
            flag = 0
        self.table[key] = (value, flag)
        return value

    @ccall
    def solve(self):
        return self.negamax(0, 0, 42, -21, 21)

    # @ccall
    # def display(self, mask: ulonglong, position: ulonglong) -> None:
    #     rows = self.rows
    #     cols = self.cols
    #     stride = rows + 1
    #     string = ""
    #     for row in range(rows):
    #         i = 1 << row
    #         line = ""
    #         for _ in range(cols):
    #             if mask & i:
    #                 color = "31" if position & i else "33"
    #                 disc = f"\x1b[{color}m\u25cf\x1b[0m "  # \u2b24
    #             else:
    #                 disc = "\u25cb "  # \u25ef
    #             line += disc
    #             i <<= stride
    #         string = line[:-1] + "\n" + string
    #     print(string[:-1])
