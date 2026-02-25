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
    table: ulonglong[8388593]
    key: ulonglong

    def __cinit__(self) -> None:
        col: uint
        tmp: ulonglong

        if compiled:
            self.order[0] = 3
            self.order[1] = 2
            self.order[2] = 4
            self.order[3] = 1
            self.order[4] = 5
            self.order[5] = 0
            self.order[6] = 6
            tmp = 1
            self.key = 0
            for col in range(7):
                self.bottom[col] = tmp << (7 * col)
                self.top[col] = tmp << (7 * col + 5)
                self.key |= self.bottom[col]
            for col in range(8388593):
                self.table[col] = 0
        else:
            self.order = (3, 2, 4, 1, 5, 0, 6)
            self.bottom = tuple(1 << (7 * col) for col in range(7))
            self.top = tuple(1 << (7 * col + 5) for col in range(7))
            self.table = [0] * 8388593
            self.key = sum(self.bottom)

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
        key: ulonglong
        idx: cint
        value: cint
        flag: uint
        col: uint
        new_mask: ulonglong
        new_position: ulonglong
        alpha_: cint
        tmp: ulonglong

        key = (self.key + mask) | position
        idx = key % 8388593
        tmp = 1
        if self.table[idx] & ((tmp << 49) - 1) == key:
            value = ((self.table[idx] >> 49) & ((tmp << 8) - 1)) - 21
            flag = self.table[idx] >> (49 + 8)
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
        for col in self.order:
            if self.free(mask, col):
                new_mask = self.move(mask, col)
                if self.win(position | (new_mask ^ mask)):
                    value = (depth + 1) // 2
                    tmp = value + 21
                    self.table[idx] = key | (tmp << 49)
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
            flag = 2
        elif value >= beta:
            flag = 1
        else:
            flag = 0
        tmp = flag
        tmp <<= 8
        tmp |= value + 21
        self.table[idx] = key | (tmp << 49)
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
