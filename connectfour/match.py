from .screen import Displayable, Screen
from .solver import ConnectFourSolver  # type: ignore


def cdiv(n: int, d: int) -> int:
    return int(n / d)


class ConnectFourMatch(Displayable):
    """Connect Four match."""

    def __init__(self, screen: Screen) -> None:
        self._game = ConnectFourSolver("connectfour/opening.txt")
        self._move_str = ""
        self._n_cols, self._n_rows = self._game.size()
        self._occupied = 0
        self._position = 0
        move_order = []
        for i in range(self._n_cols):
            move_order.append(
                cdiv(self._n_cols, 2) + cdiv((1 - 2 * (i % 2)) * (i + 1), 2)
            )
        self._move_order = tuple(move_order)
        self._move_col = self._move_order[0]
        self._colors = (1, 3)
        self._finished = False
        super().__init__(screen)
        screen.focus(self)

    def _empty_cell(self) -> str:
        return "\u25cb"  # \u25ef

    def _filled_cell(self, color: int) -> str:
        return f"\x1b[{30 + color}m\u25cf\x1b[0m"  # \u2b24

    def display(self) -> None:
        row_offset = (self._screen.rows - self._n_rows - 1) // 2 + 1
        col_offset = (self._screen.cols - 2 * self._n_cols) // 2 + 1
        turn = len(self._move_str) % 2
        if not self._finished:
            color = self._colors[turn]
            self._screen[row_offset, self._move_col * 2 + col_offset] = (
                self._filled_cell(color)
            )
        color1, color2 = self._colors
        if turn == 1:
            color1, color2 = color2, color1
        for i_row in range(self._n_rows):
            for i_col in range(self._n_cols):
                cell = 1 << (i_col * (self._n_rows + 1) + i_row)
                if self._occupied & cell:
                    color = color1 if self._position & cell else color2
                    disc = self._filled_cell(color)
                else:
                    disc = self._empty_cell()
                self._screen[
                    self._n_rows - i_row + row_offset, i_col * 2 + col_offset
                ] = disc

    def handle_key(self, key: str) -> None:
        if key == "\x1b[C" and not self._finished:
            for i_col in range(self._move_col + 1, self._n_cols):
                if self._game.free_col(self._occupied, i_col):
                    self._move_col = i_col
                    break
        elif key == "\x1b[D" and not self._finished:
            for i_col in range(self._move_col - 1, -1, -1):
                if self._game.free_col(self._occupied, i_col):
                    self._move_col = i_col
                    break
        elif (
            key in ("1", "2", "3", "4", "5", "6", "7", "8", "9")
            and not self._finished
        ):
            i_col = int(key) - 1
            if i_col < self._n_cols and self._game.free_col(
                self._occupied, i_col
            ):
                self._move_col = i_col
        elif key == "\r" and not self._finished:
            self._move_str += str(self._move_col + 1)
            self._occupied, self._position = self._game.play_moves(
                self._move_str
            )
            self._finished = self._game.winning_position(
                self._position ^ self._occupied
            )
            for i_col in self._move_order:
                if self._game.free_col(self._occupied, i_col):
                    self._move_col = i_col
                    break
            else:
                self._finished = True
        elif key in ("\b", "\x7f") and len(self._move_str) > 0:
            self._move_str = self._move_str[:-1]
            self._occupied, self._position = self._game.play_moves(
                self._move_str
            )
            self._finished = False
