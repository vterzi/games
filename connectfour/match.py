from threading import Thread
from queue import Queue
from random import choice

from .screen import Interactable, Screen
from .solver import ConnectFourSolver  # type: ignore


def cdiv(n: int, d: int) -> int:
    return int(n / d)


class ConnectFourState:
    def __init__(self) -> None:
        self._game = ConnectFourSolver("connectfour/opening.txt")
        self._move_str = ""
        self._occupied = 0
        self._position = 0
        self._turn = 0
        self.update()

    @property
    def size(self) -> tuple[int, int]:
        size = self._game.size()
        return size[0], size[1]

    @property
    def nil_score(self) -> int:
        return self._game.nil_score()

    @property
    def move_str(self) -> str:
        return self._move_str

    @property
    def occupied(self) -> int:
        return self._occupied

    @property
    def position(self) -> int:
        return self._position

    @property
    def turn(self) -> int:
        return self._turn

    @property
    def prev_turn(self) -> int:
        return self._turn ^ 1

    def update(self) -> None:
        self._occupied, self._position = self._game.play_moves(self._move_str)
        self._turn = len(self._move_str) % 2

    def free_col(self, i_col: int) -> bool:
        return self._game.free_col(self._occupied, i_col)

    def winning_position(self) -> bool:
        return self._game.winning_position(self._position ^ self._occupied)

    def push(self, move_str: str) -> None:
        self._move_str += move_str
        self.update()

    def pop(self) -> None:
        self._move_str = self._move_str[:-1]
        self.update()

    def analyze(self) -> tuple[int, ...]:
        return self._game.analyze(self._occupied, self._position)


class ConnectFourBot:
    """Connect Four bot."""

    def __init__(self, state: ConnectFourState, screen: Screen) -> None:
        self._state = state
        self._screen = screen
        self._queue: Queue[None] = Queue()

        def play() -> None:
            while True:
                self._queue.get()
                move_str = self._state.move_str
                scores = self._state.analyze()
                best_score = max(scores)
                if best_score != self._state.nil_score:
                    idxs = []
                    for i_col, score in enumerate(scores):
                        if score == best_score:
                            idxs.append(i_col)
                    i_col = choice(idxs)
                    move_str += str(i_col + 1)
                    self._screen.event(("state", move_str))

        Thread(target=play, daemon=True).start()

    def queue(self) -> None:
        self._queue.put(None)


class ConnectFourMatch(Interactable):
    """Connect Four match."""

    def __init__(
        self, screen: Screen, bots: tuple[bool, bool] = (False, False)
    ) -> None:
        self._state = ConnectFourState()
        self._n_cols, self._n_rows = self._state.size
        move_order = []
        for i in range(self._n_cols):
            move_order.append(
                cdiv(self._n_cols, 2) + cdiv((1 - 2 * (i % 2)) * (i + 1), 2)
            )
        self._move_order = tuple(move_order)
        self._move_col = self._move_order[0]
        self._colors = (1, 3)
        self._status = ""
        self._finished = False
        self._bots = (
            ConnectFourBot(self._state, screen) if bots[0] else None,
            ConnectFourBot(self._state, screen) if bots[1] else None,
        )
        bot = self._bots[self._state.turn]
        if bot is not None:
            bot.queue()
        super().__init__(screen)
        screen.focus(self)

    def _empty_cell(self) -> str:
        return "\u25cb"  # \u25ef

    def _filled_cell(self, color: int) -> str:
        return f"\x1b[{30 + color}m\u25cf\x1b[0m"  # \u2b24

    def display(self) -> None:
        row_offset = (self._screen.rows - self._n_rows + 1) // 2 + 1
        col_offset = (self._screen.cols - 2 * self._n_cols) // 2 + 1
        if not self._finished:
            color = self._colors[self._state.turn]
            self._screen[row_offset, self._move_col * 2 + col_offset] = (
                self._filled_cell(color)
            )
        color1, color2 = self._colors
        if self._state.turn == 1:
            color1, color2 = color2, color1
        for i_row in range(self._n_rows):
            for i_col in range(self._n_cols):
                cell = 1 << (i_col * (self._n_rows + 1) + i_row)
                if self._state.occupied & cell:
                    color = color1 if self._state.position & cell else color2
                    disc = self._filled_cell(color)
                else:
                    disc = self._empty_cell()
                self._screen[
                    self._n_rows - i_row + row_offset, i_col * 2 + col_offset
                ] = disc
        for i, char in enumerate(self._status):
            self._screen[self._n_rows + 2 + row_offset, i + col_offset] = char

    def handle_event(self, event: tuple[str, ...]) -> None:
        move_str = ""

        if event[0] == "key" and self._bots[self._state.turn] is None:
            not_finished = not self._finished
            key = event[1]
            if key == "\x1b[C" and not_finished:
                for i_col in range(self._move_col + 1, self._n_cols):
                    if self._state.free_col(i_col):
                        self._move_col = i_col
                        break
            elif key == "\x1b[D" and not_finished:
                for i_col in range(self._move_col - 1, -1, -1):
                    if self._state.free_col(i_col):
                        self._move_col = i_col
                        break
            elif (
                key in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
                and not_finished
            ):
                i_col = int(key) - 1
                if i_col < self._n_cols and self._state.free_col(i_col):
                    self._move_col = i_col
            elif key == "\r" and not_finished:
                move_str = str(self._move_col + 1)
            elif key in {"\b", "\x7f"}:
                self._state.pop()
                if self._bots[self._state.turn] is not None:
                    self._state.pop()
                    bot = self._bots[self._state.turn]
                    if bot is not None:
                        bot.queue()
                self._status = ""
                self._finished = False
        elif event[0] == "state":
            string = event[1]
            if string[:-1] == self._state.move_str:
                move_str = string[-1]

        if len(move_str) > 0:
            self._state.push(move_str)
            if self._state.winning_position():
                self._status = (
                    self._filled_cell(self._colors[self._state.prev_turn])
                    + " Wins!"
                )
                self._finished = True
            else:
                for i_col in self._move_order:
                    if self._state.free_col(i_col):
                        self._move_col = i_col
                        break
                else:
                    self._status = "Draw!"
                    self._finished = True
            if not self._finished:
                bot = self._bots[self._state.turn]
                if bot is not None:
                    bot.queue()
