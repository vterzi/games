from threading import Thread
from queue import Queue
from random import choice
from re import fullmatch
from typing import Generator

from .screen import Interactable, Screen
from .solver import ConnectFourSolver  # type: ignore


class ConnectFourState:
    def __init__(self) -> None:
        path = __file__.split("/")
        path[-1] = "opening.txt"
        self._game = ConnectFourSolver("/".join(path))
        self._n_cols, self._n_rows = self._game.size()
        self._move_str = ""
        self._occupied = 0
        self._position = 0
        self._turn = 0
        self.update()

    @property
    def n_cols(self) -> int:
        return self._n_cols

    @property
    def n_rows(self) -> int:
        return self._n_rows

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

    def nearest_cols(self, i_col: int) -> Generator[int, None, None]:
        n_cols = self._n_cols
        yield i_col
        for i in range(1, n_cols):
            next_i_col = i_col - i
            if next_i_col >= 0:
                yield next_i_col
            next_i_col = i_col + i
            if next_i_col < n_cols:
                yield next_i_col

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
        self._n_cols = self._state.n_cols
        self._n_rows = self._state.n_rows
        self._move_col = (self._n_cols + 1) // 2 - 1
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
        state = self._state
        n_rows = state.n_rows
        n_cols = state.n_cols
        screen = self._screen
        colors = self._colors
        row_offset = (screen.rows - n_rows + 1) // 2 + 1
        col_offset = (screen.cols - 2 * n_cols) // 2 + 1
        if not self._finished:
            color = colors[state.turn]
            screen[row_offset, self._move_col * 2 + col_offset] = (
                self._filled_cell(color)
            )
        color1, color2 = colors
        if state.turn == 1:
            color1, color2 = color2, color1
        for i_row in range(n_rows):
            for i_col in range(n_cols):
                cell = 1 << (i_col * (n_rows + 1) + i_row)
                if state.occupied & cell:
                    color = color1 if state.position & cell else color2
                    disc = self._filled_cell(color)
                else:
                    disc = self._empty_cell()
                screen[n_rows - i_row + row_offset, i_col * 2 + col_offset] = (
                    disc
                )
        for i, char in enumerate(self._status):
            screen[n_rows + 2 + row_offset, i + col_offset] = char

    def handle_event(self, event: tuple[str, ...]) -> None:
        state = self._state
        bots = self._bots
        move_str = ""

        if event[0] == "key" and bots[state.turn] is None:
            n_rows = state.n_rows
            n_cols = state.n_cols
            not_finished = not self._finished
            key = event[1]
            if key == "\x1b[C" and not_finished:
                for i_col in range(self._move_col + 1, n_cols):
                    if state.free_col(i_col):
                        self._move_col = i_col
                        break
            elif key == "\x1b[D" and not_finished:
                for i_col in range(self._move_col - 1, -1, -1):
                    if state.free_col(i_col):
                        self._move_col = i_col
                        break
            elif (
                key in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
                and not_finished
            ):
                i_col = int(key) - 1
                if i_col < n_cols and state.free_col(i_col):
                    self._move_col = i_col
            elif (
                key.startswith("\x1b[<0;")
                and key.endswith("m")
                and not_finished
            ):
                key = key[5:-1]
                match = fullmatch(r"(\d+);(\d+)", key)
                if match is not None:
                    i_col = int(match.group(1))
                    i_row = int(match.group(2))
                    screen = self._screen
                    row_offset = (screen.rows - n_rows + 1) // 2 + 1
                    col_offset = (screen.cols - 2 * n_cols) // 2 + 1
                    i_row -= row_offset
                    i_col -= col_offset
                    i_col //= 2
                    if 0 <= i_row <= n_rows and 0 <= i_col < n_cols:
                        if state.free_col(i_col):
                            self._move_col = i_col
                            move_str = str(i_col + 1)
            elif key == "\r" and not_finished:
                move_str = str(self._move_col + 1)
            elif key in {"\b", "\x7f"}:
                state.pop()
                if bots[state.turn] is not None:
                    state.pop()
                    bot = bots[state.turn]
                    if bot is not None:
                        bot.queue()
                self._status = ""
                self._finished = False
        elif event[0] == "state":
            string = event[1]
            if string[:-1] == state.move_str:
                char = string[-1]
                self._move_col = int(char) - 1
                move_str = char

        if len(move_str) > 0:
            state.push(move_str)
            if state.winning_position():
                self._status = (
                    self._filled_cell(self._colors[state.prev_turn]) + " Wins!"
                )
                self._finished = True
            else:
                for i_col in state.nearest_cols(self._move_col):
                    if state.free_col(i_col):
                        self._move_col = i_col
                        break
                else:
                    self._status = "Draw!"
                    self._finished = True
            if not self._finished:
                bot = bots[state.turn]
                if bot is not None:
                    bot.queue()
