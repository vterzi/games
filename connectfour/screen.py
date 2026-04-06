"""Classes for drawing."""

__all__ = ["Displayable", "Screen"]

from abc import ABC, abstractmethod
from sys import platform, stdin
from shutil import get_terminal_size
from signal import signal, SIGWINCH
from threading import Thread, Event
from queue import Queue
from time import time, sleep
from types import FrameType

if platform == "win32":
    from msvcrt import getch  # type: ignore

    def get_stdin_attrs() -> list:
        """Get the TTY attributes of the standard input."""
        return []

    def set_stdin_attrs(attrs: list) -> None:
        """Set the TTY attributes of the standard input."""
        pass

    def set_stdin_raw() -> None:
        """Set the mode of the standard input to raw."""
        pass

    def get_key() -> str:
        """Read a keypress."""
        key = getch()
        if key in {b"\x00", b"\xe0"}:
            key += getch()
        return key.decode(errors="ignore")

else:
    from tty import setraw
    from termios import tcgetattr, tcsetattr, TCSADRAIN

    STDIN_FD = stdin.fileno()

    def get_stdin_attrs() -> list:
        """Get the TTY attributes of the standard input."""
        return tcgetattr(STDIN_FD)

    def set_stdin_attrs(attrs: list) -> None:
        """Set the TTY attributes of the standard input."""
        tcsetattr(STDIN_FD, TCSADRAIN, attrs)

    def set_stdin_raw() -> None:
        """Set the mode of the standard input to raw."""
        setraw(STDIN_FD)

    def get_key() -> str:
        """Read a keypress."""
        key = stdin.read(1)
        if key == "\x1b":
            char = stdin.read(1)
            key += char
            if char in {"[", "O"}:
                while True:
                    char = stdin.read(1)
                    key += char
                    if char.isalpha() or char == "~":
                        # if char == "M":
                        #     key += stdin.buffer.read(3).decode(
                        #         errors="ignore",
                        #     )
                        break
        return key


class Displayable(ABC):
    """Displayable object."""

    def __init__(self, screen: "Screen") -> None:
        self._screen = screen
        screen.add(self)

    @abstractmethod
    def display(self) -> None:
        """Display the object."""


class Interactable(Displayable):
    """Interactable object."""

    @abstractmethod
    def handle_event(self, event: tuple[str, ...]) -> None:
        """Handle event."""


class Screen:
    """Alternative screen buffer."""

    @classmethod
    def _print(cls, text: str) -> None:
        """Print text."""
        print(text, end="", flush=True)

    def __init__(self, fps: float) -> None:
        self._cols = 0
        self._rows = 0
        self._buffer: list[str] = []
        self._objects: list[Displayable] = []
        self._focus: Interactable | None = None
        self._fps = fps
        self._events: Queue[tuple[str, ...]] = Queue()

    @property
    def cols(self) -> int:
        """Number of columns."""
        return self._cols

    @property
    def rows(self) -> int:
        """Number of rows."""
        return self._rows

    def _idx(self, row: int, col: int) -> int:
        """Get the buffer index for a one-based position (row, column)."""
        cols = self._cols
        rows = self._rows
        return (
            col - 1 + (row - 1) * cols
            if 1 <= col <= cols and 1 <= row <= rows
            else -1
        )

    def __getitem__(self, key: tuple[int, int]) -> str:
        idx = self._idx(key[0], key[1])
        return self._buffer[idx] if idx >= 0 else ""

    def __setitem__(self, key: tuple[int, int], value: str) -> None:
        idx = self._idx(key[0], key[1])
        if idx >= 0:
            self._buffer[idx] = value

    def add(self, obj: Displayable) -> None:
        """Add a displayable object."""
        self._objects.append(obj)

    def remove(self, obj: Displayable) -> None:
        """Remove a displayable object."""
        self._objects.remove(obj)

    def focus(self, obj: Interactable | None) -> None:
        """Focus a displayable object."""
        if obj in self._objects or obj is None:
            self._focus = obj

    def event(self, event: tuple[str, ...]) -> None:
        """Put an event in the event queue."""
        self._events.put(event)

    def clear(self) -> None:
        """Clear the buffer."""
        buffer = self._buffer
        for i in range(len(buffer)):
            buffer[i] = " "

    def display(self) -> None:
        """Display the buffer."""
        self.clear()
        for obj in self._objects:
            obj.display()
        self._print("\x1b[H" + "".join(self._buffer))

    def run(self) -> None:
        """Run the main loop."""
        events = self._events
        stop = Event()
        spf = 1 / self._fps

        self._stdin_attrs = get_stdin_attrs()
        set_stdin_raw()
        self._print("\x1b[?1049h\x1b[?25l\x1b[?1000h\x1b[?1003h\x1b[?1006h")

        def resize_handler(signum: int, frame: FrameType | None) -> None:
            size = get_terminal_size()
            cols = size.columns
            rows = size.lines
            self._cols = cols
            self._rows = rows
            self._buffer = [" "] * (cols * rows)

        resize_handler(int(SIGWINCH), None)
        signal(SIGWINCH, resize_handler)

        def listen_keys() -> None:
            while not stop.is_set():
                key = get_key()
                events.put(("key", key))

        Thread(target=listen_keys, daemon=True).start()

        try:
            while not stop.is_set():
                initial = time()
                while not events.empty():
                    event = events.get()
                    if event == ("key", "\x1b\x1b"):
                        stop.set()
                    elif self._focus is not None:
                        self._focus.handle_event(event)
                self.display()
                elapsed = time() - initial
                sleep(max(0, spf - elapsed))
        finally:
            self._print(
                "\x1b[?1049l\x1b[?25h\x1b[?1000l\x1b[?1003l\x1b[?1006l"
            )
            set_stdin_attrs(self._stdin_attrs)
