from .screen import Screen
from .match import ConnectFourMatch


def main() -> None:
    screen = Screen()
    try:
        ConnectFourMatch(screen)
        screen.listen_keys()
    finally:
        screen.close()
