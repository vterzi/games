#!/usr/bin/env python3

from sys import argv

from connectfour import Screen, ConnectFourMatch

info = """\
Usage:
  python connect4.py <disc-width> <bot-first> <bot-second> <analyze>

Arguments:
  <disc-width>  disc width in the CLI (1 or 2)
  <bot-first>   first player is a bot (1) or human (0)
  <bot-second>  second player is a bot (1) or human (0)
  <analyze>     enable (1) or disable (0) move analysis
"""

if __name__ == "__main__":
    n_args = len(argv)
    if (
        n_args == 5
        and argv[1] in {"1", "2"}
        and argv[2] in {"0", "1"}
        and argv[3] in {"0", "1"}
        and argv[4] in {"0", "1"}
    ):
        screen = Screen(60)
        ConnectFourMatch(
            screen,
            int(argv[1]),
            (bool(int(argv[2])), bool(int(argv[3]))),
            bool(int(argv[4])),
        )
        screen.run()
    else:
        print(info)
