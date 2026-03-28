from sys import argv

from connectfour import Screen, ConnectFourMatch

info = """\
Usage:
  python connect4.py <bot-first> <bot-second>

Arguments:
  <bot-first>   first player is a bot (1) or human (0)
  <bot-second>  second player is a bot (1) or human (0)
"""

if __name__ == "__main__":
    n_args = len(argv)
    if n_args == 3 and argv[1] in {"0", "1"} and argv[2] in {"0", "1"}:
        screen = Screen(60)
        ConnectFourMatch(screen, (bool(int(argv[1])), bool(int(argv[2]))))
        screen.run()
    else:
        print(info)
