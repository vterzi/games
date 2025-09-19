from sys import argv

info = """\
Usage:
  python nim.py <rows> <limit> [<first>]

Arguments:
  <rows>   number of heaps (> 0)
  <limit>  maximum objects removable per move (> 0)
  <first>  if present, play first (1) or second (0) against a bot
"""

if __name__ == "__main__":
    n_args = len(argv)
    enable_AI = n_args == 4
    if (
        n_args in (3, 4)
        and all(arg.isdigit() and int(arg) > 0 for arg in argv[1:3])
        and (not enable_AI or argv[3] in ("0", "1"))
    ):
        n_rows = int(argv[1])
        move_limit = int(argv[2])
        rows = [2 * i + 1 for i in range(n_rows)]
        width = rows[-1]
        pads = [" " * (n_rows - i - 1) for i in range(n_rows)]
        if enable_AI:
            i = int(argv[3])
            players = ("Bot", "Player")
        else:
            i = 0
            players = ("1st player", "2nd player")
        step = move_limit + 1
        print("\x1b[?1049h", end="")
        try:
            while len(rows) > 0:
                num = ""
                while not num.isdigit() or not (
                    1 <= int(num) <= min(move_limit, rows[0])
                ):
                    print("\x1b[H\x1b[J" + "\n" * (n_rows - len(rows)), end="")
                    for row, pad in zip(rows, pads):
                        string = "|" * row + pad
                        print(" " * (width - len(string)) + string + f" {row}")
                    if enable_AI and i == 0:
                        move = rows[0]
                        if len(rows) == 1 or (rows[1] - 1) % step != 0:
                            move = move - 1
                        move = max(move - move // step * step, 1)
                        num = str(move)
                    else:
                        num = input(f"{players[i]}: ")
                i = (i + 1) % 2
                rows[0] -= int(num)
                if rows[0] == 0:
                    del rows[0]
                    del pads[0]
        except KeyboardInterrupt:
            pass
        finally:
            print("\x1b[?1049l", end="")
        if len(rows) == 0:
            print(f"{players[i]} wins!")
    else:
        print(info)
