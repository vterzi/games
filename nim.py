from sys import argv

if __name__ == "__main__":
    if len(argv) == 3 and all(
        arg.isdigit() and int(arg) > 0 for arg in argv[1:]
    ):
        n_rows = int(argv[1])
        move_limit = int(argv[2])
        rows = [2 * i + 1 for i in range(n_rows)]
        width = rows[-1]
        pads = [" " * (n_rows - i - 1) for i in range(n_rows)]
        i = 0
        idxs = ("1st", "2nd")
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
                    num = input(f"{idxs[i]} player: ")
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
            print(f"{idxs[i]} player wins!")
    else:
        print("Usage: python nim.py <number-of-rows> <move-limit>")

