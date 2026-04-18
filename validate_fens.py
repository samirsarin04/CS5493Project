import csv
import re

INPUT_CSV = "data/fuzz_fens.csv"

PIECE_CHARS = set("prnbqkPRNBQK")


def validate_fen(fen: str) -> tuple[bool, str]:
    parts = fen.split()
    if len(parts) != 6:
        return False, f"expected 6 fields, got {len(parts)}"

    board, active, castling, en_passant, halfmove, fullmove = parts

    # --- Board ---
    ranks = board.split("/")
    if len(ranks) != 8:
        return False, f"board must have 8 ranks, got {len(ranks)}"

    kings = {"k": 0, "K": 0}
    for rank in ranks:
        count = 0
        for ch in rank:
            if ch.isdigit():
                count += int(ch)
            elif ch in PIECE_CHARS:
                count += 1
                if ch in kings:
                    kings[ch] += 1
            else:
                return False, f"invalid piece character '{ch}'"
        if count != 8:
            return False, f"rank '{rank}' sums to {count}, expected 8"

    if kings["k"] != 1:
        return False, f"black king count is {kings['k']}, expected 1"
    if kings["K"] != 1:
        return False, f"white king count is {kings['K']}, expected 1"

    # --- Active color ---
    if active not in ("w", "b"):
        return False, f"invalid active color '{active}'"

    # --- Castling ---
    if castling != "-" and not re.fullmatch(r"K?Q?k?q?", castling):
        return False, f"invalid castling field '{castling}'"

    # --- En passant ---
    if en_passant != "-" and not re.fullmatch(r"[a-h][36]", en_passant):
        return False, f"invalid en passant square '{en_passant}'"

    # --- Halfmove clock ---
    if not halfmove.isdigit() or int(halfmove) < 0:
        return False, f"invalid halfmove clock '{halfmove}'"

    # --- Fullmove number ---
    if not fullmove.isdigit() or int(fullmove) < 1:
        return False, f"invalid fullmove number '{fullmove}'"

    return True, "ok"


def main():
    total = valid = 0
    invalid_examples = []

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fen = row["fen"].strip()
            total += 1
            ok, reason = validate_fen(fen)
            if ok:
                valid += 1
            elif len(invalid_examples) < 10:
                invalid_examples.append((fen, reason))

    invalid = total - valid
    print(f"Total FENs : {total}")
    print(f"Valid      : {valid}  ({valid/total*100:.1f}%)")
    print(f"Invalid    : {invalid}  ({invalid/total*100:.1f}%)")

    if invalid_examples:
        print("\nSample invalid FENs:")
        for fen, reason in invalid_examples:
            print(f"  [{reason}]  {fen}")


if __name__ == "__main__":
    main()
