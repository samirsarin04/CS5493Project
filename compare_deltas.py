import csv
import statistics

DELTA_CSV = "llm_score_delta_results.csv"
SF_CSV = "fenstockfish_results.csv"


def load_sf_bestmoves(path):
    bestmoves = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            bestmoves[row["fen"]] = row["stockfish_bestmove"]
    return bestmoves


def main():
    sf_bestmoves = load_sf_bestmoves(SF_CSV)

    all_rows = []
    valid_rows = []  # rows where LLM played a valid move

    with open(DELTA_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            all_rows.append(row)
            if row["llm_standardized_delta"] != "":
                fen = row["fen"]
                sf_move = sf_bestmoves.get(fen, "")
                llm_move = row["llm_bestmove"]
                valid_rows.append({
                    "fen": fen,
                    "side_to_move": row["side_to_move"],
                    "sf_move": sf_move,
                    "llm_move": llm_move,
                    "matched": llm_move == sf_move,
                    "sf_delta": int(row["stockfish_standardized_delta"]),
                    "llm_delta": int(row["llm_standardized_delta"]),
                    "gap": int(row["llm_standardized_delta"]) - int(row["stockfish_standardized_delta"]),
                })

    total = len(all_rows)
    valid = len(valid_rows)
    invalid = total - valid

    matched = sum(1 for r in valid_rows if r["matched"])
    not_matched = valid - matched

    sf_deltas = [r["sf_delta"] for r in valid_rows]
    llm_deltas = [r["llm_delta"] for r in valid_rows]
    gaps = [r["gap"] for r in valid_rows]  # llm_delta - sf_delta (negative = LLM worse)

    print("=" * 60)
    print("LLM ACCURACY vs STOCKFISH BASELINE")
    print("=" * 60)

    print(f"\nDataset overview")
    print(f"  Total positions       : {total}")
    print(f"  LLM valid moves       : {valid}  ({100 * valid / total:.1f}%)")
    print(f"  LLM invalid moves     : {invalid}  ({100 * invalid / total:.1f}%)")

    if not valid_rows:
        print("\nNo valid LLM moves to compare.")
        return

    print(f"\nMove accuracy  (valid moves only, n={valid})")
    print(f"  LLM matched Stockfish : {matched:>5}  ({100 * matched / valid:.1f}%)")
    print(f"  LLM different move    : {not_matched:>5}  ({100 * not_matched / valid:.1f}%)")

    print(f"\nScore delta vs Stockfish baseline  (positive = mover gained)")
    print(f"  {'Metric':<30} {'Stockfish':>12} {'LLM':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Mean delta':<30} {statistics.mean(sf_deltas):>12.1f} {statistics.mean(llm_deltas):>12.1f}")
    print(f"  {'Median delta':<30} {statistics.median(sf_deltas):>12.1f} {statistics.median(llm_deltas):>12.1f}")
    print(f"  {'Stdev delta':<30} {statistics.stdev(sf_deltas):>12.1f} {statistics.stdev(llm_deltas):>12.1f}")
    print(f"  {'Min delta':<30} {min(sf_deltas):>12} {min(llm_deltas):>12}")
    print(f"  {'Max delta':<30} {max(sf_deltas):>12} {max(llm_deltas):>12}")

    print(f"\nLLM gap from Stockfish  (llm_delta - sf_delta, valid moves only)")
    print(f"  Mean gap              : {statistics.mean(gaps):>+.1f}  (negative = LLM worse on average)")
    print(f"  Median gap            : {statistics.median(gaps):>+.1f}")
    print(f"  Stdev gap             : {statistics.stdev(gaps):.1f}")

    # Worst LLM moves (biggest negative gap from stockfish)
    worst = sorted(valid_rows, key=lambda r: r["gap"])[:5]
    print(f"\nTop 5 worst LLM moves  (largest gap below Stockfish)")
    print(f"  {'SF move':<10} {'LLM move':<10} {'SF delta':>10} {'LLM delta':>10} {'Gap':>8}  FEN")
    for r in worst:
        print(f"  {r['sf_move']:<10} {r['llm_move']:<10} {r['sf_delta']:>10} {r['llm_delta']:>10} {r['gap']:>+8}  {r['fen'][:55]}")

    print("=" * 60)


if __name__ == "__main__":
    main()
