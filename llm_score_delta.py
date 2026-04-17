import csv
import chess
import chess.engine

STOCKFISH_PATH = r"C:\Users\samir\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
FENSTOCKFISH_CSV = "fenstockfish_results.csv"
LLM_INFERENCE_CSV = "llm_inference_results.csv"
OUTPUT_CSV = "llm_score_delta_results.csv"
DEPTH = 15


def score_from_perspective(pov_score, color):
    return pov_score.pov(color).score(mate_score=100000)


def load_stockfish_results(path):
    results = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            results[row["fen"]] = {
                "score_before": int(row["score_before"]) if row["score_before"] else None,
                "stockfish_score_after": int(row["score_after"]) if row["score_after"] else None,
                "stockfish_delta": int(row["delta"]) if row["delta"] else None,
            }
    return results


def main():
    sf_data = load_stockfish_results(FENSTOCKFISH_CSV)

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        with open(LLM_INFERENCE_CSV, newline="", encoding="utf-8") as infile, \
             open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:

            reader = csv.DictReader(infile)
            fieldnames = [
                "fen",
                "dataset_next_move",
                "side_to_move",
                "llm_bestmove",
                "llm_move_valid",
                "score_before",
                "stockfish_score_after",
                "stockfish_delta",
                "stockfish_standardized_delta",
                "llm_score_after",
                "llm_delta",
                "llm_standardized_delta",
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for i, row in enumerate(reader, start=1):
                fen = row["fen"]
                side_to_move = row["side_to_move"]
                llm_move = row["llm_bestmove"]
                llm_move_valid = row["llm_move_valid"] == "True"

                sf = sf_data.get(fen, {})
                score_before = sf.get("score_before")
                stockfish_score_after = sf.get("stockfish_score_after")
                stockfish_delta = sf.get("stockfish_delta")

                # Standardize Stockfish delta to be from mover's perspective
                # (positive = mover gained, regardless of color)
                if stockfish_delta is not None:
                    if side_to_move == "black":
                        stockfish_standardized_delta = -stockfish_delta
                    else:
                        stockfish_standardized_delta = stockfish_delta
                else:
                    stockfish_standardized_delta = None

                llm_score_after = None
                llm_delta = None
                llm_standardized_delta = None

                if llm_move_valid and llm_move and score_before is not None:
                    try:
                        board = chess.Board(fen)
                        side = board.turn
                        move = chess.Move.from_uci(llm_move)

                        if move in board.legal_moves:
                            board.push(move)
                            info_after = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                            llm_score_after = score_from_perspective(info_after["score"], side)
                            llm_delta = llm_score_after - score_before
                            if side_to_move == "black":
                                llm_standardized_delta = -llm_delta
                            else:
                                llm_standardized_delta = llm_delta
                    except Exception as e:
                        print(f"Row {i}: error evaluating LLM move '{llm_move}' on FEN '{fen}': {e}")

                writer.writerow({
                    "fen": fen,
                    "dataset_next_move": row["dataset_next_move"],
                    "side_to_move": side_to_move,
                    "llm_bestmove": llm_move,
                    "llm_move_valid": row["llm_move_valid"],
                    "score_before": score_before,
                    "stockfish_score_after": stockfish_score_after,
                    "stockfish_delta": stockfish_delta,
                    "stockfish_standardized_delta": stockfish_standardized_delta,
                    "llm_score_after": llm_score_after,
                    "llm_delta": llm_delta,
                    "llm_standardized_delta": llm_standardized_delta,
                })

                if i % 100 == 0:
                    print(f"Processed {i} rows")

    finally:
        engine.quit()

    print(f"Done. Results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
