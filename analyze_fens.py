import csv
from datasets import load_dataset
import chess
import chess.engine

DATASET_NAME = "bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
STOCKFISH_PATH = r"C:\Users\samir\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
OUTPUT_CSV = "fenstockfish_results.csv"
DEPTH = 15

def score_from_perspective(pov_score, color):
    return pov_score.pov(color).score(mate_score=100000)

def main():
    dataset = load_dataset(DATASET_NAME, split="train")
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    try:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "fen",
                    "dataset_next_move",
                    "side_to_move",
                    "stockfish_bestmove",
                    "score_before",
                    "score_after",
                    "delta",
                ],
            )
            writer.writeheader()

            for i, row in enumerate(dataset, start=1):
                fen = row["FEN"]
                dataset_next_move = row.get("Next move", "")

                board = chess.Board(fen)
                side = board.turn

                info_before = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                bestmove = info_before["pv"][0] if info_before.get("pv") else None
                if bestmove is None:
                    continue

                score_before = score_from_perspective(info_before["score"], side)

                board_after = board.copy()
                board_after.push(bestmove)

                info_after = engine.analyse(board_after, chess.engine.Limit(depth=DEPTH))
                score_after = score_from_perspective(info_after["score"], side)

                delta = score_after - score_before if score_before is not None else None

                writer.writerow(
                    {
                        "fen": fen,
                        "dataset_next_move": dataset_next_move,
                        "side_to_move": "white" if side else "black",
                        "stockfish_bestmove": bestmove.uci(),
                        "score_before": score_before,
                        "score_after": score_after,
                        "delta": delta,
                    }
                )

                if i % 500 == 0:
                    print(f"Processed {i} positions")

    finally:
        engine.quit()

if __name__ == "__main__":
    main()