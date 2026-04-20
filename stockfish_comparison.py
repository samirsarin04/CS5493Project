"""
Compare Stockfish's best move against each LLM's predicted move
for all valid FENs in the fuzz and chaotic datasets.

Output: stockfish_comparison.csv
"""

import chess
import csv
import os
import queue
import subprocess
import threading

STOCKFISH_PATH = r"C:\Users\samir\Downloads\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"
OUTPUT_CSV = "data/stockfish_comparison.csv"
MOVETIME_MS = 1000  # ms per position

MODELS = ["qwen25", "llama1b", "llama8b", "chessSLM"]
MODEL_LABELS = {
    "qwen25":   "Qwen2.5-7B",
    "llama1b":  "Llama-3.2-1B",
    "llama8b":  "Meta-Llama-3-8B",
    "chessSLM": "ChessSLM",
}

RESULT_FILES = {
    "fuzz": {
        "qwen25":   "data/llm_results_qwen25.csv",
        "llama1b":  "data/llm_results_llama1b.csv",
        "llama8b":  "data/llm_results_llama8b.csv",
        "chessSLM": "data/llm_fuzz_results_chessSLM.csv",
    },
    "chaotic": {
        "qwen25":   "data/llm_chaotic_results_qwen25.csv",
        "llama1b":  "data/llm_chaotic_results_llama1b.csv",
        "llama8b":  "data/llm_chaotic_results_llama8b.csv",
        "chessSLM": "data/llm_chaotic_results_chessSLM.csv",
    },
}

FIELDNAMES = [
    "dataset",
    "fen",
    "stockfish_move",
    "qwen25_move",
    "llama1b_move",
    "llama8b_move",
    "chessSLM_move",
    "qwen25_matches_sf",
    "llama1b_matches_sf",
    "llama8b_matches_sf",
    "chessSLM_matches_sf",
]


def _read_lines(stdout, q):
    for line in stdout:
        q.put(line)


def get_stockfish_move(proc, fen: str, line_queue: queue.Queue) -> str:
    proc.stdin.write(f"position fen {fen}\n")
    proc.stdin.write(f"go movetime {MOVETIME_MS}\n")
    proc.stdin.flush()

    timeout = MOVETIME_MS / 1000 + 5  # extra buffer beyond movetime
    while True:
        try:
            line = line_queue.get(timeout=timeout).strip()
        except queue.Empty:
            print(f"  Timeout on FEN: {fen[:60]}... — skipping")
            return ""
        if line.startswith("bestmove"):
            parts = line.split()
            return parts[1] if len(parts) > 1 and parts[1] != "(none)" else ""


def load_valid_fens(dataset: str) -> dict[str, dict[str, str]]:
    """Returns {fen: {model: llm_bestmove}} for all valid FENs."""
    fen_map: dict[str, dict[str, str]] = {}
    for model, path in RESULT_FILES[dataset].items():
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping.")
            continue
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                if row.get("fen_valid", "").strip().lower() != "true":
                    continue
                fen = row["fen"].strip()
                if fen not in fen_map:
                    fen_map[fen] = {}
                fen_map[fen][model] = row.get("llm_bestmove", "").strip()
    return fen_map


def start_stockfish() -> tuple[subprocess.Popen, queue.Queue]:
    proc = subprocess.Popen(
        [STOCKFISH_PATH],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    # drain until ready
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if "uciok" in line:
            break
    proc.stdin.write("isready\n")
    proc.stdin.flush()
    for line in proc.stdout:
        if "readyok" in line:
            break

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=_read_lines, args=(proc.stdout, q), daemon=True)
    t.start()
    return proc, q


def main():
    proc, line_queue = start_stockfish()
    print("Stockfish ready.")

    rows = []
    for dataset in ("fuzz", "chaotic"):
        fen_map = load_valid_fens(dataset)
        print(f"\n{dataset}: {len(fen_map)} unique valid FENs")

        for i, (fen, model_moves) in enumerate(fen_map.items(), 1):
            # Skip positions where the non-moving side is in check — Stockfish
            # treats these as illegal and may crash or hang.
            try:
                board = chess.Board(fen)
                board.turn = not board.turn
                if board.is_check():
                    sf_move = ""
                    board.turn = not board.turn
                    rows.append({"dataset": dataset, "fen": fen, "stockfish_move": "",
                                 **{f"{m}_move": model_moves.get(m, "") for m in MODELS},
                                 **{f"{m}_matches_sf": False for m in MODELS}})
                    continue
            except Exception:
                pass

            try:
                sf_move = get_stockfish_move(proc, fen, line_queue)
            except OSError:
                sf_move = ""

            # Only restart if the process actually died, not on legitimate empty moves
            if proc.poll() is not None:
                print("  Stockfish crashed, restarting...")
                try:
                    proc.kill()
                except Exception:
                    pass
                proc, line_queue = start_stockfish()

            row = {
                "dataset":  dataset,
                "fen":      fen,
                "stockfish_move": sf_move,
            }
            for model in MODELS:
                llm_move = model_moves.get(model, "")
                row[f"{model}_move"] = llm_move
                row[f"{model}_matches_sf"] = (
                    llm_move != "" and llm_move == sf_move
                )

            rows.append(row)

            if i % 50 == 0:
                print(f"  Processed {i}/{len(fen_map)}", flush=True)

    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTPUT_CSV}")

    # Print summary
    print("\n--- Agreement with Stockfish ---")
    for dataset in ("fuzz", "chaotic"):
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        total = len(dataset_rows)
        print(f"\n  {dataset.upper()} ({total} valid FENs)")
        for model in MODELS:
            matches = sum(1 for r in dataset_rows if r[f"{model}_matches_sf"] is True)
            print(f"    {MODEL_LABELS[model]:<16}: {matches}/{total} ({matches/total*100:.1f}%)")


if __name__ == "__main__":
    main()
