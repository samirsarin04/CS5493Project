import csv
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "fuzz_fens.csv")
RUN_DIRS = ["run1", "run2", "run3"]


def collect_fens():
    rows = []
    deleted = 0
    skipped = 0

    for run in RUN_DIRS:
        run_path = os.path.join(DATA_DIR, run)
        if not os.path.isdir(run_path):
            print(f"Warning: {run_path} not found, skipping.")
            continue

        for fname in os.listdir(run_path):
            fpath = os.path.join(run_path, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                with open(fpath, "r", errors="replace") as f:
                    content = f.read().strip()
            except Exception as e:
                print(f"  Could not read {fpath}: {e}")
                skipped += 1
                continue

            # Each file is a single FEN line; skip empty files
            fen = content.splitlines()[0].strip() if content else ""
            if not fen:
                skipped += 1
                os.remove(fpath)
                deleted += 1
                continue

            rows.append({"run": run, "fen": fen})
            os.remove(fpath)
            deleted += 1

    return rows, deleted, skipped


def main():
    print(f"Scanning {DATA_DIR} ...")
    rows, deleted, skipped = collect_fens()

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "fen"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} FEN entries to {OUTPUT_CSV}")
    print(f"Deleted {deleted} source files  |  Skipped (unreadable/empty): {skipped}")


if __name__ == "__main__":
    main()
