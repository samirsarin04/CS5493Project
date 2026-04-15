"""
Run this locally (where you have internet) to download and save the dataset
as a CSV. Commit the CSV to the repo and use it on the cluster.
"""

import csv
from datasets import load_dataset

DATASET_NAME = "bonna46/Chess-FEN-and-NL-Format-30K-Dataset"
OUTPUT_CSV   = "data/chess_fens.csv"

def main():
    import os
    os.makedirs("data", exist_ok=True)

    print(f"Downloading {DATASET_NAME} …")
    dataset = load_dataset(DATASET_NAME, split="train")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["FEN", "Next move"])
        writer.writeheader()
        for row in dataset:
            writer.writerow({"FEN": row["FEN"], "Next move": row.get("Next move", "")})

    print(f"Saved {len(dataset)} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
