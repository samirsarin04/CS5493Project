import csv

INPUT_CSV = "fenstockfish_results.csv"
OUTPUT_CSV = "fenstockfish_results_standardized.csv"

def main():
    with open(INPUT_CSV, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["standardized_delta"]

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                delta = row["delta"]
                side = row["side_to_move"]

                if delta == "" or delta is None:
                    row["standardized_delta"] = ""
                else:
                    delta_val = int(delta)
                    # Scores are from White's perspective throughout.
                    # For Black moves, negate so positive = good for the mover.
                    if side == "black":
                        row["standardized_delta"] = -delta_val
                    else:
                        row["standardized_delta"] = delta_val

                writer.writerow(row)

    print(f"Written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
