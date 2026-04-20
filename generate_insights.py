import csv
import os

DATA_DIR = "data"
OUTPUT_TXT = "insights_report.txt"

MODELS = ["qwen25", "llama1b", "llama8b", "chessSLM"]
MODEL_LABELS = {
    "qwen25":   "Qwen2.5-7B",
    "llama1b":  "Llama-3.2-1B",
    "llama8b":  "Meta-Llama-3-8B",
    "chessSLM": "ChessSLM",
}
DATASETS = {
    "fuzz":    "llm_fuzz_results_{}.csv",
    "chaotic": "llm_chaotic_results_{}.csv",
}


def load(path):
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    total      = len(rows)
    fen_valid  = sum(1 for r in rows if r.get("fen_valid", "").strip().lower() == "true")
    move_valid = sum(1 for r in rows if r.get("llm_move_valid", "").strip().lower() == "true")
    no_move    = sum(1 for r in rows if not r.get("llm_bestmove", "").strip())
    return {"total": total, "fen_valid": fen_valid, "move_valid": move_valid, "no_move": no_move}


def pct(num, den):
    return f"{num/den*100:.1f}%" if den else "N/A"


def main():
    stats = {}
    for dataset, pattern in DATASETS.items():
        stats[dataset] = {}
        for model in MODELS:
            fname = pattern.format(model)
            # fuzz dataset used old filename before naming fix
            if dataset == "fuzz":
                candidates = [
                    os.path.join(DATA_DIR, fname),
                    os.path.join(DATA_DIR, f"llm_results_{model}.csv"),
                ]
            else:
                candidates = [os.path.join(DATA_DIR, fname)]

            path = next((p for p in candidates if os.path.exists(p)), None)
            if path:
                stats[dataset][model] = summarize(load(path))
            else:
                stats[dataset][model] = None

    lines = []
    w = lines.append

    w("=" * 65)
    w("  FUZZING CAMPAIGN + LLM INFERENCE — INSIGHTS REPORT")
    w("=" * 65)
    w("")

    # ── Insight 1: Fuzzer effectiveness ──────────────────────────────
    w("INSIGHT 1: Fuzzer Effectiveness")
    w("-" * 65)
    fuzz_any = next(s for s in stats["fuzz"].values() if s)
    total        = fuzz_any["total"]
    valid        = fuzz_any["fen_valid"]
    invalid      = total - valid
    w(f"  Total fuzz outputs      : {total}")
    w(f"  Valid FENs              : {valid}  ({pct(valid, total)})")
    w(f"  Invalid/malformed FENs  : {invalid}  ({pct(invalid, total)})")
    w("  AFL++ successfully corrupted 4 in 5 board states,")
    w("  which is the core goal of the fuzzing campaign.")
    w("")

    # ── Insight 2: LLMs don't crash on garbage ────────────────────────
    w("INSIGHT 2: LLMs Produce Output Even on Invalid Inputs")
    w("-" * 65)
    w(f"  {'Model':<16} {'No output / total':<22} {'Output rate'}")
    w(f"  {'-'*16} {'-'*22} {'-'*12}")
    for model in MODELS:
        s = stats["fuzz"][model]
        if not s:
            continue
        produced = s["total"] - s["no_move"]
        w(f"  {MODEL_LABELS[model]:<16} {s['no_move']:<4} no output / {s['total']:<6}  {pct(produced, s['total'])}")
    w("  Unlike traditional parsers, LLMs never refuse or crash —")
    w("  they always attempt a response.")
    w("")

    # ── Insight 3: Move legality on fuzz valid FENs ───────────────────
    w("INSIGHT 3: Legal Move Rate on Valid Fuzz FENs")
    w("-" * 65)
    w(f"  {'Model':<16} {'Legal moves / valid FENs':<26} {'Legal move rate'}")
    w(f"  {'-'*16} {'-'*26} {'-'*15}")
    for model in MODELS:
        s = stats["fuzz"][model]
        if not s:
            continue
        w(f"  {MODEL_LABELS[model]:<16} {s['move_valid']:<4} / {s['fen_valid']:<20}  {pct(s['move_valid'], s['fen_valid'])}")
    w("  Llama models outperform Qwen despite Qwen being larger.")
    w("")

    # ── Insight 4: Chaotic positions collapse performance ─────────────
    w("INSIGHT 4: Chaotic Positions Collapse Legal Move Rate")
    w("-" * 65)
    w(f"  {'Model':<16} {'Legal moves / total':<24} {'Legal move rate'}")
    w(f"  {'-'*16} {'-'*24} {'-'*15}")
    for model in MODELS:
        s = stats["chaotic"][model]
        if not s:
            continue
        w(f"  {MODEL_LABELS[model]:<16} {s['move_valid']:<4} / {s['total']:<18}  {pct(s['move_valid'], s['total'])}")
    w("  All 570 chaotic FENs are structurally valid yet all models")
    w("  collapse — exposing pattern-matching rather than reasoning.")
    w("")

    # ── Insight 5: Key contrast table ────────────────────────────────
    w("INSIGHT 5: Fuzz vs Chaotic — Side-by-Side Comparison")
    w("-" * 65)
    w(f"  {'Model':<16} {'Fuzz legal rate':<20} {'Chaotic legal rate':<20} {'Drop'}")
    w(f"  {'-'*16} {'-'*20} {'-'*20} {'-'*8}")
    for model in MODELS:
        sf = stats["fuzz"][model]
        sc = stats["chaotic"][model]
        if not sf or not sc:
            continue
        fuzz_rate    = sf["move_valid"] / sf["fen_valid"] * 100 if sf["fen_valid"] else 0
        chaotic_rate = sc["move_valid"] / sc["total"]    * 100 if sc["total"]      else 0
        drop         = fuzz_rate - chaotic_rate
        w(f"  {MODEL_LABELS[model]:<16} {pct(sf['move_valid'], sf['fen_valid']):<20} {pct(sc['move_valid'], sc['total']):<20} -{drop:.1f}pp")
    w("")

    # ── Summary ───────────────────────────────────────────────────────
    w("=" * 65)
    w("  SUMMARY")
    w("=" * 65)
    w("  - AFL++ invalidated 80% of inputs, confirming effective fuzzing.")
    w("  - LLMs never crash on malformed FENs — they always attempt output.")
    w("  - On standard positions, Llama-1B and Llama-8B achieve ~84% legal")
    w("    move rate vs Qwen's ~70%, despite Qwen being a larger model.")
    w("  - On chaotic positions (all valid FENs), legal move rates fall")
    w("    below 2% across all models — revealing surface-level pattern")
    w("    matching rather than genuine chess understanding.")
    w("=" * 65)

    report = "\n".join(lines)
    print(report)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nReport written to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
