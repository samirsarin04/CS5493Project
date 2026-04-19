"""
LLM chess move inference on fuzzed FEN board states.

Usage:
    python llm_inference.py --model qwen25 --output results.csv
    python llm_inference.py --model llama3  --limit 100

Supported --model values are the keys in MODEL_REGISTRY below.
"""

import argparse
import csv
import os
import re
import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model registry ────────────────────────────────────────────────────────────
# Keys become the valid --model flag values.
MODEL_REGISTRY = {
    "llama1b":  "meta-llama/Llama-3.2-1B",
    "llama8b":  "meta-llama/Meta-Llama-3-8B",
    "qwen25":   "Qwen/Qwen2.5-7B-Instruct",
}

INPUT_CSV  = "data/fuzz_fens.csv"
FIELDNAMES = [
    "run",
    "fen",
    "model",
    "llm_bestmove",
    "fen_valid",
    "llm_move_valid",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(fen: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a chess engine. You will be given a chess position in FEN notation. "
                "Your job is to respond with the single best move in UCI format (e.g. e2e4, g1f3, e1g1 for castling). "
                "Rules:"
                "- Respond with ONLY the move in UCI format "
                "- No explanation, no punctuation, no extra text "
                "- The move must be legal in the given position"
            ),
        },
        {"role": "user", "content": f"FEN: {fen}"},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except ValueError:
        # Base models have no chat template — use a plain completion prompt
        return (
            "You are a chess engine. Given a chess position in FEN notation, "
            "respond with only the best move in UCI format (e.g. e2e4). No explanation.\n\n"
            f"FEN: {fen}\nMove:"
        )


UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")

def parse_uci_move(text: str) -> str:
    m = UCI_RE.search(text.strip().lower())
    return m.group(1) if m else ""


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_id: str):
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"  HF_HOME={hf_home}", flush=True)

    print(f"Loading tokenizer: {model_id} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id} …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}", flush=True)
    return model, tokenizer


def generate_move(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    stop_ids = [
        tokenizer.eos_token_id,
        tokenizer.encode("\n", add_special_tokens=False)[0],
    ]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=stop_ids,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LLM chess move inference on fuzzed FENs")
    p.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help=f"Model to run. Choices: {', '.join(MODEL_REGISTRY.keys())}",
    )
    p.add_argument(
        "--input", default=INPUT_CSV,
        help=f"Input CSV with 'run' and 'fen' columns (default: {INPUT_CSV}).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output CSV path. Defaults to llm_results_<model>.csv in scratch if $SCRATCH is set, else current dir.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N positions (useful for quick tests).",
    )
    return p.parse_args()


def default_output_path(model_key: str, input_path: str) -> str:
    scratch = os.environ.get("SCRATCH") or os.environ.get("SLURM_SUBMIT_DIR", ".")
    stem = os.path.splitext(os.path.basename(input_path))[0]  # e.g. "fuzz_fens" or "chaotic_fens"
    # Strip trailing "_fens" for a cleaner output name, e.g. "fuzz" -> "llm_fuzz_results_qwen25.csv"
    prefix = stem[:-5] if stem.endswith("_fens") else stem
    return os.path.join(scratch, f"llm_{prefix}_results_{model_key}.csv")


def main():
    args = parse_args()
    model_id  = MODEL_REGISTRY[args.model]
    out_path  = args.output or default_output_path(args.model, args.input)

    print(f"Model key : {args.model}", flush=True)
    print(f"Model ID  : {model_id}", flush=True)
    print(f"Input CSV : {args.input}", flush=True)
    print(f"Output CSV: {out_path}", flush=True)

    model, tokenizer = load_model(model_id)

    with open(args.input, newline="", encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))
    if args.limit:
        dataset = dataset[: args.limit]

    print(f"Positions to process: {len(dataset)}", flush=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, row in enumerate(dataset, start=1):
            fen = row["fen"].strip()
            run = row.get("run", "")

            prompt   = build_prompt(fen, tokenizer)
            new_text = generate_move(model, tokenizer, prompt)
            llm_move = parse_uci_move(new_text)

            # Check FEN validity after getting LLM output
            fen_valid  = False
            move_valid = False
            try:
                board = chess.Board(fen)
                fen_valid = True
                if llm_move:
                    move_valid = chess.Move.from_uci(llm_move) in board.legal_moves
            except Exception:
                pass

            writer.writerow({
                "run":            run,
                "fen":            fen,
                "model":          args.model,
                "llm_bestmove":   llm_move,
                "fen_valid":      fen_valid,
                "llm_move_valid": move_valid,
            })

            if i % 100 == 0:
                print(f"Processed {i}/{len(dataset)}", flush=True)

    print(f"Done. Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
