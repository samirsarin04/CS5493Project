"""
LLM chess move inference using meta-llama/Meta-Llama-3-8B on CHPC cluster.

Sends the same FEN board states used in the Stockfish run to the LLM and
records the predicted best move. No Stockfish required — scoring is done
locally after transferring the output CSV.
"""

import argparse
import csv
import os
import re
import chess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
LOCAL_DATASET = "data/chess_fens.csv"
MODEL_ID      = "meta-llama/Meta-Llama-3-8B"
OUTPUT_CSV    = "llm_inference_results.csv"
FIELDNAMES   = [
    "fen",
    "dataset_next_move",
    "side_to_move",
    "llm_raw_output",   # full text the model generated
    "llm_bestmove",     # parsed UCI move (empty if unparseable)
    "llm_move_valid",   # True/False — is it a legal move in this position?
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(fen: str) -> str:
    return (
        "You are a world-class chess engine. "
        "Given a chess position in FEN notation, output ONLY the best move "
        "in UCI notation (e.g. e2e4, g1f3, e1g1). "
        "Do not include explanations, punctuation, or any other text.\n\n"
        f"FEN: {fen}\n"
        "Best move:"
    )


UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")

def parse_uci_move(text: str) -> str:
    """Extract the first UCI-looking token from model output."""
    m = UCI_RE.search(text.strip().lower())
    return m.group(1) if m else ""


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"  HF_HOME={hf_home}", flush=True)

    print(f"Loading tokenizer: {MODEL_ID} …", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {MODEL_ID} …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    print(f"  Model loaded on {next(model.parameters()).device}", flush=True)
    return model, tokenizer


def generate_move(model, tokenizer, prompt: str) -> str:
    """Run one greedy completion and return only the newly generated text."""
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
    p = argparse.ArgumentParser(description="LLM chess move inference")
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N positions (useful for quick tests)."
    )
    p.add_argument(
        "--output", default=OUTPUT_CSV,
        help=f"Output CSV path (default: {OUTPUT_CSV})."
    )
    return p.parse_args()


def main():
    args = parse_args()

    model, tokenizer = load_model()

    with open(LOCAL_DATASET, newline="", encoding="utf-8") as f:
        dataset = list(csv.DictReader(f))
    if args.limit:
        dataset = dataset[: args.limit]

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, row in enumerate(dataset, start=1):
            fen               = row["FEN"]
            dataset_next_move = row.get("Next move", "")

            board = chess.Board(fen)
            side  = board.turn

            prompt   = build_prompt(fen)
            new_text = generate_move(model, tokenizer, prompt)
            llm_move = parse_uci_move(new_text)

            move_obj   = None
            move_valid = False
            if llm_move:
                try:
                    move_obj   = chess.Move.from_uci(llm_move)
                    move_valid = move_obj in board.legal_moves
                except ValueError:
                    pass

            writer.writerow({
                "fen":               fen,
                "dataset_next_move": dataset_next_move,
                "side_to_move":      "white" if side else "black",
                "llm_raw_output":    new_text,
                "llm_bestmove":      llm_move,
                "llm_move_valid":    move_valid,
            })

            if i % 100 == 0:
                print(f"Processed {i} positions", flush=True)

    print(f"Done. Results written to {args.output}")


if __name__ == "__main__":
    main()
