"""
Evaluate a trained (or baseline) model on The Mind.

Usage:
  python -m eval.evaluate --model Qwen/Qwen2.5-0.5B-Instruct
  python -m eval.evaluate --model checkpoints/final --rounds 1 2 3 --games 100
"""

import argparse
import json
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.env import rollout


def evaluate(model, tokenizer, num_players: int, rounds: list[int], games_per_round: int):
    results = defaultdict(list)

    for round_num in rounds:
        print(f"  Round {round_num}: ", end="", flush=True)
        for _ in range(games_per_round):
            traj = rollout(
                model=model,
                tokenizer=tokenizer,
                num_players=num_players,
                round_num=round_num,
                temperature=0.0,   # greedy at eval time
            )
            results[round_num].append(traj.success)
            print("✓" if traj.success else "✗", end="", flush=True)
        win_rate = sum(results[round_num]) / games_per_round
        print(f"  win rate: {win_rate:.1%}")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--rounds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--players", type=int, default=2)
    p.add_argument("--games", type=int, default=50, help="Games per round")
    p.add_argument("--output", default=None, help="JSON file to write results to")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    print(f"\nEvaluating {args.players}-player game | {args.games} games per round\n")
    with torch.no_grad():
        results = evaluate(model, tokenizer, args.players, args.rounds, args.games)

    summary = {
        str(r): {
            "win_rate": sum(v) / len(v),
            "wins": sum(v),
            "games": len(v),
        }
        for r, v in results.items()
    }

    print("\nSummary:")
    for round_num, stats in summary.items():
        print(f"  Round {round_num}: {stats['wins']}/{stats['games']} ({stats['win_rate']:.1%})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
