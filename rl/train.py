"""
GRPO training for The Mind LLM player.

We use Group Relative Policy Optimization (GRPO) — the same algorithm used in
DeepSeek-R1 — because it:
  1. Requires no separate value network (memory-efficient for small GPUs)
  2. Works well for sparse / binary rewards (win/lose)
  3. Is simple to implement on top of standard HuggingFace transformers

Model: Qwen/Qwen2.5-0.5B-Instruct  (smallest capable chat model)
       Can swap for Qwen2.5-1.5B-Instruct for better reasoning.

Training loop:
  For each outer iteration:
    1. Run G rollouts ("group") with current policy π_θ_old (frozen copy)
    2. Compute per-token advantages via group-relative normalization
    3. Update π_θ with clipped surrogate + KL penalty (PPO-style)
    4. Curriculum: start with round=1, increase round difficulty as win rate rises
"""

import argparse
import os
import random
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.env import batch_rollout, Trajectory


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

GRPO_DEFAULTS = dict(
    group_size=8,           # G: rollouts per update step
    num_iterations=500,     # outer training steps
    lr=1e-5,
    kl_coef=0.05,           # β in GRPO KL penalty
    clip_eps=0.2,           # ε in surrogate clipping
    temperature=1.0,        # sampling temperature for rollouts
    max_grad_norm=1.0,
    num_players=8,
    start_round=1,
    max_round=6,   # 8 players × 6 cards = 48 ≤ 52; round 7 would need 56
    win_rate_threshold=0.7, # promote to next round when win rate > this
    eval_every=25,
    save_every=100,
    output_dir="checkpoints",
)


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

def compute_returns(
    rewards: list[float], decision_steps: list[int], gamma: float = 0.99
) -> list[float]:
    """
    Discounted return per decision. Entries sharing the same game step get the same
    return-to-go so simultaneous players are not gamma-discounted against each other.
    """
    if not rewards:
        return []
    step_to_r: dict[int, float] = {}
    for r, sid in zip(rewards, decision_steps):
        step_to_r[sid] = r
    ordered = sorted(step_to_r.keys())
    G = 0.0
    ret_by_step: dict[int, float] = {}
    for sid in reversed(ordered):
        r = step_to_r[sid]
        G = r + gamma * G
        ret_by_step[sid] = G
    return [ret_by_step[sid] for sid in decision_steps]


def normalize(values: list[float]) -> list[float]:
    """Zero-mean, unit-variance normalization (group-relative)."""
    if len(values) < 2:
        return values
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = max(var ** 0.5, 1e-8)
    return [(v - mean) / std for v in values]


# ---------------------------------------------------------------------------
# GRPO core
# ---------------------------------------------------------------------------

def compute_policy_loss(
    model: torch.nn.Module,
    tokenizer,
    trajectories: list[Trajectory],
    old_model: torch.nn.Module,
    kl_coef: float,
    clip_eps: float,
) -> torch.Tensor:
    """
    Compute the GRPO objective across a group of trajectories.

    For each (prompt, completion, advantage) triple:
      - Get log-probs under current policy and old policy
      - Compute importance ratio r = exp(log π_θ - log π_old)
      - Clipped surrogate: min(r*A, clip(r, 1-ε, 1+ε)*A)
      - KL penalty: β * KL(π_old || π_θ)
    """
    # Collect all trajectory returns and normalize across the group
    all_returns = []
    for traj in trajectories:
        returns = compute_returns(traj.rewards, traj.decision_steps)
        all_returns.extend(returns)
    advantages = normalize(all_returns)

    total_loss = torch.tensor(0.0, requires_grad=True, device=next(model.parameters()).device)
    count = 0

    adv_idx = 0
    for traj in trajectories:
        traj_returns = compute_returns(traj.rewards, traj.decision_steps)
        for prompt, completion, _ in zip(traj.prompts, traj.completions, traj_returns):
            adv = advantages[adv_idx]
            adv_idx += 1

            full_text = prompt + completion
            enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(model.device) for k, v in enc.items()}

            prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

            with torch.no_grad():
                old_logits = old_model(**enc).logits
            new_logits = model(**enc).logits

            # Shift for causal LM: predict token i from tokens 0..i-1
            labels = enc["input_ids"][0, 1:]
            new_log_probs = F.log_softmax(new_logits[0, :-1], dim=-1)
            old_log_probs = F.log_softmax(old_logits[0, :-1], dim=-1)

            # Only compute loss on completion tokens
            comp_new = new_log_probs[prompt_len - 1:]
            comp_old = old_log_probs[prompt_len - 1:]
            comp_labels = labels[prompt_len - 1:]

            if comp_labels.shape[0] == 0:
                continue

            token_new = comp_new.gather(1, comp_labels.unsqueeze(1)).squeeze(1)
            token_old = comp_old.gather(1, comp_labels.unsqueeze(1)).squeeze(1)

            log_ratio = token_new - token_old
            ratio = log_ratio.exp()

            adv_t = torch.tensor(adv, device=model.device)
            surr1 = ratio * adv_t
            surr2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv_t
            surrogate = -torch.min(surr1, surr2).mean()

            # KL(π_old || π_new) at each position: Σ_a π_old(a) (log π_old(a) − log π_new(a))
            kl = (comp_old.exp() * (comp_old - comp_new)).sum(dim=-1).mean()

            step_loss = surrogate + kl_coef * kl
            total_loss = total_loss + step_loss
            count += 1

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class Curriculum:
    def __init__(self, start_round: int, max_round: int, threshold: float):
        self.round = start_round
        self.max_round = max_round
        self.threshold = threshold
        self.win_history: list[bool] = []

    def update(self, win: bool):
        self.win_history.append(win)
        if len(self.win_history) >= 20:
            win_rate = sum(self.win_history[-20:]) / 20
            if win_rate >= self.threshold and self.round < self.max_round:
                self.round += 1
                print(f"  [Curriculum] Win rate {win_rate:.2f} → advancing to round {self.round}")
                self.win_history = []

    @property
    def current_round(self):
        return self.round


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    curriculum = Curriculum(args.start_round, args.max_round, args.win_rate_threshold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting GRPO training for {args.num_iterations} iterations")
    print(f"Group size: {args.group_size} | Players: {args.num_players}\n")

    for iteration in range(1, args.num_iterations + 1):
        current_round = curriculum.current_round

        # --- Freeze old policy (reference) ---
        old_model = deepcopy(model)
        old_model.eval()

        # --- Collect rollouts ---
        model.eval()
        with torch.no_grad():
            trajectories = batch_rollout(
                model=model,
                tokenizer=tokenizer,
                batch_size=args.group_size,
                num_players=args.num_players,
                round_num=current_round,
                temperature=args.temperature,
            )
        model.train()

        wins = sum(t.success for t in trajectories)
        win_rate = wins / len(trajectories)
        avg_len = sum(len(t) for t in trajectories) / len(trajectories)

        for t in trajectories:
            curriculum.update(t.success)

        # --- GRPO update ---
        optimizer.zero_grad()
        loss = compute_policy_loss(
            model=model,
            tokenizer=tokenizer,
            trajectories=trajectories,
            old_model=old_model,
            kl_coef=args.kl_coef,
            clip_eps=args.clip_eps,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if iteration % 5 == 0:
            print(
                f"Iter {iteration:4d} | Round {current_round} | "
                f"Win {wins}/{args.group_size} ({win_rate:.0%}) | "
                f"Loss {loss.item():.4f} | AvgLen {avg_len:.1f}"
            )

        if args.eval_every > 0 and iteration % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_trajs = batch_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    batch_size=args.group_size,
                    num_players=args.num_players,
                    round_num=current_round,
                    temperature=0.0,
                )
            eval_wins = sum(t.success for t in eval_trajs)
            print(
                f"  [Eval] greedy {eval_wins}/{len(eval_trajs)} wins "
                f"({eval_wins / len(eval_trajs):.0%})"
            )
            model.train()

        if iteration % args.save_every == 0:
            ckpt_path = output_dir / f"iter_{iteration:04d}"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Saved checkpoint → {ckpt_path}")

    final_path = output_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train LLM to play The Mind via GRPO")
    p.add_argument("--model", default=DEFAULT_MODEL)
    for k, v in GRPO_DEFAULTS.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
