"""
LLM-based multi-agent environment for The Mind.

Each "agent" is one player slot driven by the same shared LLM policy.
We expose a rollout() function that runs a full round and returns
(prompts, completions, rewards) suitable for GRPO training.

Architecture:
  - Shared policy: all players use the same model (parameter-tied agents)
  - Observation: text prompt constructed from player's private hand + public pile
  - Action: PLAY or WAIT token, sampled from model logits
  - Reward: shaped per-step + terminal (see game/the_mind.py)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from game.the_mind import TheMindGame
from rl.prompts import SYSTEM_PROMPT, make_user_prompt, parse_action


MAX_STEPS_PER_ROUND = 200  # safety cap to prevent infinite loops


@dataclass
class Trajectory:
    """Stores one complete round's worth of (prompt, completion, reward) tuples."""
    prompts: list[str]
    completions: list[str]
    rewards: list[float]
    success: bool
    round_num: int
    num_players: int

    def __len__(self):
        return len(self.prompts)


def build_chat_prompt(tokenizer: PreTrainedTokenizer, obs: dict) -> str:
    """Format observation as a chat-template prompt string."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(obs)},
    ]
    # apply_chat_template returns a string ready for tokenization
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def sample_action(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    temperature: float = 1.0,
) -> tuple[str, str]:
    """
    Sample a single PLAY/WAIT token from the model.
    Returns (full_prompt_str, completion_str).
    We constrain generation to 1 token for efficiency,
    then map it back to PLAY/WAIT.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=4,       # short: "PLAY" or "WAIT"
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return prompt, completion


def rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_players: int = 2,
    round_num: int = 1,
    temperature: float = 1.0,
    device: Optional[str] = None,
) -> Trajectory:
    """
    Run one complete round of The Mind with the LLM policy.
    Returns a Trajectory of (prompt, completion, reward) for every decision made.
    """
    game = TheMindGame(num_players=num_players)
    obs_dict = game.reset(round_num=round_num)

    prompts: list[str] = []
    completions: list[str] = []
    step_rewards: list[float] = []

    done = False
    step = 0

    while not done and step < MAX_STEPS_PER_ROUND:
        step += 1
        actions = []
        step_prompts = []
        step_completions = []

        # Each player decides simultaneously (we query them sequentially for simplicity)
        for pid in range(num_players):
            if not obs_dict[pid]["hand"]:
                # Player has no cards left — implicitly WAIT
                actions.append(False)
                continue

            prompt = build_chat_prompt(tokenizer, obs_dict[pid])
            _, completion = sample_action(model, tokenizer, prompt, temperature)
            action = parse_action(completion)

            step_prompts.append(prompt)
            step_completions.append(completion)
            actions.append(action)

        obs_dict, reward, done, info = game.step(actions)

        # Assign the step reward to all players who made a decision this step
        for p, c in zip(step_prompts, step_completions):
            prompts.append(p)
            completions.append(c)
            step_rewards.append(reward)

    success = game.result.value == "success"
    return Trajectory(
        prompts=prompts,
        completions=completions,
        rewards=step_rewards,
        success=success,
        round_num=round_num,
        num_players=num_players,
    )


def batch_rollout(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    num_players: int = 2,
    round_num: int = 1,
    temperature: float = 1.0,
) -> list[Trajectory]:
    """Run multiple rollouts and return a list of trajectories."""
    return [
        rollout(model, tokenizer, num_players, round_num, temperature)
        for _ in range(batch_size)
    ]
