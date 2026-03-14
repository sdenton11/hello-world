# The Mind — RL with LLMs

Teaching a small LLM to play **The Mind** (standard 52-card deck version) using
**Group Relative Policy Optimization (GRPO)**.

---

## The Game

Players must collectively play all cards in ascending order (1–52) with:
- **No communication** of any kind
- Cards ranked: Clubs 1–13, Diamonds 14–26, Hearts 27–39, Spades 40–52
- Round N: each player receives N cards

Playing a card when someone else holds a lower unplayed card → round fails.

---

## Why GRPO?

| Property | Why it fits |
|---|---|
| No value network needed | Saves memory; feasible on a single consumer GPU |
| Binary / sparse rewards | GRPO's group-relative normalization handles this gracefully |
| Same algorithm as DeepSeek-R1 | Well-validated for LLM policy learning |

---

## Model

Default: **`Qwen/Qwen2.5-0.5B-Instruct`** (~500M params, fits in ~6–8 GB VRAM)

Upgrade to `Qwen/Qwen2.5-1.5B-Instruct` for stronger reasoning (needs ~16 GB).

---

## Architecture

```
the-mind-rl/
├── game/
│   └── the_mind.py        # Turn-based game simulator
├── rl/
│   ├── prompts.py         # System + user prompt templates, action parser
│   ├── env.py             # rollout() and batch_rollout() — LLM plays the game
│   └── train.py           # GRPO training loop + curriculum
├── eval/
│   └── evaluate.py        # Win-rate evaluation script
└── configs/
    └── train_config.yaml  # Hyperparameters
```

### Decision loop (per timestep)

```
For each player (with cards remaining):
  prompt = system_prompt + hand + pile_state
  completion = model.generate(prompt, max_new_tokens=4)
  action = parse("PLAY" | "WAIT")

game.step(actions) -> reward, done
```

All players share the **same model weights** (parameter-tied multi-agent).
This is valid because all players are structurally identical.

---

## Reward structure

| Event | Reward |
|---|---|
| Round success | +1.0 |
| Out-of-order play (failure) | -1.0 - (remaining cards / total cards) |
| Card(s) played legally | +0.05 per card |
| All players waited | -0.01 (stalling penalty) |

---

## Training

```bash
pip install -r requirements.txt

# Quick start (2 players, rounds 1-5, 500 iterations)
python -m rl.train

# Custom
python -m rl.train \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --num-players 3 \
  --group-size 16 \
  --num-iterations 1000 \
  --output-dir checkpoints
```

### Curriculum

Training starts at **Round 1** (1 card each) and automatically advances to
harder rounds once the rolling 20-game win rate exceeds `--win-rate-threshold`
(default 70%).

---

## Evaluation

```bash
python -m eval.evaluate --model checkpoints/final --rounds 1 2 3 --games 100
```

---

## Key challenges and design notes

### 1. Implicit coordination without communication
The LLM must learn a **timing policy**: how long to wait before playing, based
only on its own cards and the current pile top. This is analogous to learning
a threshold strategy — play when your card is likely the global minimum.

### 2. Simultaneous actions via sequential queries
We query players one-at-a-time but treat all decisions within a timestep as
simultaneous (no player sees another's choice before deciding). This is a
standard approximation in multi-agent RL.

### 3. Why text instead of raw logits?
Text prompts allow us to:
- Leverage the model's pre-trained world knowledge about card games
- Easily inspect and debug agent reasoning
- Transfer to multi-model setups later

---

## Next steps / extensions

- [ ] Add chain-of-thought scratchpad (allow 1-2 sentence reasoning before PLAY/WAIT)
- [ ] Try 3-4 player games
- [ ] Implement lives system (3 mistakes allowed per game across rounds)
- [ ] Self-play with different model checkpoints per player
- [ ] Distill the learned policy into a smaller model
