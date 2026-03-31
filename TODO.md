# TODO

## Phase 1 — Local Development (M1 Pro)

Goal: get smoke tests running locally before committing to a full training run.

### 1. Upgrade model to Qwen3.5-0.8B
- [ ] Update `DEFAULT_MODEL` in `rl/train.py` and `eval/evaluate.py` from `Qwen/Qwen2.5-0.5B-Instruct` → `Qwen/Qwen3.5-0.8B`
- [ ] Update `configs/train_config.yaml` model field
- [ ] Verify chat template still works (same ChatML format — should be compatible)
- [ ] Note: disable thinking mode — 0.8B is prone to thinking loops; keep `enable_thinking=False` (default)

### 2. Add MPS support for Apple Silicon
- [ ] Update device detection in `rl/train.py` and `eval/evaluate.py`:
  ```python
  if torch.cuda.is_available():
      device, dtype = "cuda", torch.bfloat16
  elif torch.backends.mps.is_available():
      device, dtype = "mps", torch.float16
  else:
      device, dtype = "cpu", torch.float32
  ```
- [ ] Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in env before running (some Metal ops not yet implemented)
- [ ] Use `device_map="mps"` explicitly — `"auto"` can silently fall back to CPU on Apple Silicon
- [ ] Use `torch.float16` on MPS (not `bfloat16` — broken on PyTorch < 2.6 / macOS < 14)

### 3. Smoke test locally
- [ ] Test game logic with no model (fast, no deps):
  ```python
  from game.the_mind import TheMindGame
  g = TheMindGame(num_players=2)
  obs = g.reset(round_num=1)
  obs, reward, done, info = g.step([True, False])
  ```
- [ ] Run eval baseline (downloads model, runs 5 games):
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 python -m eval.evaluate \
    --model Qwen/Qwen3.5-0.8B --players 2 --rounds 1 --games 5
  ```
- [ ] Run 3-iteration training smoke test (verify gradient flow doesn't crash):
  ```bash
  PYTORCH_ENABLE_MPS_FALLBACK=1 python -m rl.train \
    --num-players 2 --num-iterations 3 --group-size 2 --save-every 999 \
    --output-dir /tmp/test_ckpt
  ```

---

## Phase 2 — Full Training on Modal

Goal: run the full 500-iteration GRPO training run on a cloud GPU.

### Setup
- [ ] Sign up at modal.com (free tier = $30/month compute credit, no card needed)
- [ ] `pip install modal && modal setup`
- [ ] Create `modal_train.py` wrapping `rl/train.py` as a Modal function

### Modal script outline
```python
import modal

app = modal.App("the-mind-grpo")
vol = modal.Volume.from_name("the-mind-checkpoints", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "accelerate", "sentencepiece", "protobuf")
)

@app.function(
    gpu="A10G",           # 24GB VRAM, ~$1.10/hr — sweet spot for 0.8B
    image=image,
    volumes={"/checkpoints": vol},
    timeout=3600 * 6,     # 6hr ceiling — must set explicitly or job dies at 5min
)
def train():
    import sys
    sys.argv = [
        "train",
        "--model", "Qwen/Qwen3.5-0.8B",
        "--num-players", "8",
        "--num-iterations", "500",
        "--group-size", "8",
        "--output-dir", "/checkpoints",
    ]
    from rl.train import parse_args, train
    train(parse_args())

@app.local_entrypoint()
def main():
    train.remote()
```

### Cost estimate (A10G @ $1.10/hr)
| Scenario | Wall time | Cost |
|---|---|---|
| Fast (30s/iter) | ~4 hrs | ~$5 |
| Middle (75s/iter) | ~10 hrs | ~$11 |
| Slow (120s/iter) | ~17 hrs | ~$18 |

All scenarios fit within the free $30/month credit.

### Checklist before running on Modal
- [ ] Smoke test passes locally (Phase 1 complete)
- [ ] Modal script created and tested with `--num-iterations 5` first
- [ ] `modal.Volume` in place so checkpoints survive if container dies
- [ ] `timeout` set to at least 6 hours
- [ ] Run full 500 iterations: `modal run modal_train.py`
- [ ] Download final checkpoint: `modal volume get the-mind-checkpoints /checkpoints/final ./checkpoints/final`
