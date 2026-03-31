"""
Prompt templates for The Mind LLM player.

The LLM receives its private observation (hand, pile state) and must output
either PLAY or WAIT. We keep prompts minimal so even a 0.5B model can handle them.

For GRPO we generate a batch of completions per observation and score them.
"""

SYSTEM_PROMPT = """You are playing The Mind, a cooperative card game.

Rules:
- Cards are ranked 1-52 (Clubs 1-13, Diamonds 14-26, Hearts 27-39, Spades 40-52).
- All players must collectively play ALL cards in ascending order.
- No communication is allowed.
- On your turn, you see only YOUR hand and what's been played so far.
- Decide: PLAY your lowest card now, or WAIT.
- Play too early (someone else has a lower card) → round fails.
- Never play → round stalls.

Your response must be exactly one word: PLAY or WAIT."""


def make_user_prompt(obs: dict) -> str:
    """Build the per-step user prompt from a player observation."""
    hand_str = ", ".join(obs["hand_readable"])
    pile_str = obs["pile_top_readable"] if obs["pile_top"] else "nothing yet"

    return (
        f"Round {obs['round_num']} | {obs['num_players']} players\n"
        f"Cards on pile: {obs['cards_on_pile']}/{obs['total_cards']} (top: {pile_str})\n"
        f"Your hand: {hand_str}\n"
        f"Your lowest card: {obs['hand_readable'][0] if obs['hand'] else 'none'}\n\n"
        f"Decision (PLAY or WAIT):"
    )


def parse_action(response: str) -> bool:
    """Parse LLM response to a boolean action. Returns True=PLAY, False=WAIT."""
    text = response.strip().upper()
    # Accept first word containing PLAY or WAIT
    for word in text.split():
        if "PLAY" in word:
            return True
        if "WAIT" in word:
            return False
    # Default to WAIT if unparseable (penalized by game stalling)
    return False
