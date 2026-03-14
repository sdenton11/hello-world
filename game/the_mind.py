"""
The Mind - Standard 52-card deck version.

Cards are ranked globally 1-52:
  Clubs    1-13  (A=1, 2=2, ..., K=13)
  Diamonds 14-26
  Hearts   27-39
  Spades   40-52

Players must collectively play ALL cards in ascending order with no communication.
Each round N, each player receives N cards.
At each timestep, every player simultaneously decides: PLAY (their lowest card) or WAIT.

If a player plays a card while another player holds a lower unplayed card → round fails.
Round succeeds when all cards are played in order.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


SUITS = ["Clubs", "Diamonds", "Hearts", "Spades"]
SUIT_OFFSETS = {"Clubs": 0, "Diamonds": 13, "Hearts": 26, "Spades": 39}
RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]


def card_to_global(suit: str, rank_idx: int) -> int:
    """Convert suit + rank index (0-based) to global value 1-52."""
    return SUIT_OFFSETS[suit] + rank_idx + 1


def global_to_card(val: int) -> str:
    """Convert global value 1-52 to human-readable card string."""
    val -= 1
    suit = SUITS[val // 13]
    rank = RANKS[val % 13]
    return f"{rank} of {suit}"


def build_deck() -> list[int]:
    """Return a shuffled list of global card values 1-52."""
    deck = list(range(1, 53))
    random.shuffle(deck)
    return deck


class RoundResult(Enum):
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass
class PlayerState:
    player_id: int
    hand: list[int] = field(default_factory=list)  # sorted ascending

    def lowest_card(self) -> Optional[int]:
        return self.hand[0] if self.hand else None

    def play_lowest(self) -> int:
        return self.hand.pop(0)


@dataclass
class TheMindGame:
    """
    Turn-based simulation of The Mind for RL.

    Each step: receive a list of booleans (one per player) indicating PLAY or WAIT.
    Returns observation, reward, done, info.
    """

    num_players: int = 2
    max_rounds: int = 5  # train on early rounds first

    # Runtime state
    round_num: int = field(init=False, default=0)
    players: list[PlayerState] = field(init=False, default_factory=list)
    play_pile: list[int] = field(init=False, default_factory=list)
    lives: int = field(init=False, default=3)
    result: RoundResult = field(init=False, default=RoundResult.IN_PROGRESS)

    def reset(self, round_num: int = 1) -> dict:
        """Start a new round. Returns initial observations per player."""
        self.round_num = round_num
        self.play_pile = []
        self.result = RoundResult.IN_PROGRESS

        deck = build_deck()
        self.players = []
        for pid in range(self.num_players):
            hand = sorted(deck[pid * round_num : (pid + 1) * round_num])
            self.players.append(PlayerState(player_id=pid, hand=hand))

        return self._observations()

    def step(self, actions: list[bool]) -> tuple[dict, float, bool, dict]:
        """
        Apply one timestep of simultaneous actions.

        actions: list of bool, one per player. True = PLAY lowest card, False = WAIT.
        Returns: (observations, reward, done, info)
        """
        assert len(actions) == self.num_players
        assert self.result == RoundResult.IN_PROGRESS

        played_this_step = []
        for pid, play in enumerate(actions):
            if play and self.players[pid].hand:
                card = self.players[pid].play_lowest()
                played_this_step.append((card, pid))

        if not played_this_step:
            # Everyone waited — small negative to discourage infinite stalling
            return self._observations(), -0.01, False, {"event": "all_waited"}

        # Sort by card value so we process in the order played
        played_this_step.sort()

        # The top of the play pile before this step
        pile_top = self.play_pile[-1] if self.play_pile else 0

        # Check for violations: a card lower than pile_top exists in any hand
        # (played out-of-order relative to cards still held)
        for card, pid in played_this_step:
            # Check if any OTHER player (or this player's remaining hand) has a lower card
            for p in self.players:
                if p.hand and p.hand[0] < card:
                    # Someone played too early — failure
                    self.result = RoundResult.FAILURE
                    reward = -1.0 - (self._cards_remaining() / (self.num_players * self.round_num))
                    return self._observations(), reward, True, {
                        "event": "out_of_order",
                        "bad_card": card,
                        "bad_player": pid,
                        "lower_card_holder": p.player_id,
                        "lower_card": p.hand[0],
                    }

        # All played cards are legal — extend the pile
        for card, pid in played_this_step:
            self.play_pile.append(card)

        # Check win condition
        if self._cards_remaining() == 0 and not any(p.hand for p in self.players):
            self.result = RoundResult.SUCCESS
            return self._observations(), 1.0, True, {"event": "round_complete"}

        # Partial progress reward
        progress = len(self.play_pile) / (self.num_players * self.round_num)
        return self._observations(), 0.05 * len(played_this_step), False, {
            "event": "cards_played",
            "cards": [c for c, _ in played_this_step],
            "progress": progress,
        }

    def _observations(self) -> dict:
        """Return per-player observations (each player only sees their own hand + public info)."""
        pile_top = self.play_pile[-1] if self.play_pile else 0
        cards_on_pile = len(self.play_pile)
        total_cards = self.num_players * self.round_num

        obs = {}
        for p in self.players:
            obs[p.player_id] = {
                "player_id": p.player_id,
                "hand": list(p.hand),
                "hand_readable": [global_to_card(c) for c in p.hand],
                "pile_top": pile_top,
                "pile_top_readable": global_to_card(pile_top) if pile_top else "empty",
                "cards_on_pile": cards_on_pile,
                "total_cards": total_cards,
                "cards_remaining": total_cards - cards_on_pile,
                "round_num": self.round_num,
                "num_players": self.num_players,
            }
        return obs

    def _cards_remaining(self) -> int:
        return sum(len(p.hand) for p in self.players)
