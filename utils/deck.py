import random
from typing import List


def create_deck() -> List[int]:
    """Creates a shuffled deck of cards numbered 1 to 100."""
    deck = list(range(1, 101))
    random.shuffle(deck)
    return deck


def draw_card(deck: List[int]) -> int:
    """Draws a card from the deck. Raises IndexError if deck is empty."""
    if not deck:
        raise IndexError("Deck is empty")
    return deck.pop()
