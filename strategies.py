"""
strategies.py
-------------
Three fixed bidding strategies for use in auction simulations.

Each strategy accepts a valuation (the bidder's true private value)
and returns a bid amount that never exceeds that valuation.
"""

import random


def truthful_strategy(valuation: float) -> float:
    """
    Truthful (Dominant) Strategy:
    - The bidder bids exactly their true private valuation.
    - Theoretically optimal in second-price auctions (Vickrey).

    Args:
        valuation (float): The bidder's true private value for the item.

    Returns:
        float: A bid equal to the valuation.

    Raises:
        ValueError: If valuation is negative.
    """
    if valuation < 0:
        raise ValueError(f"Valuation must be non-negative, got {valuation}.")

    return valuation


def shaded_strategy(valuation: float, shade_factor: float = 0.8) -> float:
    """
    Bid Shading Strategy:
    - The bidder bids a fixed fraction below their true valuation.
    - Common in first-price auctions to balance winning probability
      against profit margin.
    - Default shade_factor of 0.8 means bidding 80% of true value.

    Args:
        valuation   (float): The bidder's true private value for the item.
        shade_factor (float): Fraction of valuation to bid (0 < shade_factor <= 1).
                              Defaults to 0.8.

    Returns:
        float: A shaded bid strictly <= valuation.

    Raises:
        ValueError: If valuation is negative or shade_factor is out of range.
    """
    if valuation < 0:
        raise ValueError(f"Valuation must be non-negative, got {valuation}.")
    if not (0 < shade_factor <= 1):
        raise ValueError(f"shade_factor must be in (0, 1], got {shade_factor}.")

    return round(valuation * shade_factor, 4)


def random_strategy(valuation: float) -> float:
    """
    Random Strategy:
    - The bidder bids a uniformly random amount between 0 and their valuation.
    - Serves as a noisy baseline; performance is highly variable.

    Args:
        valuation (float): The bidder's true private value for the item.

    Returns:
        float: A random bid in the range [0, valuation].

    Raises:
        ValueError: If valuation is negative.
    """
    if valuation < 0:
        raise ValueError(f"Valuation must be non-negative, got {valuation}.")

    return round(random.uniform(0, valuation), 4)
