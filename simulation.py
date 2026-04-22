"""
simulation.py
-------------
Orchestrates the full auction simulation across N rounds.

Participants:
  - One AIBidder (Q-learning)
  - Three fixed-strategy bidders: truthful, shaded, random

Each round:
  1. Draw private valuations from Uniform(10, 100)
  2. Each bidder produces a bid via their strategy
  3. Auction engine determines winner and price paid
  4. AI bidder receives reward signal and updates Q-table
  5. Metrics are collected into a row

Returns a Pandas DataFrame — one row per round.
"""

import random
import pandas as pd

from auction_engine import first_price_auction, second_price_auction
from strategies import truthful_strategy, shaded_strategy, random_strategy
from ai_bidder import AIBidder


# ---------------------------------------------------------------------------
# Bidder registry
# ---------------------------------------------------------------------------

FIXED_BIDDERS = {
    "truthful": truthful_strategy,
    "shaded":   shaded_strategy,
    "random":   random_strategy,
}


# ---------------------------------------------------------------------------
# Core simulation function
# ---------------------------------------------------------------------------

def run_simulation(
    n_rounds: int = 200,
    auction_type: str = "second",          # "first" | "second"
    shade_factor: float = 0.8,             # used by shaded_strategy
    val_low: float = 10.0,                 # lower bound of valuation distribution
    val_high: float = 100.0,               # upper bound of valuation distribution
    ai_n_levels: int = 11,                 # discrete bid levels for AI bidder
    ai_alpha: float = 0.1,                 # Q-learning rate
    ai_gamma: float = 0.95,                # Q-learning discount factor
    ai_epsilon: float = 1.0,               # initial exploration rate
    ai_epsilon_decay: float = 0.995,       # epsilon decay per round
    ai_epsilon_min: float = 0.05,          # minimum exploration floor
    seed: int | None = None,               # optional RNG seed for reproducibility
) -> pd.DataFrame:
    """
    Run an auction simulation and return per-round metrics as a DataFrame.

    Args:
        n_rounds        (int):   Number of auction rounds.
        auction_type    (str):   "first" for first-price, "second" for second-price.
        shade_factor    (float): Bid fraction used by the shaded strategy.
        val_low         (float): Lower bound for Uniform valuation draws.
        val_high        (float): Upper bound for Uniform valuation draws.
        ai_n_levels     (int):   Number of discrete bid fractions for the AI bidder.
        ai_alpha        (float): AI Q-learning rate.
        ai_gamma        (float): AI discount factor.
        ai_epsilon      (float): AI initial epsilon.
        ai_epsilon_decay(float): AI epsilon decay rate.
        ai_epsilon_min  (float): AI epsilon minimum.
        seed            (int|None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: One row per round with the following columns:
            round           - Round index (1-indexed)
            auction_type    - "first" or "second"
            winner_id       - ID of the winning bidder
            price_paid      - Price the winner paid
            winner_value    - Winner's true private valuation
            highest_value   - Highest valuation among all bidders
            efficiency      - winner_value / highest_value  (allocative efficiency)
            ai_bid          - The AI bidder's bid this round
            ai_valuation    - The AI bidder's private valuation this round
            ai_bid_ratio    - ai_bid / ai_valuation
            ai_reward       - Reward received by the AI bidder this round
            ai_epsilon      - Epsilon at time of bidding (exploration rate)
            ai_best_fraction- AI's current best Q-table fraction (post-update)
            <bidder>_bid    - Bid placed by each individual bidder
            <bidder>_value  - Valuation drawn by each individual bidder
    """
    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if auction_type not in ("first", "second"):
        raise ValueError(f"auction_type must be 'first' or 'second', got {auction_type!r}.")
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}.")
    if val_low >= val_high:
        raise ValueError(f"val_low must be < val_high, got {val_low} >= {val_high}.")

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    if seed is not None:
        random.seed(seed)

    # ------------------------------------------------------------------
    # Instantiate AI bidder (fresh state each simulation run)
    # ------------------------------------------------------------------
    ai = AIBidder(
        bidder_id="ai_bidder",
        n_levels=ai_n_levels,
        alpha=ai_alpha,
        gamma=ai_gamma,
        epsilon=ai_epsilon,
        epsilon_decay=ai_epsilon_decay,
        epsilon_min=ai_epsilon_min,
    )

    # Select auction engine
    auction_fn = first_price_auction if auction_type == "first" else second_price_auction

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    records = []

    for round_num in range(1, n_rounds + 1):

        # 1. Draw private valuations
        valuations = {
            bidder_id: random.uniform(val_low, val_high)
            for bidder_id in FIXED_BIDDERS
        }
        valuations["ai_bidder"] = random.uniform(val_low, val_high)

        # 2. Collect bids
        bids = {}
        for bidder_id, strategy_fn in FIXED_BIDDERS.items():
            val = valuations[bidder_id]
            if bidder_id == "shaded":
                bids[bidder_id] = strategy_fn(val, shade_factor)
            else:
                bids[bidder_id] = strategy_fn(val)

        ai_val = valuations["ai_bidder"]
        ai_epsilon_snapshot = ai.epsilon          # capture before get_bid decays it
        ai_bid = ai.get_bid(ai_val)
        bids["ai_bidder"] = ai_bid

        # 3. Run auction
        winner_id, price_paid = auction_fn(bids)

        # 4. Update AI bidder
        ai_won = (winner_id == "ai_bidder")
        ai_reward = ai.update(
            valuation=ai_val,
            won=ai_won,
            price_paid=price_paid if ai_won else 0.0,
        )

        # 5. Compute metrics
        winner_value   = valuations[winner_id]
        highest_value  = max(valuations.values())
        efficiency     = winner_value / highest_value if highest_value > 0 else 0.0
        ai_bid_ratio   = ai_bid / ai_val if ai_val > 0 else 0.0

        # 6. Build record row
        row = {
            "round":            round_num,
            "auction_type":     auction_type,
            "winner_id":        winner_id,
            "price_paid":       round(price_paid, 4),
            "winner_value":     round(winner_value, 4),
            "highest_value":    round(highest_value, 4),
            "efficiency":       round(efficiency, 4),
            "ai_bid":           round(ai_bid, 4),
            "ai_valuation":     round(ai_val, 4),
            "ai_bid_ratio":     round(ai_bid_ratio, 4),
            "ai_reward":        round(ai_reward, 4),
            "ai_epsilon":       round(ai_epsilon_snapshot, 4),
            "ai_best_fraction": round(ai.best_fraction, 4),
        }

        # Per-bidder bid and valuation columns
        for bidder_id in list(FIXED_BIDDERS.keys()) + ["ai_bidder"]:
            row[f"{bidder_id}_bid"]   = round(bids[bidder_id], 4)
            row[f"{bidder_id}_value"] = round(valuations[bidder_id], 4)

        records.append(row)

    return pd.DataFrame(records)
