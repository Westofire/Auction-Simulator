"""
simulation.py
-------------
Orchestrates the full auction simulation across N rounds.

Participants:
  - One AIBidder (Q-learning)
  - One BanditBidder (UCB or Thompson Sampling) — optional
  - Three fixed-strategy bidders: truthful, shaded, random

Each round:
  1. Draw private valuations from Uniform(10, 100)
  2. Each bidder produces a bid via their strategy
  3. Auction engine determines winner and price paid
  4. Learning agents receive reward signal and update their models
  5. Metrics are collected into a row

Returns a Pandas DataFrame — one row per round.
"""

import random
import pandas as pd

from auction_engine import first_price_auction, second_price_auction
from strategies import truthful_strategy, shaded_strategy, random_strategy
from ai_bidder import AIBidder
from bandit_bidder import BanditBidder


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
    include_bandit: bool = False,          # whether to include BanditBidder
    bandit_algorithm: str = "ucb",         # "ucb" | "thompson"
    bandit_ucb_c: float = 2.0,             # UCB exploration coefficient
    seed: int | None = None,               # optional RNG seed for reproducibility
) -> pd.DataFrame:
    """
    Run an auction simulation and return per-round metrics as a DataFrame.

    Args:
        n_rounds         (int):   Number of auction rounds.
        auction_type     (str):   "first" for first-price, "second" for second-price.
        shade_factor     (float): Bid fraction used by the shaded strategy.
        val_low          (float): Lower bound for Uniform valuation draws.
        val_high         (float): Upper bound for Uniform valuation draws.
        ai_n_levels      (int):   Number of discrete bid fractions for the AI bidder.
        ai_alpha         (float): AI Q-learning rate.
        ai_gamma         (float): AI discount factor.
        ai_epsilon       (float): AI initial epsilon.
        ai_epsilon_decay (float): AI epsilon decay rate.
        ai_epsilon_min   (float): AI epsilon minimum.
        include_bandit   (bool):  If True, adds a BanditBidder to the simulation.
        bandit_algorithm (str):   "ucb" or "thompson" — bandit strategy to use.
        bandit_ucb_c     (float): Exploration coefficient for UCB algorithm.
        seed             (int|None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: One row per round. Bandit columns are present only when
                      include_bandit=True; all other columns are always present.
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

    # Instantiate BanditBidder if requested (same n_levels for fair comparison)
    bandit = BanditBidder(
        bidder_id="bandit_bidder",
        n_levels=ai_n_levels,
        algorithm=bandit_algorithm,
        ucb_c=bandit_ucb_c,
    ) if include_bandit else None

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
        if bandit is not None:
            valuations["bandit_bidder"] = random.uniform(val_low, val_high)

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

        # Collect bandit bid if active
        if bandit is not None:
            bandit_val = valuations["bandit_bidder"]
            bandit_bid = bandit.get_bid(bandit_val)
            bids["bandit_bidder"] = bandit_bid

        # 3. Run auction
        winner_id, price_paid = auction_fn(bids)

        # 4. Update AI bidder (Q-learning)
        ai_won = (winner_id == "ai_bidder")
        ai_reward = ai.update(
            valuation=ai_val,
            won=ai_won,
            price_paid=price_paid if ai_won else 0.0,
        )

        # Update Bandit bidder if active
        if bandit is not None:
            bandit_won = (winner_id == "bandit_bidder")
            bandit_reward = bandit.update(
                valuation=valuations["bandit_bidder"],
                won=bandit_won,
                price_paid=price_paid if bandit_won else 0.0,
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

        # Bandit metrics — only present when include_bandit=True
        if bandit is not None:
            bandit_val   = valuations["bandit_bidder"]
            bandit_bid   = bids["bandit_bidder"]
            row["bandit_bid"]           = round(bandit_bid, 4)
            row["bandit_valuation"]     = round(bandit_val, 4)
            row["bandit_bid_ratio"]     = round(bandit_bid / bandit_val if bandit_val > 0 else 0.0, 4)
            row["bandit_reward"]        = round(bandit_reward, 4)
            row["bandit_best_fraction"] = round(bandit.best_fraction, 4)

        # Per-bidder bid and valuation columns (all active bidders)
        active_bidders = list(FIXED_BIDDERS.keys()) + ["ai_bidder"]
        if bandit is not None:
            active_bidders.append("bandit_bidder")

        for bidder_id in active_bidders:
            row[f"{bidder_id}_bid"]   = round(bids[bidder_id], 4)
            row[f"{bidder_id}_value"] = round(valuations[bidder_id], 4)

        records.append(row)

    return pd.DataFrame(records)
