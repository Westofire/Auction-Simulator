"""
simulation.py
-------------
Orchestrates the full auction simulation across N rounds.

Participants:
  - One AIBidder     (Q-learning)                    — always active
  - One BanditBidder (UCB or Thompson Sampling)      — optional
  - One DQNBidder    (Deep Q-Network via PyTorch)    — optional
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
from dqn_bidder import DQNBidder


# ---------------------------------------------------------------------------
# Bidder registry
# ---------------------------------------------------------------------------

FIXED_BIDDERS = {
    "truthful": truthful_strategy,
    "shaded":   shaded_strategy,
    "random":   random_strategy,
}



# ---------------------------------------------------------------------------
# Regret helper
# ---------------------------------------------------------------------------

def compute_optimal_profit(
    valuation: float,
    auction_type: str,
    shade_factor: float,
    all_bids: dict,
    agent_id: str,
) -> float:
    """
    Compute the profit the agent would have earned using the optimal strategy
    for this auction type, given the same valuation and the same competitors.

    Second-Price optimal = truthful bidding (bid = valuation).
    First-Price  optimal = bid shading     (bid = valuation * shade_factor).

    We replace only this agent's bid with the optimal bid, re-run the auction
    logic on the same competitor bids, and return the resulting profit.

    Args:
        valuation    (float): Agent's private value this round.
        auction_type (str):   "first" or "second".
        shade_factor (float): Shade fraction (used only for first-price).
        all_bids     (dict):  Full bid dict from this round (all bidders).
        agent_id     (str):   The agent whose optimal profit we are computing.

    Returns:
        float: Profit under optimal strategy (0 if optimal bid would lose).
    """
    # Compute the optimal bid for this agent
    if auction_type == "second":
        optimal_bid = valuation                      # truthful
    else:
        optimal_bid = valuation * shade_factor       # shaded

    # Build a counterfactual bid dict — replace only this agent's bid
    counterfactual_bids = {k: v for k, v in all_bids.items()}
    counterfactual_bids[agent_id] = optimal_bid

    # Re-run the same auction engine on counterfactual bids
    if auction_type == "first":
        from auction_engine import first_price_auction
        winner_id, price_paid = first_price_auction(counterfactual_bids)
    else:
        from auction_engine import second_price_auction
        winner_id, price_paid = second_price_auction(counterfactual_bids)

    if winner_id == agent_id:
        return valuation - price_paid
    return 0.0


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
    include_dqn: bool = False,             # whether to include DQNBidder
    dqn_epsilon_decay: float = 0.995,      # DQN epsilon decay per round
    dqn_gamma: float = 0.95,               # DQN discount factor
    dqn_lr: float = 1e-3,                  # DQN Adam learning rate
    dqn_batch_size: int = 32,              # DQN replay mini-batch size
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
        include_dqn      (bool):  If True, adds a DQNBidder to the simulation.
        dqn_epsilon_decay(float): DQN epsilon decay rate per round.
        dqn_gamma        (float): DQN discount factor.
        dqn_lr           (float): DQN Adam learning rate.
        dqn_batch_size   (int):   DQN replay buffer mini-batch size.
        seed             (int|None): Random seed for reproducibility.

    Returns:
        pd.DataFrame: One row per round. Bandit/DQN columns present only when
                      the respective agent is enabled.
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

    # Instantiate DQNBidder if requested
    dqn = DQNBidder(
        bidder_id="dqn_bidder",
        n_levels=ai_n_levels,
        val_high=val_high,
        epsilon_decay=dqn_epsilon_decay,
        gamma=dqn_gamma,
        lr=dqn_lr,
        batch_size=dqn_batch_size,
    ) if include_dqn else None

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
        if dqn is not None:
            valuations["dqn_bidder"] = random.uniform(val_low, val_high)

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

        # Collect DQN bid if active
        if dqn is not None:
            dqn_val = valuations["dqn_bidder"]
            dqn_bid = dqn.get_bid(dqn_val)
            bids["dqn_bidder"] = dqn_bid

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

        # Update DQN bidder if active
        if dqn is not None:
            dqn_won = (winner_id == "dqn_bidder")
            dqn_reward = dqn.update(
                valuation=valuations["dqn_bidder"],
                won=dqn_won,
                price_paid=price_paid if dqn_won else 0.0,
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

        # DQN metrics — only present when include_dqn=True
        if dqn is not None:
            dqn_val = valuations["dqn_bidder"]
            dqn_bid = bids["dqn_bidder"]
            row["dqn_bid"]           = round(dqn_bid, 4)
            row["dqn_valuation"]     = round(dqn_val, 4)
            row["dqn_bid_ratio"]     = round(dqn_bid / dqn_val if dqn_val > 0 else 0.0, 4)
            row["dqn_reward"]        = round(dqn_reward, 4)
            row["dqn_epsilon"]       = round(dqn.epsilon, 4)
            row["dqn_best_fraction"] = round(dqn.best_fraction, 4)
            row["dqn_loss"]          = round(dqn.loss_history[-1] if dqn.loss_history else 0.0, 6)

        # Per-bidder bid and valuation columns (all active bidders)
        active_bidders = list(FIXED_BIDDERS.keys()) + ["ai_bidder"]
        if bandit is not None:
            active_bidders.append("bandit_bidder")
        if dqn is not None:
            active_bidders.append("dqn_bidder")

        for bidder_id in active_bidders:
            row[f"{bidder_id}_bid"]   = round(bids[bidder_id], 4)
            row[f"{bidder_id}_value"] = round(valuations[bidder_id], 4)

        # 7. Regret calculation — regret = optimal_profit − actual_profit
        #    Uses counterfactual bids: same competitors, optimal bid for agent
        ai_optimal  = compute_optimal_profit(ai_val, auction_type, shade_factor, bids, "ai_bidder")
        row["ai_regret"] = round(max(0.0, ai_optimal - ai_reward), 4)

        if bandit is not None:
            b_val = valuations["bandit_bidder"]
            b_opt = compute_optimal_profit(b_val, auction_type, shade_factor, bids, "bandit_bidder")
            row["bandit_regret"] = round(max(0.0, b_opt - bandit_reward), 4)

        if dqn is not None:
            d_val = valuations["dqn_bidder"]
            d_opt = compute_optimal_profit(d_val, auction_type, shade_factor, bids, "dqn_bidder")
            row["dqn_regret"] = round(max(0.0, d_opt - dqn_reward), 4)

        records.append(row)

    return pd.DataFrame(records)
