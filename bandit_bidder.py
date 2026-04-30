"""
bandit_bidder.py
----------------
A Multi-Armed Bandit bidder that learns optimal bid levels through either:
  - UCB  (Upper Confidence Bound): optimistic exploration via confidence intervals
  - Thompson Sampling            : probabilistic exploration via Beta distributions

Design mirrors AIBidder (ai_bidder.py) exactly — same public API:
    get_bid(valuation)  ->  float
    update(valuation, won, price_paid)  ->  float (reward)
    reset()

Bid space : same discrete levels as AIBidder — linspace(0.5, 1.0, n_levels)
Reward    : valuation - price_paid if won, else 0  (identical to Q-learner)
"""

import math
import numpy as np


class BanditBidder:
    """
    Multi-Armed Bandit bidder for auction environments.

    Each discrete bid fraction (arm) has statistics that are updated after
    every round. The agent selects arms by either UCB or Thompson Sampling.

    UCB (Upper Confidence Bound):
        score(a) = mean_reward(a) + c * sqrt(ln(t) / n(a))
        Picks the arm with the highest optimistic upper bound.
        Exploration is automatic — under-tried arms get a bonus.

    Thompson Sampling:
        Models each arm's reward as a Beta(alpha, beta) distribution.
        Draws a sample from each arm's posterior, picks the highest draw.
        Naturally balances exploration vs exploitation probabilistically.
        Note: rewards are normalised to [0,1] for Beta compatibility.
    """

    def __init__(
        self,
        bidder_id: str = "bandit_bidder",
        n_levels: int = 11,
        algorithm: str = "ucb",        # "ucb" | "thompson"
        ucb_c: float = 2.0,            # UCB exploration coefficient
    ):
        """
        Args:
            bidder_id  (str):   Unique identifier shown in simulation results.
            n_levels   (int):   Number of discrete bid fractions in [0.5, 1.0].
                                Must match AIBidder's n_levels for fair comparison.
            algorithm  (str):   "ucb" for Upper Confidence Bound,
                                "thompson" for Thompson Sampling.
            ucb_c      (float): Exploration coefficient for UCB.
                                Higher = more exploration. Typical range: 1–3.
        """
        if algorithm not in ("ucb", "thompson"):
            raise ValueError(f"algorithm must be 'ucb' or 'thompson', got {algorithm!r}.")

        self.bidder_id = bidder_id
        self.algorithm = algorithm
        self.ucb_c     = ucb_c

        # Discrete bid fractions: same space as AIBidder for fair comparison
        self.bid_fractions = np.linspace(0.5, 1.0, n_levels)
        self.n_actions     = len(self.bid_fractions)

        # ── UCB statistics ──────────────────────────────────────────────
        # counts[a]       : number of times arm a was pulled
        # mean_rewards[a] : running mean reward for arm a
        self.counts       = np.zeros(self.n_actions)
        self.mean_rewards = np.zeros(self.n_actions)

        # ── Thompson Sampling statistics (Beta distribution) ─────────────
        # alpha_params[a] : pseudo-successes  (reward > 0 pulls)
        # beta_params[a]  : pseudo-failures   (reward == 0 pulls)
        # Initialised to (1, 1) = uniform prior (no preference)
        self.alpha_params = np.ones(self.n_actions)
        self.beta_params  = np.ones(self.n_actions)

        # Global round counter (used by UCB ln(t) term)
        self._t: int = 0

        # Last chosen action index — needed to match get_bid → update flow
        self._last_action_index: int | None = None

        # History for charts and analysis (mirrors AIBidder interface)
        self.reward_history: list[float] = []
        self.bid_history:    list[float] = []

    # ------------------------------------------------------------------
    # Public API  (mirrors AIBidder exactly)
    # ------------------------------------------------------------------

    def get_bid(self, valuation: float) -> float:
        """
        Select a bid level using the configured bandit algorithm.

        Args:
            valuation (float): Bidder's private value for the current item.

        Returns:
            float: Chosen bid amount, guaranteed <= valuation.
        """
        if valuation < 0:
            raise ValueError(f"Valuation must be non-negative, got {valuation}.")

        self._t += 1  # increment global round counter before selection

        if self.algorithm == "ucb":
            action_index = self._select_ucb()
        else:
            action_index = self._select_thompson()

        self._last_action_index = action_index

        bid = round(float(self.bid_fractions[action_index] * valuation), 4)
        self.bid_history.append(bid)

        return bid

    def update(self, valuation: float, won: bool, price_paid: float = 0.0) -> float:
        """
        Update bandit statistics based on auction outcome.

        Args:
            valuation  (float): Bidder's private value for the item.
            won        (bool):  True if this bidder won the auction.
            price_paid (float): Price paid by the winner (0 if this bidder lost).

        Returns:
            float: Reward received this round.
        """
        if self._last_action_index is None:
            raise RuntimeError("update() called before get_bid() in this round.")

        # Reward: profit if won, 0 if lost — identical to AIBidder
        reward = (valuation - price_paid) if won else 0.0

        a = self._last_action_index
        self._update_ucb_stats(a, reward)
        self._update_thompson_stats(a, reward, valuation)

        self.reward_history.append(reward)
        self._last_action_index = None

        return reward

    def reset(self) -> None:
        """
        Fully reset learned state and history.
        Call between independent simulation runs.
        """
        self.counts       = np.zeros(self.n_actions)
        self.mean_rewards = np.zeros(self.n_actions)
        self.alpha_params = np.ones(self.n_actions)
        self.beta_params  = np.ones(self.n_actions)
        self._t                  = 0
        self._last_action_index  = None
        self.reward_history      = []
        self.bid_history         = []

    # ------------------------------------------------------------------
    # Properties for inspection / charts
    # ------------------------------------------------------------------

    @property
    def best_fraction(self) -> float:
        """Bid fraction the bandit currently considers best (highest mean reward)."""
        return float(self.bid_fractions[np.argmax(self.mean_rewards)])

    @property
    def total_reward(self) -> float:
        """Cumulative reward across all rounds."""
        return float(sum(self.reward_history))

    @property
    def win_count(self) -> int:
        """Number of rounds where the bidder earned a positive reward."""
        return sum(1 for r in self.reward_history if r > 0)

    # ------------------------------------------------------------------
    # Private helpers — UCB
    # ------------------------------------------------------------------

    def _select_ucb(self) -> int:
        """
        UCB1 arm selection.

        For any arm never tried, score = +inf (force exploration first).
        Otherwise: score = mean_reward + c * sqrt(ln(t) / n(a))
        """
        scores = np.full(self.n_actions, np.inf)

        for a in range(self.n_actions):
            if self.counts[a] > 0:
                exploration_bonus = self.ucb_c * math.sqrt(
                    math.log(self._t) / self.counts[a]
                )
                scores[a] = self.mean_rewards[a] + exploration_bonus

        return int(np.argmax(scores))

    def _update_ucb_stats(self, action_index: int, reward: float) -> None:
        """
        Incremental mean update (Welford-style) for UCB statistics.
            new_mean = old_mean + (reward - old_mean) / new_count
        """
        self.counts[action_index] += 1
        n = self.counts[action_index]
        self.mean_rewards[action_index] += (reward - self.mean_rewards[action_index]) / n

    # ------------------------------------------------------------------
    # Private helpers — Thompson Sampling
    # ------------------------------------------------------------------

    def _select_thompson(self) -> int:
        """
        Thompson Sampling arm selection.

        Draw one sample from each arm's Beta(alpha, beta) posterior.
        Pick the arm with the highest sample.

        Beta distribution requires rewards in [0, 1].
        We normalise by treating any positive reward as a "success" signal
        and use the normalised reward magnitude to weight the alpha update.
        """
        samples = np.random.beta(self.alpha_params, self.beta_params)
        return int(np.argmax(samples))

    def _update_thompson_stats(
        self, action_index: int, reward: float, valuation: float
    ) -> None:
        """
        Update Beta posterior for Thompson Sampling.

        Success (reward > 0): increment alpha by normalised reward in (0,1].
        Failure (reward == 0): increment beta by 1 (one more failure count).

        Normalising by valuation maps reward into (0, 1] so it fits Beta's
        support — a larger profit fraction produces a stronger alpha boost.
        """
        if reward > 0 and valuation > 0:
            # Normalised reward: how much of max possible profit did we capture?
            normalised = min(reward / valuation, 1.0)
            self.alpha_params[action_index] += normalised
        else:
            # No reward this round — count as failure
            self.beta_params[action_index] += 1.0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BanditBidder(id={self.bidder_id!r}, "
            f"algorithm={self.algorithm!r}, "
            f"best_fraction={self.best_fraction:.2f}, "
            f"total_reward={self.total_reward:.2f}, "
            f"rounds={self._t})"
        )
