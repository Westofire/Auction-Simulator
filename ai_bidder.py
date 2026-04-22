"""
ai_bidder.py
------------
A reinforcement learning bidder that learns optimal bidding strategies
over successive auction rounds using Q-learning with epsilon-greedy exploration.

Bid space : discrete levels from 0.5x to 1.0x of the bidder's valuation.
Q-table   : maps bid_level_index -> expected reward.
Reward    : valuation - price_paid if won, else 0.
"""

import random
import numpy as np


class AIBidder:
    """
    Q-Learning bidder that adapts its bidding strategy over time.

    The action space is a fixed set of discrete bid fractions applied
    to the bidder's valuation each round. The Q-table stores the expected
    cumulative reward for each action and is updated after every auction.
    """

    def __init__(
        self,
        bidder_id: str = "ai_bidder",
        n_levels: int = 11,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
    ):
        """
        Args:
            bidder_id     (str):   Unique identifier for this bidder.
            n_levels      (int):   Number of discrete bid levels between 0.5–1.0x.
                                   Default 11 gives steps of 0.05 each.
            alpha         (float): Learning rate (0, 1].
            gamma         (float): Discount factor for future rewards [0, 1].
            epsilon       (float): Initial exploration probability.
            epsilon_decay (float): Multiplicative decay applied after each round.
            epsilon_min   (float): Floor for epsilon — ensures some exploration always.
        """
        self.bidder_id = bidder_id
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Discrete bid fractions: n_levels evenly spaced values in [0.5, 1.0]
        self.bid_fractions = np.linspace(0.5, 1.0, n_levels)
        self.n_actions = len(self.bid_fractions)

        # Q-table: one Q-value per action (state is implicitly the valuation fraction)
        self.q_table = np.zeros(self.n_actions)

        # Track the last action taken so we can update after the auction resolves
        self._last_action_index: int | None = None

        # History for analysis
        self.reward_history: list[float] = []
        self.bid_history: list[float] = []
        self.epsilon_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bid(self, valuation: float) -> float:
        """
        Choose a bid using epsilon-greedy policy and record the action.

        Args:
            valuation (float): The bidder's private value for the current item.

        Returns:
            float: The chosen bid amount (<= valuation).
        """
        if valuation < 0:
            raise ValueError(f"Valuation must be non-negative, got {valuation}.")

        action_index = self._select_action()
        self._last_action_index = action_index

        bid = round(float(self.bid_fractions[action_index] * valuation), 4)
        self.bid_history.append(bid)
        self.epsilon_history.append(self.epsilon)

        return bid

    def update(self, valuation: float, won: bool, price_paid: float = 0.0) -> float:
        """
        Update the Q-table based on the auction outcome.

        Args:
            valuation  (float): The bidder's private value for the item.
            won        (bool):  Whether this bidder won the auction.
            price_paid (float): The price the winner paid (0 if bidder lost).

        Returns:
            float: The reward signal received this round.
        """
        if self._last_action_index is None:
            raise RuntimeError("update() called before get_bid() in this round.")

        reward = (valuation - price_paid) if won else 0.0

        self._update_q_table(self._last_action_index, reward)
        self._decay_epsilon()

        self.reward_history.append(reward)
        self._last_action_index = None

        return reward

    def reset(self) -> None:
        """
        Fully reset the bidder's learned state and history.
        Useful between independent simulation runs.
        """
        self.q_table = np.zeros(self.n_actions)
        self._last_action_index = None
        self.reward_history = []
        self.bid_history = []
        self.epsilon_history = []
        # Reset epsilon to initial (stored at construction; default to 1.0)
        self.epsilon = 1.0

    # ------------------------------------------------------------------
    # Properties for inspection
    # ------------------------------------------------------------------

    @property
    def best_fraction(self) -> float:
        """The bid fraction the Q-table currently considers optimal."""
        return float(self.bid_fractions[np.argmax(self.q_table)])

    @property
    def total_reward(self) -> float:
        """Cumulative reward earned across all rounds."""
        return sum(self.reward_history)

    @property
    def win_count(self) -> int:
        """Number of rounds in which the bidder earned a positive reward."""
        return sum(1 for r in self.reward_history if r > 0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_action(self) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            # Explore: pick a random bid level
            return random.randint(0, self.n_actions - 1)
        else:
            # Exploit: pick the action with the highest Q-value
            return int(np.argmax(self.q_table))

    def _update_q_table(self, action_index: int, reward: float) -> None:
        """
        Q-learning update rule (single-state, no next-state transition):
            Q(a) <- Q(a) + alpha * (reward + gamma * max(Q) - Q(a))

        Because every auction is a fresh item (no persistent state carries over),
        the future value term uses the current best Q-value as a soft target.
        """
        current_q = self.q_table[action_index]
        best_future_q = np.max(self.q_table)
        self.q_table[action_index] = current_q + self.alpha * (
            reward + self.gamma * best_future_q - current_q
        )

    def _decay_epsilon(self) -> None:
        """Decay epsilon after each round, respecting the minimum floor."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AIBidder(id={self.bidder_id!r}, "
            f"epsilon={self.epsilon:.3f}, "
            f"best_fraction={self.best_fraction:.2f}, "
            f"total_reward={self.total_reward:.2f})"
        )
