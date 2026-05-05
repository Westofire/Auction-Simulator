"""
dqn_bidder.py
-------------
A Deep Q-Network (DQN) bidder for auction environments.

Architecture:
    Input  : 1 neuron  — normalised valuation (valuation / val_high)
    Hidden : 2 × 32 neurons with ReLU activations
    Output : n_levels neurons — one Q-value per discrete bid fraction

Training:
    - Experience replay buffer (deque) stores (state, action, reward, next_state)
    - Mini-batch sampled each round once buffer has enough transitions
    - MSE loss between predicted Q(s,a) and target r + γ * max Q(s',a')
    - Epsilon-greedy exploration with exponential decay

Public API mirrors AIBidder and BanditBidder exactly:
    get_bid(valuation)                         -> float
    update(valuation, won, price_paid)         -> float  (reward)
    reset()

Bid space : linspace(0.5, 1.0, n_levels) — same as all other AI agents.
Reward    : valuation − price_paid if won, 0 if lost — identical to others.
"""

import random
from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Neural network definition
# ---------------------------------------------------------------------------

class _QNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Fully-connected Q-network.
    Input  : 1  (normalised valuation)
    Hidden : 32 → 32 (ReLU)
    Output : n_actions (one Q-value per bid fraction)
    """

    def __init__(self, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# DQN Bidder
# ---------------------------------------------------------------------------

class DQNBidder:
    """
    Deep Q-Network bidder.

    Uses a small neural network to approximate the Q-function over
    discrete bid fractions. Trained online via experience replay.

    Falls back to random bidding with a warning if PyTorch is not installed.
    """

    def __init__(
        self,
        bidder_id: str = "dqn_bidder",
        n_levels: int = 11,
        val_high: float = 100.0,           # used to normalise input to [0,1]
        epsilon: float = 1.0,              # initial exploration rate
        epsilon_decay: float = 0.995,      # multiplicative decay per round
        epsilon_min: float = 0.05,         # exploration floor
        gamma: float = 0.95,               # discount factor
        lr: float = 1e-3,                  # Adam learning rate
        batch_size: int = 32,              # replay mini-batch size
        buffer_size: int = 2000,           # max transitions stored
    ):
        """
        Args:
            bidder_id     (str):   Unique ID shown in simulation results.
            n_levels      (int):   Discrete bid fractions in [0.5, 1.0].
                                   Must match other agents for fair comparison.
            val_high      (float): Upper bound of valuation distribution.
                                   Used to normalise network input to [0, 1].
            epsilon       (float): Initial exploration probability.
            epsilon_decay (float): Decay multiplied each round.
            epsilon_min   (float): Minimum exploration floor.
            gamma         (float): Discount factor for future rewards.
            lr            (float): Adam optimiser learning rate.
            batch_size    (int):   Number of transitions sampled per update.
            buffer_size   (int):   Capacity of the experience replay buffer.
        """
        self.bidder_id     = bidder_id
        self.val_high      = val_high
        self.epsilon       = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min   = epsilon_min
        self.gamma         = gamma
        self.batch_size    = batch_size

        # Discrete bid fractions — identical space to AIBidder / BanditBidder
        self.bid_fractions = np.linspace(0.5, 1.0, n_levels)
        self.n_actions     = len(self.bid_fractions)

        # ── PyTorch components (skipped if torch not available) ──────────
        if TORCH_AVAILABLE:
            self.device    = torch.device("cpu")   # CPU is fine for this scale
            self.q_net     = _QNetwork(self.n_actions).to(self.device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
            self.loss_fn   = nn.MSELoss()
        else:
            self.q_net = None

        # Experience replay buffer: stores (state, action, reward, next_state)
        self.replay_buffer: deque = deque(maxlen=buffer_size)

        # State tracking across get_bid → update pairs
        self._last_action_index: int | None  = None
        self._last_state:        float | None = None   # normalised valuation

        # History for charts and analysis
        self.reward_history: list[float] = []
        self.bid_history:    list[float] = []
        self.loss_history:   list[float] = []   # training loss per round

    # ------------------------------------------------------------------
    # Public API  (mirrors AIBidder / BanditBidder)
    # ------------------------------------------------------------------

    def get_bid(self, valuation: float) -> float:
        """
        Select a bid using epsilon-greedy over network Q-values.

        Args:
            valuation (float): Bidder's private value for the current item.

        Returns:
            float: Chosen bid amount, guaranteed <= valuation.
        """
        if valuation < 0:
            raise ValueError(f"Valuation must be non-negative, got {valuation}.")

        # Normalise input to [0, 1] for stable network training
        norm_val = valuation / self.val_high
        self._last_state = norm_val

        if not TORCH_AVAILABLE or random.random() < self.epsilon:
            # Explore: random action
            action_index = random.randint(0, self.n_actions - 1)
        else:
            # Exploit: greedy action from Q-network
            action_index = self._greedy_action(norm_val)

        self._last_action_index = action_index

        bid = round(float(self.bid_fractions[action_index] * valuation), 4)
        self.bid_history.append(bid)

        return bid

    def update(self, valuation: float, won: bool, price_paid: float = 0.0) -> float:
        """
        Store transition, train network, decay epsilon.

        Args:
            valuation  (float): Bidder's private value for the item.
            won        (bool):  Whether this bidder won the auction.
            price_paid (float): Price paid by the winner (0 if this bidder lost).

        Returns:
            float: Reward received this round.
        """
        if self._last_action_index is None:
            raise RuntimeError("update() called before get_bid() in this round.")

        # Reward: profit if won, 0 if lost — same as all other agents
        reward = (valuation - price_paid) if won else 0.0

        # Next state = same normalised valuation (stateless auction setting)
        next_state = valuation / self.val_high

        # Store transition in replay buffer
        self.replay_buffer.append((
            self._last_state,
            self._last_action_index,
            reward,
            next_state,
        ))

        # Train on a mini-batch once buffer has enough samples
        loss_val = 0.0
        if TORCH_AVAILABLE and len(self.replay_buffer) >= self.batch_size:
            loss_val = self._train_step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.reward_history.append(reward)
        self.loss_history.append(loss_val)
        self._last_action_index = None
        self._last_state        = None

        return reward

    def reset(self) -> None:
        """
        Fully reset learned weights, buffer, and history.
        Call between independent simulation runs.
        """
        if TORCH_AVAILABLE:
            self.q_net     = _QNetwork(self.n_actions).to(self.device)
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)

        self.replay_buffer      = deque(maxlen=self.replay_buffer.maxlen)
        self.epsilon            = 1.0
        self._last_action_index = None
        self._last_state        = None
        self.reward_history     = []
        self.bid_history        = []
        self.loss_history       = []

    # ------------------------------------------------------------------
    # Properties for inspection / charts
    # ------------------------------------------------------------------

    @property
    def best_fraction(self) -> float:
        """
        Bid fraction with the highest Q-value at the midpoint valuation.
        Uses val_high * 0.55 as a representative state for inspection.
        """
        if not TORCH_AVAILABLE or self.q_net is None:
            return float(self.bid_fractions[self.n_actions // 2])

        mid_val = 0.55   # normalised midpoint of valuation range
        return float(self.bid_fractions[self._greedy_action(mid_val)])

    @property
    def total_reward(self) -> float:
        """Cumulative reward across all rounds."""
        return float(sum(self.reward_history))

    @property
    def win_count(self) -> int:
        """Number of rounds where the bidder earned a positive reward."""
        return sum(1 for r in self.reward_history if r > 0)

    @property
    def torch_available(self) -> bool:
        """True if PyTorch is installed and the network is active."""
        return TORCH_AVAILABLE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _greedy_action(self, norm_val: float) -> int:
        """Run a forward pass and return the argmax action."""
        self.q_net.eval()
        with torch.no_grad():
            state_t = torch.tensor([[norm_val]], dtype=torch.float32, device=self.device)
            q_vals  = self.q_net(state_t)          # shape: (1, n_actions)
        return int(q_vals.argmax(dim=1).item())

    def _train_step(self) -> float:
        """
        Sample a mini-batch from replay buffer and do one gradient step.

        Target: y = r + γ * max_a' Q(s', a')
        Loss  : MSE(Q(s, a), y)

        Returns:
            float: Scalar loss value for this step (logged to loss_history).
        """
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        # Convert to tensors — shape (batch, 1) for states
        states_t      = torch.tensor([[s] for s in states],      dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor([[s] for s in next_states], dtype=torch.float32, device=self.device)
        actions_t     = torch.tensor(actions,  dtype=torch.long,  device=self.device)
        rewards_t     = torch.tensor(rewards,  dtype=torch.float32, device=self.device)

        # Current Q-values for the chosen actions: Q(s, a)
        self.q_net.train()
        q_all     = self.q_net(states_t)                          # (batch, n_actions)
        q_current = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (batch,)

        # Target Q-values: r + γ * max_a' Q(s', a')
        with torch.no_grad():
            q_next  = self.q_net(next_states_t)                   # (batch, n_actions)
            q_target = rewards_t + self.gamma * q_next.max(dim=1).values

        # MSE loss and gradient step
        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DQNBidder(id={self.bidder_id!r}, "
            f"epsilon={self.epsilon:.3f}, "
            f"best_fraction={self.best_fraction:.2f}, "
            f"total_reward={self.total_reward:.2f}, "
            f"torch={TORCH_AVAILABLE})"
        )
