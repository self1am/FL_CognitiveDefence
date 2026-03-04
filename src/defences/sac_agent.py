# src/defences/sac_agent.py
"""
Soft Actor-Critic (SAC) agent for the POSG cognitive defence.

State:  Concatenated GRU belief states of all active clients  (dim = N × hidden_dim).
Action: Continuous weight vector  a ∈ [0, 1]^N  controlling per-client
        contribution to the federated aggregation.

Key design choices
------------------
* **Beta-distribution policy head** – maps the actor output to (0, 1) per
  client via a Beta(α, β) parameterisation.  This keeps actions in the valid
  range *and* makes the stochastic policy smooth, which is critical for the
  entropy-regularised SAC objective.
* **Twin Q-networks** – standard SAC trick to mitigate overestimation.
* **Automatic entropy tuning** – learns the temperature α online.

The agent exposes a simple ``select_action`` / ``update`` API so the POSG
defence can treat it as a black box.
"""

from __future__ import annotations

import copy
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


# ---------------------------------------------------------------------------
# Helper: MLP builder
# ---------------------------------------------------------------------------

def _mlp(dims: list[int], activation: type = nn.ReLU, output_activation: type = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class BetaPolicyNetwork(nn.Module):
    """
    Actor that outputs Beta-distribution parameters for each client weight.

    Given state s, produces  α(s), β(s) ∈ ℝ₊^N  so that
        a_i ~ Beta(α_i, β_i)    ∈ (0, 1)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.shared = _mlp([state_dim] + hidden_dims, activation=nn.ReLU)
        self.alpha_head = nn.Linear(hidden_dims[-1], action_dim)
        self.beta_head = nn.Linear(hidden_dims[-1], action_dim)
        # Optimistic initialisation: bias the Beta toward high weights initially.
        # Target mode = (α-1)/(α+β-2) ≈ 0.8  →  α≈5, β≈2
        # softplus(4.0) + 1 ≈ 5.02  |  softplus(0.54) + 1 ≈ 2.02
        nn.init.constant_(self.alpha_head.bias, 4.0)
        nn.init.constant_(self.beta_head.bias, 0.54)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(state)
        # Softplus ensures positivity; +1 keeps α,β ≥ 1 (unimodal by default)
        alpha = F.softplus(self.alpha_head(h)) + 1.0
        beta = F.softplus(self.beta_head(h)) + 1.0
        return alpha, beta

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action & compute log-probability.

        Returns
        -------
        action : Tensor  (action_dim,)  values in (0, 1)
        log_prob : Tensor  scalar – sum of per-dimension log probs
        """
        alpha, beta_param = self.forward(state)
        dist = Beta(alpha, beta_param)
        # rsample for reparameterisation trick
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        """Return the mode of the Beta distribution (for evaluation)."""
        alpha, beta_param = self.forward(state)
        # Mode of Beta(α,β) = (α-1)/(α+β-2) when α,β > 1
        mode = (alpha - 1.0) / (alpha + beta_param - 2.0 + 1e-8)
        return mode.clamp(0.0, 1.0)


class TwinQNetwork(nn.Module):
    """Twin soft Q-networks  Q₁(s,a), Q₂(s,a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]
        self.q1 = _mlp([state_dim + action_dim] + hidden_dims + [1])
        self.q2 = _mlp([state_dim + action_dim] + hidden_dims + [1])

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple numpy replay buffer for transitions (s, a, r, s', done)."""

    def __init__(self, state_dim: int, action_dim: int, capacity: int = 100_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device = torch.device("cpu")):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.states[idxs]).to(device),
            torch.from_numpy(self.actions[idxs]).to(device),
            torch.from_numpy(self.rewards[idxs]).to(device),
            torch.from_numpy(self.next_states[idxs]).to(device),
            torch.from_numpy(self.dones[idxs]).to(device),
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    Soft Actor-Critic agent with automatic temperature tuning.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the concatenated belief state.
    action_dim : int
        Number of clients (each gets a continuous weight in [0, 1]).
    hidden_dims : list[int]
        Hidden-layer sizes for actor & critic MLPs.
    lr_actor, lr_critic, lr_alpha : float
        Learning rates.
    gamma : float
        Discount factor for the long-horizon reward.
    tau : float
        Polyak averaging coefficient for target networks.
    buffer_capacity : int
        Replay buffer size.
    batch_size : int
        Mini-batch size for gradient updates.
    init_alpha : float
        Initial entropy temperature.
    device : str
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_capacity: int = 100_000,
        batch_size: int = 64,
        init_alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_dim = action_dim
        hidden_dims = hidden_dims or [256, 256]

        # Networks
        self.actor = BetaPolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TwinQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)

        # Freeze target
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimisers
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim)  # heuristic: -dim(A)
        self.log_alpha = torch.tensor(math.log(init_alpha), requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay buffer
        self.replay = ReplayBuffer(state_dim, action_dim, buffer_capacity)

        # Training flag
        self._training = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Given a state vector, return an action (weight vector in [0,1]^N).

        Parameters
        ----------
        state : ndarray of shape ``(state_dim,)``
        deterministic : bool
            If True use the policy mode (no sampling).

        Returns
        -------
        action : ndarray of shape ``(action_dim,)``
        """
        with torch.no_grad():
            s = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            if deterministic:
                action = self.actor.deterministic(s)
            else:
                action, _ = self.actor.sample(s)
            return action.squeeze(0).cpu().numpy()

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool) -> None:
        self.replay.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, min_buffer_size: int = 256) -> Optional[dict]:
        """
        Perform a single SAC gradient step if the buffer is large enough.

        Returns a dict of loss metrics or ``None`` if the buffer is too small.
        """
        if len(self.replay) < max(self.batch_size, min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size, self.device
        )

        # ---- Critic update ----
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            td_target = rewards + self.gamma * (1.0 - dones) * q_target

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # ---- Actor update ----
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ---- Alpha (temperature) update ----
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # ---- Soft-update target ----
        self._polyak_update()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _polyak_update(self) -> None:
        for p, p_target in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_target.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha,
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "alpha_optim": self.alpha_optim.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])

    def set_training(self, mode: bool = True) -> None:
        """Toggle training / evaluation mode."""
        self._training = mode
        self.actor.train(mode)
        self.critic.train(mode)
