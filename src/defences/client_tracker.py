# src/defences/client_tracker.py
"""
ClientTracker: GRU-based per-client belief-state module.

For each client *i* at round *t* the tracker:
  1. Ingests an observation vector  x_i^{(t)}  (norms, cosine-sim, Fisher trace …)
  2. Feeds it through a single-layer GRU to obtain a hidden belief state  b_i^{(t)}.
  3. Exposes the belief state so the SAC policy can condition on it.

The GRU naturally captures the *temporal signature* of gradient evolution,
making it possible to detect slow-drift ("boiling-frog") poisoning attacks that
a stateless detector would miss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ClientGRU(nn.Module):
    """Single-client GRU cell that maps an observation to a belief state."""

    def __init__(self, obs_dim: int, hidden_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        # Project raw observation into the GRU input space
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : Tensor  (obs_dim,) or (batch, obs_dim)
        h_prev : Tensor  (hidden_dim,) or (batch, hidden_dim), optional

        Returns
        -------
        h_next : Tensor  (hidden_dim,)  —  the updated belief state.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.input_proj(obs)
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_dim, device=obs.device)
        elif h_prev.dim() == 1:
            h_prev = h_prev.unsqueeze(0)
        h_next = self.gru_cell(x, h_prev)
        return h_next.squeeze(0)


class ClientTracker(nn.Module):
    """
    Manages a pool of per-client GRU belief states.

    All clients share the *same* GRU weights (parameter-efficient), but each
    client has its own hidden state ``h_i`` stored in an internal dictionary.
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.gru = ClientGRU(obs_dim, hidden_dim)
        # Hidden states are *not* nn.Parameters – they are mutable buffers.
        self._hidden_states: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, client_id: str, obs: torch.Tensor) -> torch.Tensor:
        """
        Ingest a new observation for *client_id* and return updated belief.

        Parameters
        ----------
        client_id : str
        obs : Tensor of shape ``(obs_dim,)``

        Returns
        -------
        belief : Tensor of shape ``(hidden_dim,)``
        """
        h_prev = self._hidden_states.get(client_id)
        h_next = self.gru(obs, h_prev)
        self._hidden_states[client_id] = h_next.detach()  # detach to avoid graph leak
        return h_next

    def get_belief(self, client_id: str) -> torch.Tensor:
        """Return current belief state (zeros if unseen client)."""
        if client_id in self._hidden_states:
            return self._hidden_states[client_id]
        return torch.zeros(self.hidden_dim)

    def get_all_beliefs(self, client_ids: list[str]) -> torch.Tensor:
        """
        Concatenate belief states for a list of clients.

        Returns shape ``(sum of hidden_dims,)`` = ``(N * hidden_dim,)``
        """
        beliefs = [self.get_belief(cid) for cid in client_ids]
        return torch.cat(beliefs, dim=-1)

    def reset_client(self, client_id: str) -> None:
        """Clear the hidden state for a specific client."""
        self._hidden_states.pop(client_id, None)

    def reset_all(self) -> None:
        """Clear all hidden states (e.g. between experiments)."""
        self._hidden_states.clear()

    def belief_entropy(self, client_ids: list[str], eps: float = 1e-8) -> torch.Tensor:
        """
        Compute the entropy of the current trust beliefs.

        We interpret each belief vector's L1-normalised absolute values as a
        pseudo-distribution and compute Shannon entropy.  High entropy → the
        defender is uncertain → the reward should penalise this.
        """
        beliefs = torch.stack([self.get_belief(cid) for cid in client_ids])
        # Normalise to pseudo-probability over hidden dimensions
        probs = torch.abs(beliefs) + eps
        probs = probs / probs.sum(dim=-1, keepdim=True)
        entropy = -(probs * probs.log()).sum(dim=-1)  # per-client entropy
        return entropy.mean()  # scalar: average entropy across clients
