from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from jackbot import ACTION_SIZE, BELIEF_SIZE, OBS_SIZE
from jackbot.training.ops import (
    chosen_flat_indices,
    segment_entropy,
    segment_env_ids,
    segment_gumbel_sample,
    segment_log_softmax,
)


@dataclass(slots=True)
class PolicyOutput:
    logits: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    belief_logits: torch.Tensor


class JackbotNet(nn.Module):
    def __init__(self, hidden_size: int = 2048) -> None:
        super().__init__()
        self.obs_trunk = nn.Sequential(
            nn.Linear(OBS_SIZE, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size + ACTION_SIZE, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.belief_head = nn.Linear(hidden_size, BELIEF_SIZE)
        self._init_weights()

    def _init_weights(self) -> None:
        _init_mlp_linear_layers(self.obs_trunk, _gelu_gain())
        _init_mlp_linear_layers(self.action_head, _gelu_gain())
        _orthogonal_linear(self.action_head[-1], 0.01)
        _orthogonal_linear(self.value_head, 1.0)
        _orthogonal_linear(self.belief_head, 1.0)

    def forward(
        self,
        obs: torch.Tensor,
        action_features: torch.Tensor,
        action_offsets: torch.Tensor,
    ) -> PolicyOutput:
        trunk = self.obs_trunk(obs)
        env_ids = segment_env_ids(action_offsets)
        action_context = trunk[env_ids]
        logits = self.action_head(torch.cat([action_context, action_features], dim=-1)).squeeze(-1)
        log_probs = segment_log_softmax(logits, action_offsets)
        values = self.value_head(trunk).squeeze(-1)
        belief_logits = self.belief_head(trunk)
        return PolicyOutput(logits, log_probs, values, belief_logits)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_features: torch.Tensor,
        action_offsets: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(obs, action_features, action_offsets)
        if deterministic:
            rel_actions = _segment_argmax(output.logits, action_offsets)
        else:
            rel_actions = segment_gumbel_sample(output.log_probs, action_offsets)
        selected = chosen_flat_indices(action_offsets, rel_actions)
        return rel_actions, output.log_probs[selected], output.values

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_features: torch.Tensor,
        action_offsets: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(obs, action_features, action_offsets)
        selected = chosen_flat_indices(action_offsets, actions)
        entropies = segment_entropy(output.logits, action_offsets)
        return (
            output.log_probs[selected],
            output.values,
            output.belief_logits,
            entropies,
            output.logits,
        )


def _segment_argmax(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    actions = []
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        actions.append(int(torch.argmax(logits[start:end]).item()))
    return torch.tensor(actions, device=logits.device, dtype=torch.long)


def _orthogonal_linear(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def _init_mlp_linear_layers(module: nn.Module, gain: float) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            _orthogonal_linear(layer, gain)


def _gelu_gain() -> float:
    return 2.0**0.5
