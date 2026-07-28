from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from jackbot import (
    ACTION_CONSEQUENCE_SIZE,
    ACTION_SIZE,
    BELIEF_SIZE,
    OBS_SIZE,
    PUBLIC_HISTORY_SIZE,
)
from jackbot.training.config import TrainConfig, feature_flags
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
    def __init__(
        self,
        hidden_size: int = 2048,
        feature_schema: str = "base_v1",
        centralized_critic: bool = False,
        critic_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.feature_schema = feature_schema
        self.use_action_consequences, self.use_public_history = feature_flags(feature_schema)
        self.centralized_critic = centralized_critic
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
        self.history_adapter = (
            nn.Linear(PUBLIC_HISTORY_SIZE, hidden_size, bias=False)
            if self.use_public_history
            else None
        )
        self.action_consequence_adapter = (
            nn.Linear(ACTION_CONSEQUENCE_SIZE, hidden_size, bias=False)
            if self.use_action_consequences
            else None
        )
        critic_input_size = OBS_SIZE + BELIEF_SIZE
        if self.use_public_history:
            critic_input_size += PUBLIC_HISTORY_SIZE
        self.central_value_delta = (
            nn.Sequential(
                nn.Linear(critic_input_size, critic_hidden_size),
                nn.GELU(),
                nn.Linear(critic_hidden_size, 1),
            )
            if centralized_critic
            else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        _init_mlp_linear_layers(self.obs_trunk, _gelu_gain())
        _init_mlp_linear_layers(self.action_head, _gelu_gain())
        _orthogonal_linear(self.action_head[-1], 0.01)
        _orthogonal_linear(self.value_head, 1.0)
        _orthogonal_linear(self.belief_head, 1.0)
        if self.history_adapter is not None:
            nn.init.zeros_(self.history_adapter.weight)
        if self.action_consequence_adapter is not None:
            nn.init.zeros_(self.action_consequence_adapter.weight)
        if self.central_value_delta is not None:
            _init_mlp_linear_layers(self.central_value_delta, _gelu_gain())
            nn.init.zeros_(self.central_value_delta[-1].weight)
            nn.init.zeros_(self.central_value_delta[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        action_features: torch.Tensor,
        action_offsets: torch.Tensor,
        public_history: torch.Tensor | None = None,
        action_consequences: torch.Tensor | None = None,
        privileged_hands: torch.Tensor | None = None,
    ) -> PolicyOutput:
        trunk = self._observation_trunk(obs, public_history)
        env_ids = segment_env_ids(action_offsets)
        action_context = trunk[env_ids]
        action_hidden = self.action_head[0](torch.cat([action_context, action_features], dim=-1))
        if self.action_consequence_adapter is not None:
            consequences = _required_features(
                action_consequences,
                ACTION_CONSEQUENCE_SIZE,
                "action consequences",
            )
            action_hidden = action_hidden + self.action_consequence_adapter(consequences)
        action_hidden = self.action_head[1](action_hidden)
        logits = self.action_head[2](action_hidden).squeeze(-1)
        log_probs = segment_log_softmax(logits, action_offsets)
        values = self._values(obs, trunk, public_history, privileged_hands)
        belief_logits = self.belief_head(trunk)
        return PolicyOutput(logits, log_probs, values, belief_logits)

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        action_features: torch.Tensor,
        action_offsets: torch.Tensor,
        deterministic: bool = False,
        public_history: torch.Tensor | None = None,
        action_consequences: torch.Tensor | None = None,
        privileged_hands: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(
            obs,
            action_features,
            action_offsets,
            public_history,
            action_consequences,
            privileged_hands,
        )
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
        public_history: torch.Tensor | None = None,
        action_consequences: torch.Tensor | None = None,
        privileged_hands: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(
            obs,
            action_features,
            action_offsets,
            public_history,
            action_consequences,
            privileged_hands,
        )
        selected = chosen_flat_indices(action_offsets, actions)
        entropies = segment_entropy(output.logits, action_offsets)
        return (
            output.log_probs[selected],
            output.values,
            output.belief_logits,
            entropies,
            output.logits,
        )

    def value(
        self,
        obs: torch.Tensor,
        public_history: torch.Tensor | None = None,
        privileged_hands: torch.Tensor | None = None,
    ) -> torch.Tensor:
        trunk = self._observation_trunk(obs, public_history)
        return self._values(obs, trunk, public_history, privileged_hands)

    def _observation_trunk(
        self,
        obs: torch.Tensor,
        public_history: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden = self.obs_trunk[0](obs)
        if self.history_adapter is not None:
            history = _required_features(public_history, PUBLIC_HISTORY_SIZE, "public history")
            hidden = hidden + self.history_adapter(history)
        hidden = self.obs_trunk[1](hidden)
        hidden = self.obs_trunk[2](hidden)
        return self.obs_trunk[3](hidden)

    def _values(
        self,
        obs: torch.Tensor,
        trunk: torch.Tensor,
        public_history: torch.Tensor | None,
        privileged_hands: torch.Tensor | None,
    ) -> torch.Tensor:
        values = self.value_head(trunk).squeeze(-1)
        if self.central_value_delta is None or privileged_hands is None:
            return values
        critic_parts = [obs]
        if self.use_public_history:
            critic_parts.append(
                _required_features(public_history, PUBLIC_HISTORY_SIZE, "public history")
            )
        critic_parts.append(_required_features(privileged_hands, BELIEF_SIZE, "privileged hands"))
        delta = self.central_value_delta(torch.cat(critic_parts, dim=-1)).squeeze(-1)
        return values + delta


def make_model(config: TrainConfig) -> JackbotNet:
    return JackbotNet(
        hidden_size=config.hidden_size,
        feature_schema=config.feature_schema,
        centralized_critic=config.centralized_critic,
        critic_hidden_size=config.critic_hidden_size,
    )


def _required_features(
    features: torch.Tensor | None,
    expected_size: int,
    label: str,
) -> torch.Tensor:
    if features is None or features.shape[-1] != expected_size:
        actual = None if features is None else features.shape[-1]
        raise ValueError(f"expected {label} size {expected_size}, got {actual}")
    return features


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
