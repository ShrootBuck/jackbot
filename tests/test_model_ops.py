from __future__ import annotations

import torch

from jackbot import ACTION_CONSEQUENCE_SIZE, ACTION_SIZE, BELIEF_SIZE, OBS_SIZE, PUBLIC_HISTORY_SIZE
from jackbot.training.model import JackbotNet
from jackbot.training.ops import segment_log_softmax


def test_segment_log_softmax_sums_per_segment() -> None:
    logits = torch.tensor([1.0, 2.0, 0.5, -1.0, 3.0])
    offsets = torch.tensor([0, 2, 5])

    log_probs = segment_log_softmax(logits, offsets)

    assert torch.allclose(log_probs[0:2].exp().sum(), torch.tensor(1.0))
    assert torch.allclose(log_probs[2:5].exp().sum(), torch.tensor(1.0))


def test_model_forward_and_action_eval_shapes() -> None:
    model = JackbotNet(hidden_size=32)
    obs = torch.randn(3, OBS_SIZE)
    action_features = torch.randn(7, ACTION_SIZE)
    offsets = torch.tensor([0, 2, 5, 7])
    actions = torch.tensor([1, 0, 1])

    output = model(obs, action_features, offsets)
    log_probs, values, belief_logits, entropies, _ = model.evaluate_actions(
        obs,
        action_features,
        offsets,
        actions,
    )

    assert output.logits.shape == (7,)
    assert output.values.shape == (3,)
    assert belief_logits.shape == (3, 156)
    assert log_probs.shape == (3,)
    assert values.shape == (3,)
    assert entropies.shape == (3,)


def test_enhanced_model_is_function_preserving_at_initialization() -> None:
    torch.manual_seed(3)
    base = JackbotNet(hidden_size=32)
    enhanced = JackbotNet(
        hidden_size=32,
        feature_schema="a1h1",
        centralized_critic=True,
        critic_hidden_size=16,
    )
    result = enhanced.load_state_dict(base.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {
        "history_adapter.weight",
        "action_consequence_adapter.weight",
        "central_value_delta.0.weight",
        "central_value_delta.0.bias",
        "central_value_delta.2.weight",
        "central_value_delta.2.bias",
    }

    obs = torch.randn(3, OBS_SIZE)
    action_features = torch.randn(7, ACTION_SIZE)
    offsets = torch.tensor([0, 2, 5, 7])
    history = torch.randn(3, PUBLIC_HISTORY_SIZE)
    consequences = torch.randn(7, ACTION_CONSEQUENCE_SIZE)
    privileged = torch.randn(3, BELIEF_SIZE)
    base_output = base(obs, action_features, offsets)
    enhanced_output = enhanced(
        obs,
        action_features,
        offsets,
        history,
        consequences,
        privileged,
    )

    assert torch.equal(base_output.logits, enhanced_output.logits)
    assert torch.equal(base_output.log_probs, enhanced_output.log_probs)
    assert torch.equal(base_output.values, enhanced_output.values)
    assert torch.equal(base_output.belief_logits, enhanced_output.belief_logits)


def test_privileged_hands_can_change_value_but_never_policy() -> None:
    model = JackbotNet(
        hidden_size=32,
        centralized_critic=True,
        critic_hidden_size=16,
    )
    with torch.no_grad():
        model.central_value_delta[-1].weight.fill_(0.01)
    obs = torch.randn(2, OBS_SIZE)
    actions = torch.randn(4, ACTION_SIZE)
    offsets = torch.tensor([0, 2, 4])
    first = model(obs, actions, offsets, privileged_hands=torch.zeros(2, BELIEF_SIZE))
    second = model(obs, actions, offsets, privileged_hands=torch.ones(2, BELIEF_SIZE))

    assert torch.equal(first.logits, second.logits)
    assert not torch.equal(first.values, second.values)
