from __future__ import annotations

import torch

from jackbot import ACTION_SIZE, OBS_SIZE
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
