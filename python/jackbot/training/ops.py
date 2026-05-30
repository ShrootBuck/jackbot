from __future__ import annotations

import torch


def segment_env_ids(offsets: torch.Tensor) -> torch.Tensor:
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(counts.numel(), device=offsets.device, dtype=torch.long),
        counts.to(torch.long),
    )


def segment_log_softmax(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    env_ids = segment_env_ids(offsets)
    segment_count = offsets.numel() - 1
    max_values = torch.full(
        (segment_count,),
        -torch.inf,
        device=logits.device,
        dtype=logits.dtype,
    )
    max_values.scatter_reduce_(0, env_ids, logits, reduce="amax", include_self=True)
    centered = logits - max_values[env_ids]
    exp_values = centered.exp()
    sums = torch.zeros(segment_count, device=logits.device, dtype=logits.dtype)
    sums.scatter_add_(0, env_ids, exp_values)
    return centered - sums[env_ids].clamp_min(1e-12).log()


def segment_entropy(logits: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    log_probs = segment_log_softmax(logits, offsets)
    probs = log_probs.exp()
    env_ids = segment_env_ids(offsets)
    entropy = torch.zeros(offsets.numel() - 1, device=logits.device, dtype=logits.dtype)
    entropy.scatter_add_(0, env_ids, -(probs * log_probs))
    return entropy


def segment_gumbel_sample(log_probs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    actions = []
    probs = log_probs.exp()
    for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True):
        segment = probs[start:end]
        actions.append(int(torch.multinomial(segment, 1).item()))
    return torch.tensor(actions, device=log_probs.device, dtype=torch.long)


def chosen_flat_indices(offsets: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return offsets[:-1].to(torch.long) + actions.to(torch.long)
