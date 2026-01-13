import torch
from typing import Callable, Any, Literal

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float = 1e-8,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """
    Computes rewards for each group of rollout responses, normalized by the group mean and std.

    This implements a group-relative reward mechanism where advantages are calculated 
    locally within each group (responses to the same prompt).

    Args:
        reward_fn: A callable that scores a single (response, ground_truth) pair. 
            Expected to return a dict containing at least the "reward" key.
        rollout_responses: A list of generated strings from the policy. 
            Length: rollout_batch_size (n_prompts * group_size).
        repeated_ground_truths: A list of ground truth strings, where each ground 
            truth is repeated `group_size` times to match the rollouts.
        group_size: The number of rollout samples generated per prompt.
        advantage_eps: A small epsilon value added to the denominator to prevent 
            division by zero during standardization.
        normalize_by_std: Whether to divide the centered rewards by the group 
            standard deviation. If False, only mean-subtraction is performed.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
            - **advantages**: Tensor of shape (rollout_batch_size,) containing the 
                normalized rewards used for policy gradient updates.
            - **raw_rewards**: Tensor of shape (rollout_batch_size,) containing the 
                unnormalized scalar rewards from reward_fn.
            - **metadata**: A dictionary containing logging statistics such as 
                mean, std, and min/max values of the rewards.
    """
    raw_rewards = []
    for i in range(len(rollout_responses)):
        reward = reward_fn(rollout_responses[i], repeated_ground_truths[i])["reward"]
        raw_rewards.append(reward)
    
    raw_rewards = torch.tensor(raw_rewards)
    grouped_rewards = raw_rewards.reshape((-1, group_size))
    mean = grouped_rewards.mean(-1, keepdim=True).expand(-1, group_size).reshape(-1)
    std = grouped_rewards.std(-1, keepdim=True).expand(-1, group_size).reshape(-1)
    advantages = raw_rewards - mean
    if normalize_by_std:
        advantages = advantages / (std + advantage_eps)
    
    metadata = {}
    return (advantages, raw_rewards, metadata)

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the policy-gradient loss at every token position.

    The loss is calculated as the negative product of the reward (or advantage) 
    and the log-probabilities. This follows the REINFORCE algorithm logic 
    where higher rewards increase the likelihood of the actions taken.

    Args:
        raw_rewards_or_advantages: Tensor of shape (batch_size, 1) or (batch_size,). 
            The scalar reward or normalized advantage assigned to each 
            complete rollout response.
        policy_log_probs: Tensor of shape (batch_size, sequence_length). 
            The log-probabilities of the tokens sampled by the policy.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, sequence_length) representing 
            the per-token policy-gradient loss. This should typically be averaged 
            over the batch or sequence length during the optimization step.
    """
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Computes the clipped surrogate loss for Group Relative Policy Optimization (GRPO).

    This calculates the PPO-style clipped objective using per-token log probabilities 
    and a sequence-level advantage.

    Args:
        advantages: Tensor of shape (batch_size, 1) representing the 
            group-normalized advantages ($A$).
        policy_log_probs: Tensor of shape (batch_size, sequence_length) 
            containing the log probabilities from the current policy being trained.
        old_log_probs: Tensor of shape (batch_size, sequence_length) 
            containing the log probabilities from the reference (old) policy.
        cliprange: The clipping parameter $\epsilon$ (e.g., 0.2) that 
            defines the interval $[1-\epsilon, 1+\epsilon]$.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]:
            - **loss**: Tensor of shape (batch_size, sequence_length) containing 
                the per-token clipped surrogate loss.
            - **metadata**: A dictionary containing training diagnostics, such as:
                - `clip_frac`: The fraction of tokens where the clipping was active.
                - `approx_kl`: Approximate KL divergence between old and new policies.
    """

    # Note here is log prob, we need to convert it back to normal prob
    importance_coef = torch.exp(policy_log_probs) / torch.exp(old_log_probs)
    clipped = torch.clamp(importance_coef, 1-cliprange, 1+cliprange)
    clipped = clipped * advantages
    unclipped = importance_coef * advantages
    final = -torch.minimum(unclipped, clipped)

    metadata = {}
    return final, metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Selects and computes the desired policy-gradient loss based on the specified type.

    Args:
        policy_log_probs: Tensor of shape (batch_size, sequence_length) containing 
            per-token log-probabilities from the policy being trained.
        loss_type: The loss formulation to use. Options are:
            - "no_baseline": Standard REINFORCE using raw rewards.
            - "reinforce_with_baseline": REINFORCE using centered/normalized advantages.
            - "grpo_clip": PPO-style clipped surrogate objective.
        raw_rewards: Required if loss_type is "no_baseline". 
            Tensor of shape (batch_size, 1).
        advantages: Required if loss_type is "reinforce_with_baseline" or "grpo_clip". 
            Tensor of shape (batch_size, 1).
        old_log_probs: Required if loss_type is "grpo_clip". 
            Tensor of shape (batch_size, sequence_length).
        cliprange: Required if loss_type is "grpo_clip". 
            The scalar epsilon ($\epsilon$) used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]:
            - **loss**: Tensor of shape (batch_size, sequence_length) representing 
                the per-token loss.
            - **metadata**: A dictionary of statistics from the underlying 
                routine (e.g., clip fraction or reward means).
    """
    if loss_type == "no_baseline":
        if raw_rewards is None:
            raise ValueError("loss_type is no_baseline but raw_rewards not provided")
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        if advantages is None:
            raise ValueError("loss_type is reinforce_with_baseline but advantages not provided")
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError("loss_type is grpo_clip but advantages or old_log_probs or cliprange not provided")
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    raise ValueError(f"unsupported loss_type {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Computes the mean of a tensor along a given dimension, considering only masked elements.

    Args:
        tensor: The data tensor to be averaged.
        mask: A tensor of the same shape as `tensor`, where positions with 1 
            are included in the mean and 0 are ignored.
        dim: The dimension over which to average. If None, the mean is 
            computed over all masked elements in the entire tensor.

    Returns:
        torch.Tensor: The masked mean. The resulting shape follows the same 
            semantics as `torch.mean(dim)`.
    """
    masked = tensor * mask
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Executes a forward-and-backward pass on a single microbatch.

    Args:
        policy_log_probs: Tensor of shape (batch_size, sequence_length) containing 
            per-token log-probabilities from the policy being trained.
        response_mask: Tensor of shape (batch_size, sequence_length) where 1 indicates 
            a response token and 0 indicates prompt or padding.
        gradient_accumulation_steps: The number of microbatches used to simulate 
            a larger batch size per optimizer step.
        loss_type: The loss formulation to utilize ("no_baseline", 
            "reinforce_with_baseline", or "grpo_clip").
        raw_rewards: Required if loss_type is "no_baseline". 
            Tensor of shape (batch_size, 1).
        advantages: Required if loss_type is not "no_baseline". 
            Tensor of shape (batch_size, 1).
        old_log_probs: Required if loss_type is "grpo_clip". 
            Tensor of shape (batch_size, sequence_length).
        cliprange: Required if loss_type is "grpo_clip". 
            The scalar epsilon ($\epsilon$) for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, Any]]:
            - **loss**: A scalar tensor representing the microbatch loss, 
                normalized by the number of gradient accumulation steps.
            - **metadata**: A dictionary containing underlying loss statistics 
                and additional logging metrics.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange
    )
    loss = masked_mean(loss, response_mask, dim=-1)
    loss = loss.mean()
    loss /= gradient_accumulation_steps
    loss.backward()

    return (loss, metadata)