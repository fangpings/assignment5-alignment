import torch
from typing import Callable, Any, Literal
from dataclasses import dataclass
import argparse

@dataclass
class GRPOConfig:
    """
    Configuration parameters for Group Relative Policy Optimization (GRPO) training.
    """
    # Model and data paths
    model_name: str = "Qwen/Qwen2.5-Math-1.5B"
    prompt_path: str = "cs336_alignment/prompts/r1_zero.prompt"
    output_dir: str = "outputs"

    # Training Loop
    # Total number of rollout-train cycles. This is the outer loop of GRPO
    n_grpo_steps: int = 200
    eval_steps: int = 10
    learning_rate: float = 1e-5
    # How many times to train on the same rollout data before collecting fresh samples.
    # 1 = on-policy: Generate data → train once → discard → repeat
    # >1 = off-policy: Generate data → train multiple epochs → discard → repeat
    epochs_per_rollout_batch: int = 1  # On-policy training

    # Reward & Advantage
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    advantage_eps: float = 1e-6
    use_std_normalization: bool = True
    clip_range: float = 0.2

    # Rollout & Sampling
    # Number of samples generated per GRPO step. With group_size=8, this means 256÷8 = 32 unique prompts, each getting 8 responses.
    rollout_batch_size: int = 256
    group_size: int = 8  # Number of responses per prompt
    sampling_temperature: float = 1.0
    sampling_min_tokens: int = 4  # Disallow empty/near-empty responses
    sampling_max_tokens: int = 1024

    # Hardware & Optimization
    # Batch size for gradient updates during the training phase. Can be same as rollout_batch_size or different depending on memory constraints.
    train_batch_size: int = 256  # On-policy: usually matches rollout_batch_size
    gradient_accumulation_steps: int = 128  # Microbatch size = 2; tuned for H100 memory

    def __post_init__(self):
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, (
            "train_batch_size must be divisible by gradient_accumulation_steps"
        )
        self.micro_train_batch_size = self.train_batch_size // self.gradient_accumulation_steps

        assert self.rollout_batch_size % self.group_size == 0, (
            "rollout_batch_size must be divisible by group_size"
        )
        self.n_prompts_per_rollout_batch = self.rollout_batch_size // self.group_size

        assert self.train_batch_size >= self.group_size, (
            "train_batch_size must be greater than or equal to group_size"
        )
        self.n_microbatches_per_rollout_batch = self.rollout_batch_size // self.micro_train_batch_size

def parse_grpo_args() -> GRPOConfig:
    """
    Parse command-line arguments and return a GRPOConfig instance.

    Returns:
        GRPOConfig: Configuration object with all training parameters.
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using Group Relative Policy Optimization (GRPO)"
    )
    # Model and data paths
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="cs336_alignment/prompts/r1_zero.prompt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs"
    )

    # Training Loop
    parser.add_argument(
        "--n_grpo_steps",
        type=int,
        default=200
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--epochs_per_rollout_batch",
        type=int,
        default=1
    )

    # Reward & Advantage
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["no_baseline", "reinforce_with_baseline", "grpo_clip"],
        default="reinforce_with_baseline"
    )
    parser.add_argument(
        "--advantage_eps",
        type=float,
        default=1e-6
    )
    parser.add_argument(
        "--use_std_normalization",
        type=lambda x: x.lower() == 'true',
        default=True
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2
    )

    # Rollout & Sampling
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--sampling_min_tokens",
        type=int,
        default=4
    )
    parser.add_argument(
        "--sampling_max_tokens",
        type=int,
        default=1024
    )

    # Hardware & Optimization
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=128
    )

    args = parser.parse_args()

    # Create GRPOConfig from parsed args
    return GRPOConfig(
        model_name=args.model_name,
        prompt_path=args.prompt_path,
        output_dir=args.output_dir,
        n_grpo_steps=args.n_grpo_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        loss_type=args.loss_type,
        advantage_eps=args.advantage_eps,
        use_std_normalization=args.use_std_normalization,
        clip_range=args.clip_range,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

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