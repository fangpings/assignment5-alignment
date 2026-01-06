from transformers import PreTrainedTokenizer, PreTrainedModel
import torch
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from vllm import LLM
from unittest.mock import patch
import wandb

import os
import random
import numpy as np
from typing import Callable

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens and 0 for other tokens (prompt or padding).
    Args:

    prompt_strs: list[str] List of prompt strings. 
    output_strs: list[str] List of output strings.
    tokenizer: PreTrainedTokenizer Tokenizer to use for tokenization.

    Returns:

    dict[str, torch.Tensor]. 
    Let prompt_and_output_lens be a list containing the lengths of the tokenized prompt and output strings. Then the returned dictionary should have the following keys:
        input_ids torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            the tokenized prompt and output strings, with the final token sliced off.

        labels torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
            shifted input ids, i.e., the input ids without the first token.

        response_mask torch.Tensor of shape (batch_size, max(prompt_and_output_lens) -1): 
            a mask on the response tokens in the labels.
    """
    prompt_input_ids = tokenizer(prompt_strs)["input_ids"]
    output_input_ids = tokenizer(output_strs)["input_ids"]
    input_ids = [a + b for a, b in zip(prompt_input_ids, output_input_ids)]
    max_length = max([len(x) for x in input_ids])
    input_ids_padded = []
    for input_id in input_ids:
        input_id_padded = input_id + [tokenizer.pad_token_id] * (max_length - len(input_id))
        input_ids_padded.append(input_id_padded)
    input_ids_padded = torch.tensor(input_ids_padded)
    
    response_masks = torch.zeros_like(input_ids_padded, dtype=torch.bool)
    for i in range(len(response_masks)):
        p_len = len(prompt_input_ids[i])
        o_len = len(output_input_ids[i])
        response_masks[i, p_len:p_len+o_len] = True
        
    
    return {
        "input_ids": input_ids_padded[:, :-1],
        "labels": input_ids_padded[:, 1:],
        "response_mask": response_masks[:, 1:]
    }
    

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:

        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.

    Returns:

        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """

    # we need to calculate probs first, then calculate entropy H(x) = -\sum_i p_i * log(p_i)
    # so the problem is how to get p, but it's actually easier to get log(p)!
    # to get p, we need to perform a softmax: p_i = e^x_i / \sum_i e^x_i
    # if you do a log, it becomes log(p_i) = x_i - log(\sum_i e^x_i)
    # and torch actually provides a function logsumexp that does exactly what we want (and handles overflow)
    # once we have the log prob, to get the original prob we just need to exp it
    #
    # log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    #
    # originally i was thinking the method above, but later i found out there is a builtin function to do this
    log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
    prob = torch.exp(log_prob)

    entropy = -torch.sum(log_prob * prob, dim=-1)

    return entropy.detach()

def get_response_log_probs(
    model: PreTrainedModel, 
    input_ids: torch.Tensor, 
    labels: torch.Tensor, 
    return_token_entropy: bool = False, 
) -> dict[str, torch.Tensor]:
    """
    Args:

        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt + response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling compute_entropy.

    Returns:

        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log p θ (x t | x <t ).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy for each position (present only if return_token_entropy=True).
    """
    # NOTE: this function returns the log prob of the token that label represents
    # which is EXACTLY the cross entropy loss (without minus sign)
    # because cross entropy loss is just \sum_i^c y_i * log(p(x_i)), where y_i is the one hot on i-th element in the vocab and x_i is the logit on i-th element
    # this is essentially just log(p(x_{y}))
    # note if the model is loaded using AutoModelForCausalLM, it will give you CausalLMOutputWithPast, which contains logits in shape (batch_size, sequence_length, vocab_size) and loss(optional)
    # on the other hand, if the model is loaded using AutoModel, it will give you BaseModelOutputWithPast, which contains last_hidden_state
    logits = model(input_ids).logits
    
    # again, log softmax prob is equal to x - logsumexp(x)
    log_probs_all = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # here is the tricky part, basically what we need to do is
    #   I have a tensor a of shape b,s,v and another tensor b of shape b,s. the element of tensor b at position i,j represents the index of tensor a on v dimension. 
    #   now I want to do a select on a using b, so that the resulted tensor c has shape b,s and c[i,j]=a[i, j, b[i,j]]
    # what we should use is torch.gather: Gathers values along an axis specified by dim
    # it works as follow, and out has same shape of index
    #    out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    # what we need to do here is first extend tensor b to 3d by unsqueeze at last dim, now it becomes shape b,s,1
    # by the rule of gather, k will always be 0, so it becomes out[i,j,0] = input[i, j, index[i, j, 0]] and shape of b,s,1
    # we can finally do a squeeze to get what we want
    log_probs = torch.gather(log_probs_all, index=labels.unsqueeze(-1), dim=-1).squeeze(-1)

    ret = {"log_probs": log_probs}

    # the interesting thing here is that, the token entropy we generate here at index i is all based
    # on the ground truth (at least in sft, TODO change this comment if rl stage is different)
    # so it's basically calculating the conditional entropy based on ground truth.
    # the goal here is to measure the entropy given CORRECT history
    if return_token_entropy:
        ret["token_entropy"] = compute_entropy(logits)

    return ret

def masked_normalize(
    tensor: torch.Tensor, 
    mask: torch.Tensor, 
    normalize_constant: float, 
    dim: int | None = None, 
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those elements where mask == 1.

    Args:

        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float the constant to divide by for normalization.
        dim: int | None the dimension to sum along before normalization. If None, sum over all dimensions.

    Returns:

        torch.Tensor the normalized sum, where masked elements (mask == 0) don’t contribute to the sum.
    """

    # tensor[~mask] = 0
    # mutation is not good, better create a new one
    # interesting we can just multiply the mask
    tensor = tensor * mask
    tensor /= normalize_constant
    return tensor.sum(dim)


def init_vllm(model_id: str, device: str, gpu_memory_utilization: float = 0.85):
    """ 
    Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy.
    """

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py 
    # Patch vLLM to make sure we can 
    # (1) place the vLLM model on the desired device (world_size_patch) and 
    # (2) avoid a test that is not designed for our setting (profiling_patch).

    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)

    profiling_patch = patch( "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id, 
            device=device, 
            dtype=torch.bfloat16, 
            enable_prefix_caching=True, 
            gpu_memory_utilization=gpu_memory_utilization, 
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """ 
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """ 
    state_dict = policy.state_dict() 
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model 
    llm_model.load_weights(state_dict.items())

def set_seed(seed=42):
    """
    Sets the seed for reproducibility across python, numpy, and pytorch.
    """
    # 1. Base Python randomness
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Numpy randomness
    np.random.seed(seed)
    
    # 3. PyTorch randomness
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # 4. CuDNN backend (Crucial for GPU reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    vllm_set_random_seed(seed)

def log_generations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_prompt: list[str],
    ground_truth: list[str],
    reward_fn: Callable[[str, str], dict[str, float]], 
):
    """
    This function logs the following items
    1. The input prompt.
    2. The response generated by the SFT/RL model.
    3. The ground-truth answer.
    4. The reward information, including format, answer, and total reward.
    5. The average token entropy of the response.
    6. The average response length, average response length for correct responses, and average response length for incorrect responses.
    """

    # it's really hard to get the correct average entropy
    # the problem is that we use vllm for rollout, and vllm does not save raw logits
    # the only thing vllm provides is the top-k log probs
    # we are going to use this for entropy approximation
