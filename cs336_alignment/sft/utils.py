from transformers import PreTrainedModel
import torch

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

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor, 
    gradient_accumulation_steps: int, 
    normalize_constant: float = 1.0, 
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.

    Args:

        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the SFT policy being trained.
        response_mask (batch_size, sequence_length), 1 for response tokens, 0 for prompt/padding.
        gradient_accumulation_steps Number of microbatches per optimizer step. 
        normalize_constant The constant by which to divide the sum. It is fine to leave this as 1.0.

    Returns:

        tuple[torch.Tensor, dict[str, torch.Tensor]].
            loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
            metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.
    """

    # so here policy_log_probs should be the output from get_response_log_probs
    # which is just the cross entropy loss (without a minus sign)
    # 
    # QUESTION: I cannot understand the right answer here. It's basically summing within a training sample
    # then taking average across batch. why is it correct? if it is correct then longer samples will have more contribution to the loss?
    loss = -masked_normalize(policy_log_probs, response_mask, dim=-1, normalize_constant=normalize_constant)
    loss = loss.mean()

    # gradient accumulation requires you to 1. divide the gradient by accumulation steps 2. do loss.backward() every step
    # 3. run optimizer.step() every accumulation steps
    loss /= gradient_accumulation_steps
    loss.backward()

    return (loss.detach(), {})