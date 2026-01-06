from cs336_alignment.utils import (
    masked_normalize, 
    get_response_log_probs,
    init_vllm,
    load_policy_into_vllm_instance,
    set_seed
)
from cs336_alignment.data_utils import load_gsm8k, SftDataset, get_collate_fn_sft
from cs336_alignment.eval.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval.measure import evaluate_vllm, get_reward_statistics

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

import os
import uuid

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

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a language model using vLLM on a set of prompts"
    )
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4
    )

    set_seed()

    args = parser.parse_args()
    model_device = "cuda:0"
    vllm_device = "cuda:1"

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(model_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    llm = init_vllm(args.model_name, vllm_device)

    train_prompts, train_responses = load_gsm8k(args.prompt_path, "train")
    dataset = SftDataset(train_prompts, train_responses)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn_sft(tokenizer),
        num_workers=4
    )

    eval_prompts, eval_responses = load_gsm8k(args.prompt_path, "eval")

    wandb.login(host=os.environ["WANDB_ENDPOINT"])
    with wandb.init(project="alignment-test", name="sft-"+str(uuid.uuid1()), mode="disabled") as run:
        run.define_metric("train/*", step_metric="global_step")
        run.define_metric("eval/*", step_metric="epoch")
        global_step = 0

        for epoch in tqdm(range(args.num_epochs), desc="Overall Epochs"):
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
            for idx, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(model_device)
                labels = batch["labels"].to(model_device)
                response_mask = batch["response_mask"].to(model_device)

                log_probs_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
                log_probs = log_probs_dict["log_probs"]
                entropy = log_probs_dict["token_entropy"]

                loss, _ = sft_microbatch_train_step(log_probs, response_mask, args.gradient_accumulation_steps)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                if (idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    run.log({"train/loss": loss.item(), "train/entropy": entropy.mean().detach().item(), "global_step": global_step})
                
                global_step += 1
            
            # run rollout after every epoch
            load_policy_into_vllm_instance(model, llm)
            outputs = evaluate_vllm(
                llm,
                reward_fn=r1_zero_reward_fn,
                prompts=eval_prompts,
                answers=eval_responses
            )
            stats = get_reward_statistics(outputs)

            run.log(stats | {"epoch": epoch})

if __name__ == "__main__":
    main()