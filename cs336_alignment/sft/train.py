from cs336_alignment.sft.utils import (
    masked_normalize,
    get_response_log_probs,
    sft_microbatch_train_step
)
from cs336_alignment.utils import (
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
        default=3,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
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

    train_prompts, train_responses = load_gsm8k(args.prompt_path, split="train", answer_only=False)
    dataset = SftDataset(train_prompts, train_responses)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn_sft(tokenizer),
        num_workers=4
    )

    eval_prompts, eval_responses = load_gsm8k(args.prompt_path, split="test", answer_only=True)

    wandb.login(host=os.environ["WANDB_ENDPOINT"])
    with wandb.init(
        project="alignment-test",
        name="sft-"+str(uuid.uuid1())[:6],
        config={
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    ) as run:
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

            log_stats = {"epoch": epoch}
            for k in stats:
                log_stats["eval/"+k] = stats[k]
            run.log(log_stats)

            path = os.path.join(args.output_dir, str(epoch))
            os.makedirs(path, exist_ok=True)
            model.save_pretrained(path)
            import json
            with open(os.path.join(path, "eval.jsonl"), "w") as f:
                for output in outputs:
                    f.write(json.dumps(output) + "\n")

        # Save final model and tokenizer for vLLM
        final_model_path = os.path.join(args.output_dir, "model")
        os.makedirs(final_model_path, exist_ok=True)
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)


if __name__ == "__main__":
    main()