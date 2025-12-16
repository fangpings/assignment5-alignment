from vllm import LLM, SamplingParams
from datasets import Dataset
from cs336_alignment.data_utils import load_gsm8k
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

from typing import Callable
import re
    
def evaluate_vllm(
    llm: LLM, 
    reward_fn: Callable[[str, str], dict[str, float]], 
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams 
) -> list[dict]:

    """ 
    Evaluate a language model on a list of prompts, 
    compute evaluation metrics, and serialize results to disk. 

    Return value: a dict of question, generated_answer, expected_answer, format_reward, answer_reward, reward
    """

    outputs = llm.generate(prompts, eval_sampling_params)
    rets = []
    for output, expected_answer in zip(outputs, answers):
        generated_text = output.outputs[0].text
        prompt = output.prompt
        reward = reward_fn(generated_text, expected_answer)
        ret = {
            "question": prompt,
            "generated_text": generated_text,
            "expected_answer": expected_answer
        } | reward
        rets.append(ret)
    return rets


def get_prompt_answer_pair(prompt_template: str, dataset: Dataset) -> tuple[list[str], list[str]]:
    prompts = []
    answers = []
    for record in dataset:
        prompt = prompt_template.replace("{question}", record["question"])
        prompts.append(prompt)
        answers.append(record["answer"])
    return prompts, answers

def print_reward_statistics(outputs: list[dict]):
    """
    Compute and print statistics for each reward type in the evaluation outputs.

    Args:
        outputs: List of evaluation results, each containing reward metrics
    """
    import numpy as np

    reward_types = ["format_reward", "answer_reward", "reward"]

    print("\n=== Evaluation Results ===")
    print(f"Total samples: {len(outputs)}\n")

    for reward_type in reward_types:
        values = [output[reward_type] for output in outputs]

        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        print(f"{reward_type}:")
        print(f"  Mean:   {mean_val:.4f}")
        print(f"  Median: {median_val:.4f}")
        print(f"  Std:    {std_val:.4f}")
        print(f"  Min:    {min_val:.4f}")
        print(f"  Max:    {max_val:.4f}")
        print()

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate a language model using vLLM on a set of prompts"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name or path of the model to evaluate"
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Path to file containing prompts"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save evaluation results"
    )

    args = parser.parse_args()

    default_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    llm = LLM(model=args.model_name)

    with open(args.prompt_path) as f:
        prompt_template = f.read()

    dataset = load_gsm8k()
    prompts, answers = get_prompt_answer_pair(prompt_template, dataset["test"])

    outputs = evaluate_vllm(
        llm,
        r1_zero_reward_fn,
        prompts,
        answers,
        default_sampling_params
    )

    import json
    with open(args.output_path, "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")

    # Print reward statistics
    print_reward_statistics(outputs)

if __name__ == "__main__":
    main()