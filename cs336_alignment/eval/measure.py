from vllm import LLM, SamplingParams
from datasets import Dataset
from cs336_alignment.data_utils import load_gsm8k
from cs336_alignment.eval.drgrpo_grader import r1_zero_reward_fn

from typing import Callable
import re

default_sampling_params = SamplingParams(
    temperature=1.0, 
    top_p=1.0, 
    max_tokens=1024, 
    stop=["</answer>"], 
    include_stop_str_in_output=True, 
    logprobs=20 # set to 20 to approximate token entropy
)
    
def evaluate_vllm(
    llm: LLM, 
    reward_fn: Callable[[str, str], dict[str, float]], 
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams = default_sampling_params
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

def get_reward_statistics(outputs: list[dict]) -> dict[str, dict[str, float]]:
    """
    Compute and print statistics for each reward type in the evaluation outputs.

    Args:
        outputs: List of evaluation results, each containing reward metrics

    Return:
        statistics for each reward type
    """
    import numpy as np

    reward_types = ["format_reward", "answer_reward", "reward"]

    ret = {}
    for reward_type in reward_types:
        values = [output[reward_type] for output in outputs]
        ret[reward_type] = np.mean(values)
        
    return ret

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

    prompts, answers = load_gsm8k(args.prompt_path, "eval")

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
    stats = get_reward_statistics(outputs)
    print("\n=== Evaluation Results ===")
    print(f"Total samples: {len(outputs)}\n")
    for reward_type in stats:
        print(f"{reward_type}:")
        print(f"  Mean:   {stats[reward_type]:.4f}")
        print()

"""
command
uv run python -m cs336_alignment.eval.measure --model_name Qwen/Qwen2-Math-1.5B --prompt_path cs336_alignment/prompts/r1_zero.prompt --output_path result/gsm8k_qwen_baseline.json
"""
if __name__ == "__main__":
    main()