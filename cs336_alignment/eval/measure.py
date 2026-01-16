try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
from datasets import Dataset
from cs336_alignment.data_utils import load_gsm8k
from cs336_alignment.eval.drgrpo_grader import r1_zero_reward_fn

from typing import Callable
import re
import numpy as np

if VLLM_AVAILABLE:
    default_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
        logprobs=20 # set to 20 to approximate token entropy
    )
else:
    default_sampling_params = None

def calculate_avg_token_entropy(logprobs) -> float:
    """
    Calculate average token entropy from logprobs.

    Args:
        logprobs: List of token logprobs from vllm output

    Returns:
        Average entropy across all tokens in bits
    """
    token_entropies = []
    if logprobs:
        for token_logprobs in logprobs:
            if token_logprobs:
                # Calculate entropy: -sum(p * log(p))
                logprob_values = np.array([lp.logprob for lp in token_logprobs.values()])
                probs = np.exp(logprob_values)
                entropy = -np.sum(probs * logprob_values)
                token_entropies.append(entropy)

    return np.mean(token_entropies) if token_entropies else 0.0

def evaluate_vllm(
    llm,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params = default_sampling_params
) -> list[dict]:

    """ 
    Evaluate a language model on a list of prompts, 
    compute evaluation metrics, and serialize results to disk. 

    Return value: a dict of question, generated_answer, expected_answer, format_reward, answer_reward, reward
    """

    # For reference, output struct: https://github.com/vllm-project/vllm/blob/8ef50d9a6b91b7800e69a846219069a29a0298a4/vllm/outputs.py#L86
    outputs = llm.generate(prompts, eval_sampling_params)
    rets = []
    for output, expected_answer in zip(outputs, answers):
        generated_text = output.outputs[0].text
        prompt = output.prompt
        reward = reward_fn(generated_text, expected_answer)

        # Calculate average token entropy from logprobs
        avg_entropy = calculate_avg_token_entropy(output.outputs[0].logprobs)

        ret = {
            "question": prompt,
            "generated_text": generated_text,
            "expected_answer": expected_answer,
            "avg_token_entropy": avg_entropy
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

    reward_types = ["format_reward", "answer_reward", "reward", "avg_token_entropy"]

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

    if not VLLM_AVAILABLE:
        raise ImportError("vllm is not available. This script requires vllm to run.")

    default_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    llm = LLM(model=args.model_name)

    prompts, answers = load_gsm8k(args.prompt_path, split="test", answer_only=True)

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