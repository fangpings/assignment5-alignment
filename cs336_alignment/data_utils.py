from datasets import load_dataset
from transformers import PreTrainedTokenizer
import torch

from torch.utils.data import Dataset

def load_gsm8k(prompt_template_path: str, split: str, answer_only: bool):
    """
    Load gsm8k dataset for train & test. Original dataset has 2 columns
    * question
    * answer: it contains solution steps & answer, separated by ####

    Args:
        prompt_template_path: Path to the prompt template file
        split: Dataset split to load, either "train" or "test"
        answer_only: If True, returns answer only; if False, returns solution with answer

    Returns
        tuple[list[str], list[str]]

        * first item is the list of formatted prompt
        * second item, if answer_only is True, then returns answer only, otherwise returns solution with answer
    """
    with open(prompt_template_path) as f:
        prompt_template = f.read()

    dataset = load_dataset("gsm8k", "main")
    dataset = dataset[split]

    prompts = []
    answers = []

    for example in dataset:
        parts = example["answer"].split("####")
        solution = parts[0].strip() if len(parts) > 0 else ""
        answer = parts[1].strip() if len(parts) > 1 else ""
        question = prompt_template.replace("{question}", example["question"])

        prompts.append(question)
        if answer_only:
            answers.append(answer)
        else:
            formatted_solution = f"{solution} </think> <answer> {answer} </answer>"
            answers.append(formatted_solution)

    return prompts, answers

def get_collate_fn_sft(tokenizer):
    def collate_fn_sft(batch):
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]

        # Tokenize the batch
        tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
        tokenized["raw_prompts"] = prompts
        tokenized["raw_responses"] = responses

        return tokenized
    return collate_fn_sft

class SftDataset(Dataset):
    def __init__(self, prompts, responses):
        self.prompts = prompts
        self.responses = responses

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # Return raw strings - tokenization happens in collate_fn
        ret = {
            "prompt": self.prompts[idx],
            "response": self.responses[idx],
        }
        return ret

def collate_fn_rl(batch):
    prompt_token_ids = [item["prompt_token_ids"] for item in batch] # (batch_size, various_len)
    response_token_ids = [item["response_token_ids"] for item in batch] # (batch_size, various_len)
    old_log_probs = [item["old_log_probs"] for item in batch] # (batch_size, various_len) (same shape to response_token_ids)
    raw_rewards = [item["raw_rewards"] for item in batch] # (batch_size, )
    advantages = [item["advantages"] for item in batch] # (batch_size, )

    ret = pad_and_assemble(
        prompt_token_ids,
        response_token_ids,
        old_log_probs=old_log_probs,
    )

    ret = ret | {
        "raw_rewards": torch.tensor(raw_rewards).unsqueeze(-1), # make it (batch_size, 1) for broadcast
        "advantages": torch.tensor(advantages).unsqueeze(-1),
    }
    return ret

class RlDataset(Dataset):
    def __init__(self, 
        prompt_token_ids, 
        response_token_ids, 
        old_log_probs,
        raw_rewards,
        advantages
    ):
        self.prompt_token_ids = prompt_token_ids
        self.response_token_ids = response_token_ids
        self.old_log_probs = old_log_probs
        self.raw_rewards = raw_rewards
        self.advantages = advantages

    def __len__(self):
        return len(self.prompt_token_ids)

    def __getitem__(self, idx):
        ret = {
            "prompt_token_ids": self.prompt_token_ids[idx],
            "response_token_ids": self.response_token_ids[idx],
            "old_log_probs": self.old_log_probs[idx],
            "raw_rewards": self.raw_rewards[idx],
            "advantages": self.advantages[idx],
        }
        return ret

def pad_and_assemble(
    prompt_input_ids: list[list[int]],
    output_input_ids: list[list[int]],
    old_log_probs: list[list[float]] = None, # used in rl
    pad_token_id: int = 0,
) -> dict[str, torch.Tensor]:
    input_ids = [a + b for a, b in zip(prompt_input_ids, output_input_ids)]
    max_length = max([len(x) for x in input_ids])
    input_ids_padded = []
    for input_id in input_ids:
        input_id_padded = input_id + [pad_token_id] * (max_length - len(input_id))
        input_ids_padded.append(input_id_padded)
    input_ids_padded = torch.tensor(input_ids_padded)
    
    response_masks = torch.zeros_like(input_ids_padded, dtype=torch.bool)
    for i in range(len(response_masks)):
        p_len = len(prompt_input_ids[i])
        o_len = len(output_input_ids[i])
        response_masks[i, p_len:p_len+o_len] = True
    
    ret = {
        "input_ids": input_ids_padded[:, :-1],
        "labels": input_ids_padded[:, 1:],
        "response_mask": response_masks[:, 1:]
    }
    
    if old_log_probs:
        pad_log_probs = torch.zeros_like(input_ids_padded, dtype=torch.float)
        for i in range(len(pad_log_probs)):
            p_len = len(prompt_input_ids[i])
            o_len = len(output_input_ids[i])
            pad_log_probs[i, p_len:p_len+o_len] = torch.tensor(old_log_probs[i])
        ret = ret | {"old_log_probs": pad_log_probs[:, :-1]}

    return ret
    

def tokenize_prompt_and_output(
    prompt_strs: list[str], 
    output_strs: list[str], 
    tokenizer: PreTrainedTokenizer,
) -> dict[str, torch.Tensor]:
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

    ret = pad_and_assemble(
        prompt_input_ids,
        output_input_ids,
        pad_token_id=tokenizer.pad_token_id
    )

    return ret