from datasets import load_dataset, DatasetDict

from cs336_alignment.utils import tokenize_prompt_and_output
from torch.utils.data import Dataset

def load_gsm8k(prompt_template_path: str, mode: str):
    """
    Load gsm8k dataset for train & test. Original dataset has 2 columns
    * question
    * answer: it contains solution steps & answer, separated by ####

    Returns
        tuple[list[str], list[str]]

        * first item is the list of formatted prompt
        * second item, if mode is eval, then returns answer only, otherwise returns solution with answer
    """
    with open(prompt_template_path) as f:
        prompt_template = f.read()
    
    dataset = load_dataset("gsm8k", "main")
    if mode == "eval":
        dataset = dataset["test"]
    else:
        dataset = dataset["train"]

    prompts = []
    answers = []

    for example in dataset:
        parts = example["answer"].split("####")
        solution = parts[0].strip() if len(parts) > 0 else ""
        answer = parts[1].strip() if len(parts) > 1 else ""
        question = prompt_template.replace("{question}", example["question"])
    
        prompts.append(question)
        if mode != "eval":
            formatted_solution = f"{solution} </think> <answer> {answer} </answer>"
            answers.append(formatted_solution)
        else:
            answers.append(answer)
    
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
        return {
            "prompt": self.prompts[idx],
            "response": self.responses[idx]
        }