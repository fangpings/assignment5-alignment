from datasets import load_dataset, DatasetDict

def load_gsm8k() -> DatasetDict:
    """
    Load gsm8k dataset for train & test. Original dataset has 2 columns
    * question
    * answer: it contains solution steps & answer, separated by ####

    Returns dataset with 3 columns:
    * question
    * solution
    * answer
    """
    dataset = load_dataset("gsm8k", "socratic")

    def split_answer(example):
        parts = example["answer"].split("####")
        return {
            "question": example["question"],
            "solution": parts[0].strip() if len(parts) > 0 else "",
            "answer": parts[1].strip() if len(parts) > 1 else ""
        }

    dataset = dataset.map(split_answer, remove_columns=["answer"])

    return dataset
