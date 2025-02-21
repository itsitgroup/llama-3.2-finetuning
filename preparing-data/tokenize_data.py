from transformers import AutoTokenizer

def tokenize_codebase(codebase: str, tokenizer: AutoTokenizer) -> dict:
    """
    Tokenize the codebase using the provided tokenizer.
    Args:
        codebase (str): The codebase to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.
    Returns:
        dict: The tokenized data.
    """
    tokenized_data = tokenizer(
        codebase,
        return_tensors="pt",  # Return PyTorch tensors
        truncation=True,      # Truncate to model's max length
        padding=True          # Pad to the longest sequence
    )
    return tokenized_data


def get_tokenized_data(code_base_path: str) -> dict:
    """
    Get the tokenized data for a given codebase file.
    Args:
        code_base_path (str): Path to the .txt file containing the codebase.
    Returns:
        dict: The tokenized data.
    """
    # Load the tokenizer for the LLaMA model
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3.2-3B")

    # Read the codebase from the .txt file
    with open(code_base_path, "r", encoding="utf-8") as f:
        codebase = f.read()

    # Tokenize the codebase
    tokenized_data = tokenize_codebase(codebase, tokenizer)
    return tokenized_data


if __name__ == "__main__":
    # Path to your codebase file
    tokenized_data = get_tokenized_data("codebase.txt")
    print(tokenized_data)