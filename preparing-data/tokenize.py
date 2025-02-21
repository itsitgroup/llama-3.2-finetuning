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
    tokenized_data = tokenizer(codebase, return_tensors="pt", truncation=True, padding=True)
    return tokenized_data

def get_tokenized_data(code_bas_path: str) -> dict:
    """
    Get the tokenized data.

    Args:
        tokenized_data (dict): The tokenized data.

    Returns:
        dict: The tokenized data.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3.2-3B")

    # Read your codebase from the .txt file
    with open(code_bas_path, "r", encoding="utf-8") as f:
        codebase = f.read()

    # Tokenize the codebase
    tokenized_data = tokenize_codebase(codebase, tokenizer)

    return tokenized_data

if __name__ == "__main__":

    tokenized_data = get_tokenized_data("codebase.txt")

    print(tokenized_data)
