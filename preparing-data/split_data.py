from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
from tokenize_data import get_tokenized_data
import json

def get_split_data(data_path: str, test_size: float = 0.1) -> tuple:
    """
    Split the tokenized data into training and validation sets.
    Args:
        test_size (float): The proportion of the dataset to include in the validation set.
    Returns:
        tuple: A tuple containing the training and validation sets.
    """
    # Convert tokenized data to a list of token IDs
    tokenized_data = get_tokenized_data(data_path)
    input_ids = tokenized_data["input_ids"][0].tolist()

    # Split the data into training and validation sets
    train_ids, val_ids = train_test_split(input_ids, test_size=test_size, random_state=42)

    # Print the sizes of the splits
    print(f"Training samples: {len(train_ids)}")
    print(f"Validation samples: {len(val_ids)}")
    return  train_ids, val_ids

def save_split_data(train_ids: list, val_ids: list):
    """
    Save the split data to JSON files.
    Args:
        data_path (str): Path to the original codebase file.
        train_ids (list): List of token IDs for the training set.
        val_ids (list): List of token IDs for the validation set.
    """
    # Prepare the dataset in a dictionary format
    dataset = {
        "train": {"input_ids": train_ids},
        "validation": {"input_ids": val_ids},
    }

    # Save the dataset as a JSON file
    with open("codebase_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/LLaMA-3.2-3B")

    # Save training data as plain text
    with open("train.txt", "w", encoding="utf-8") as f:
        for ids in train_ids:
            f.write(tokenizer.decode(ids) + "\n")

    # Save validation data as plain text
    with open("val.txt", "w", encoding="utf-8") as f:
        for ids in val_ids:
            f.write(tokenizer.decode(ids) + "\n")

def split_and_save_data(data_path: str, test_size: float = 0.1):
    """
    Split the tokenized data and save the resulting datasets.
    Args:
        data_path (str): Path to the original codebase file.
        test_size (float): The proportion of the dataset to include in the validation set.
    """
    train_ids, val_ids = get_split_data(data_path, test_size)
    save_split_data(data_path, train_ids, val_ids)

if __name__ == "__main__":
    split_and_save_data("./data/codebase.txt")
