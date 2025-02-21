from sklearn.model_selection import train_test_split
from tokenize_data import get_tokenized_data

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

if __name__ == "__main__":
    get_split_data("./data/codebase.txt")
