# Import pandas
import pandas as pd
from datasets import Dataset

from pesto.train import train


def run():
    # Read CSV file
    data_path = 'preprocessedDataCopy.csv'  # Update this path to your CSV file
    data = pd.read_csv(data_path)

    # Convert data to Hugging Face's Dataset format
    dataset = Dataset.from_pandas(data)

    train(dataset)


if __name__ == '__main__':
    run()