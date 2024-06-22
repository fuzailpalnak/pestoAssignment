# Import pandas
import pandas as pd
from datasets import Dataset

from pesto.processing import run_preprocessing
from pesto.train import train


def run_from_pre_processed_file():
    # Read CSV file
    data_path = 'preprocessedDataCopy.csv'  # Update this path to your CSV file
    data = pd.read_csv(data_path)

    # Convert data to Hugging Face's Dataset format
    dataset = Dataset.from_pandas(data)

    train(dataset)


def run_from_raw_data():
    # Convert data to Hugging Face's Dataset format
    dataset = Dataset.from_pandas(run_preprocessing())

    train(dataset)
