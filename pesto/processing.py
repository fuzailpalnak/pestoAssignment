import os
from typing import Callable

import pandas as pd
import contractions
import re

from datasets import load_dataset
import nlpaug.augmenter.word as naw
from textgenie import TextGenie

from pesto import ROOT_DIR

SAVE_DIR = os.path.join(ROOT_DIR, os.path.join(*["data", "input", "processedData"]))
os.makedirs(SAVE_DIR)

# SYN_AUG = naw.SynonymAug(aug_src='wordnet')
# BACK_TRANSLATE_AUG = naw.BackTranslationAug(
#     from_model_name='facebook/wmt19-en-de',
#     to_model_name='facebook/wmt19-de-en'
# )
# CONTEXT_AUG = naw.ContextualWordEmbsAug(
#     model_path='roberta-base', action="substitute")
TEXTGENIE = TextGenie("ramsrigouthamg/t5_paraphraser", "bert-base-uncased")

# Load the dataset
DATASET = load_dataset("Kaludi/Customer-Support-Responses")


def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text


def normalize_text(text: str):
    """

    :param text:
    :return:
    """

    # Remove punctuation
    text = re.sub(r"[^\w\s\']", "", text)

    # Expand contractions (example)
    text = expand_contractions(text)

    # Convert to lowercase
    text = text.lower()

    return text


# def synonym(text: str):
#     """
#
#     :param text:
#     :return:
#     """
#     return normalize_text(SYN_AUG.augment(text)[0])


def paraphrase(text: str):
    return normalize_text(TEXTGENIE.augment_sent_t5(text, "paraphrase: ")[0])


# def context(text: str):
#     """
#
#     :param text:
#     :return:
#     """
#     return CONTEXT_AUG.augment(text)


# def back_translate(text: str):
#     """
#
#     :param text:
#     :return:
#     """
#     return BACK_TRANSLATE_AUG.augment(text)


def augmentation(df, technique: Callable):
    """

    :param df:
    :param technique:
    :return:
    """
    df["query"] = df["query"].apply(technique)
    df["response"] = df["response"].apply(technique)

    return df


def run_preprocessing():
    # Convert to DataFrame for exploration
    df = pd.DataFrame(DATASET["train"])

    # Apply normalization
    df["query"] = df["query"].apply(normalize_text)
    df["response"] = df["response"].apply(normalize_text)
    df_para = augmentation(df.copy(), paraphrase)
    df = pd.concat([df, df_para], ignore_index=True)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    df.to_csv(
        os.path.join(SAVE_DIR, "preprocessedData.csv"), sep="\t", encoding="utf-8"
    )
    df.to_csv(os.path.join(SAVE_DIR, "preprocessedDataCopy.csv"), index=False)

    return df


if __name__ == "__main__":
    run_preprocessing()
