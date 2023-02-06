"""This module defines a configurable ALLDataset class."""

import torch
from pytorch_transformers import BertTokenizer
from torch.utils.data import Dataset
from loguru import logger
import csv

logger.info("Loading the tokenizer")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
yelp = None
sst = None


def rpad(array, n=256):
    current_len = len(array)
    if current_len > n:
        return array[: n - 1]
    extra = n - current_len
    return array + ([0] * extra)


class AllDataset(Dataset):

    def __init__(self, split="train"):

        logger.info(f"Loading {split} set")

        file_path = "data/yelp/"
        yelp_train = csv.reader(open(file_path + "yelp_train.csv", encoding="utf-8"))
        yelp_dev = csv.reader(open(file_path + "yelp_dev.csv", encoding="utf-8"))
        yelp_test = csv.reader(open(file_path + "yelp_test.csv", encoding="utf-8"))
        if split == "train":
            self.yelp = yelp_train
        elif split == "dev":
            self.yelp = yelp_dev
        else:
            self.yelp = yelp_test

        logger.info("Tokenizing")

        self.data = [
                (
                    rpad(
                        tokenizer.encode("[CLS] " + line[1] + " [SEP]"), n=128
                    ),
                    int(line[0]),
                )
                for line in self.yelp
            ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X, y = self.data[index]
        X = torch.tensor(X)
        return X, y


if __name__ == "__main__":

    trainset = AllDataset("train")
