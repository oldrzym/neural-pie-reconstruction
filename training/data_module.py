"""
Data loading and preprocessing for PIE reconstruction.
"""

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional
import torch


class PIEDataset(Dataset):
    """Dataset for PIE reconstruction task."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_input_length: int = 512,
        max_target_length: int = 64,
        input_column: str = "input",
        target_column: str = "output",
    ):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.input_column = input_column
        self.target_column = target_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # Prepare input
        input_text = str(row[self.input_column])
        target_text = str(row[self.target_column])

        # Tokenize input
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Prepare labels (replace padding token id with -100 for loss calculation)
        labels = target_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels,
        }


def create_dataloaders(
    config: Dict,
    tokenizer,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """Create train, val, test dataloaders."""

    data_config = config["data"]

    train_dataset = PIEDataset(
        data_path=data_config["train_file"],
        tokenizer=tokenizer,
        max_input_length=data_config["max_input_length"],
        max_target_length=data_config["max_target_length"],
        input_column=data_config["input_column"],
        target_column=data_config["target_column"],
    )

    val_dataset = PIEDataset(
        data_path=data_config["val_file"],
        tokenizer=tokenizer,
        max_input_length=data_config["max_input_length"],
        max_target_length=data_config["max_target_length"],
        input_column=data_config["input_column"],
        target_column=data_config["target_column"],
    )

    test_dataset = PIEDataset(
        data_path=data_config["test_file"],
        tokenizer=tokenizer,
        max_input_length=data_config["max_input_length"],
        max_target_length=data_config["max_target_length"],
        input_column=data_config["input_column"],
        target_column=data_config["target_column"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
