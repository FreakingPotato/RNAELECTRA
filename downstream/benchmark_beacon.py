#!/usr/bin/env python3
"""
BEACON Benchmark Evaluation Script for RNA Models
Based on: BEACON: Benchmark for Comprehensive RNA Tasks and Language Models (NeurIPS 2024)

Covers the function and engineering tasks: SPL, APA, NcRNA, Modif, MRL, PRS,
CRI-On and CRI-Off. Padding is applied per batch rather than to a global maximum
length.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)

# Local modules
from task_config import (
    get_task_config,
    get_model_config,
    list_tasks,
    list_models,
    get_task_max_length,
    get_label_column,
)
from metrics import compute_metrics_fn

# Custom tokenizer
# Resolve the repository root so `model/` is importable when this script is
# run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.tokenizer import NucEL_Tokenizer


# =============================================================================
# Dataset with Dynamic Padding
# =============================================================================
class BEACONDataset(Dataset):
    """Dataset for BEACON benchmark tasks with NO padding (dynamic padding in collate_fn)."""

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        problem_type: str = "single_label_classification",
        num_labels: int = 1,
        label_encoder: dict = None,
        label_column: str = "label",
        sequence_columns: list = None,
    ):
        """
        Initialize BEACON dataset.

        Args:
            csv_path: Path to CSV file
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length for tokenization
            problem_type: Type of problem (single_label_classification, multi_label_classification, regression)
            num_labels: Number of labels/outputs
            label_encoder: Pre-built label encoder for classification
            label_column: Name of the label column in CSV
            sequence_columns: List of sequence columns for multi-sequence inputs (e.g., CRI-Off)
        """
        if isinstance(label_column, list):
            self.df = pd.read_csv(csv_path)
        elif problem_type == "token_classification":
            self.df = pd.read_csv(csv_path, dtype={label_column: str})
        else:
            self.df = pd.read_csv(csv_path)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.problem_type = problem_type
        self.num_labels = num_labels
        self.label_encoder = label_encoder
        self.label_column = label_column
        self.sequence_columns = sequence_columns


        # Build label encoder for single-label classification if not provided
        if problem_type == "single_label_classification" and label_encoder is None:
            self.unique_labels = sorted(self.df[self.label_column].unique())
            self.label_encoder = {l: i for i, l in enumerate(self.unique_labels)}
            self.num_labels = len(self.unique_labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Column naming varies across BEACON tasks.
        if 'seq' in row.index:
            seq = str(row['seq']).upper()
        elif 'sequence' in row.index:
            seq = str(row['sequence']).upper()
        elif hasattr(self, 'sequence_columns') and self.sequence_columns:
            seqs = [str(row[col]).upper() for col in self.sequence_columns]
            seq = ' '.join(seqs)
        elif 'input' in row.index:
            seq = str(row['input']).upper()
        else:
            raise ValueError("CSV must contain either 'seq', 'sequence', or 'input' column")

        # The released tokenizer uses a DNA alphabet.
        seq = seq.replace('U', 'T')
        # Tokenize WITHOUT padding (padding=False)
        encoding = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Dynamic padding
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }

        # Handle labels based on problem type
        label_value = row[self.label_column]

        if self.problem_type == "single_label_classification":
            label = self.label_encoder[label_value]
            item['labels'] = torch.tensor(label, dtype=torch.long)

        elif self.problem_type == "multi_label_classification":
            if isinstance(label_value, str):
                labels = [float(x) for x in label_value.replace(',', ' ').split()]
            else:
                labels = [float(label_value)] * self.num_labels
            item['labels'] = torch.tensor(labels, dtype=torch.float)

        elif self.problem_type == "token_classification":
            # Label is a string of digits, one per position
            if isinstance(label_value, str):
                labels = [int(c) for c in label_value]
            else:
                labels = [int(label_value)] * self.max_length

            # Truncate to match sequence length (no padding here)
            seq_len = len(seq)
            if len(labels) > seq_len:
                labels = labels[:seq_len]
            elif len(labels) < seq_len:
                # This shouldn't happen if data is correct, but handle it
                labels = labels + [-100] * (seq_len - len(labels))

            item['labels'] = torch.tensor(labels, dtype=torch.long)

        else:  # regression
            if isinstance(self.label_column, list):
                # Multiple label columns (e.g., PRS task with ON, OFF, ON_OFF)
                labels = [float(row[col]) for col in self.label_column]
            elif self.num_labels > 1:
                if isinstance(label_value, str):
                    labels = [float(x) for x in label_value.replace(',', ' ').split()]
                else:
                    labels = [float(label_value)] * self.num_labels
            else:
                labels = [float(label_value)]
            item['labels'] = torch.tensor(labels, dtype=torch.float)

        return item

    def get_num_labels(self) -> int:
        """Return the number of labels."""
        return self.num_labels


# =============================================================================
# Custom Collate Functions for Dynamic Padding
# =============================================================================
def collate_fn_sequence_level(batch):
    """
    Collate function for sequence-level tasks (classification/regression).
    Pads to maximum length in the batch.
    """
    # Find max token length in this batch
    max_token_len = max(b['input_ids'].size(0) for b in batch)

    # Pad input_ids and attention_mask
    input_ids = torch.stack([
        torch.cat([b['input_ids'],
                   torch.zeros(max_token_len - b['input_ids'].size(0), dtype=torch.long)])
        for b in batch
    ])

    attention_mask = torch.stack([
        torch.cat([b['attention_mask'],
                   torch.zeros(max_token_len - b['attention_mask'].size(0), dtype=torch.long)])
        for b in batch
    ])

    # Stack labels (no padding needed for sequence-level)
    labels = torch.stack([b['labels'] for b in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


def collate_fn_token_classification(batch):
    """
    Collate function for token classification tasks (e.g., SPL).
    Pads both inputs and labels to maximum length in the batch.
    """
    # Find max token length in this batch
    max_token_len = max(b['input_ids'].size(0) for b in batch)

    # Pad input_ids and attention_mask
    input_ids = torch.stack([
        torch.cat([b['input_ids'],
                   torch.zeros(max_token_len - b['input_ids'].size(0), dtype=torch.long)])
        for b in batch
    ])

    attention_mask = torch.stack([
        torch.cat([b['attention_mask'],
                   torch.zeros(max_token_len - b['attention_mask'].size(0), dtype=torch.long)])
        for b in batch
    ])

    # Pad labels with -100 (ignored in loss)
    labels = torch.stack([
        torch.cat([b['labels'],
                   torch.full((max_token_len - b['labels'].size(0),), -100, dtype=torch.long)])
        for b in batch
    ])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="BEACON Benchmark Evaluation (Function & Engineering Tasks)")
    parser.add_argument(
        "--task", type=str, default="mrl",
        help=f"Task name. Available: {list_tasks()}"
    )
    parser.add_argument(
        "--model", type=str, default="RNAElectra",
        help=f"Model name. Available: {list_models()}"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="Override the encoder path for --model with a local checkpoint directory "
             "or Hugging Face identifier",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data_dir", type=str, default="./beacon_data", help="Path to BEACON data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (paper default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (paper range: 1e-5 to 5e-3)")
    parser.add_argument("--max_length", type=int, default=None, help="Max sequence length (default: from task config)")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Output directory")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Warmup steps (paper default: 50)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay (paper default: 0.01)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2", help="Attention implementation to use")
    args = parser.parse_args()

    # Validate task and model
    try:
        task_cfg = get_task_config(args.task)
        model_cfg = get_model_config(args.model)
        if args.model_path:
            model_cfg = {**model_cfg, "path": args.model_path}
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Use task-specific max_length if not overridden
    max_length = args.max_length if args.max_length else get_task_max_length(args.task)
    label_column = get_label_column(args.task)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*60}")
    print(f"BEACON Benchmark Evaluation")
    print(f"{'='*60}")
    print(f"Task: {args.task} ({task_cfg['problem_type']})")
    print(f"Model: {args.model} ({model_cfg['path']})")
    print(f"Primary Metric: {task_cfg['metric']}")
    print(f"Epochs: {task_cfg['epochs']}")
    print(f"Max Length: {max_length}")
    print(f"Label Column: {label_column}")
    print(f"Seed: {args.seed}")
    print(f"Dynamic padding: enabled")
    print(f"{'='*60}\n")

    print("Loading tokenizer...")
    tokenizer = NucEL_Tokenizer(k=1, model_max_length=1024)

    # Determine num_labels
    num_labels = task_cfg['num_labels']
    sequence_columns = task_cfg.get('sequence_columns', None)

    # Load data
    data_folder = os.path.join(args.data_dir, task_cfg['folder'])
    train_path = os.path.join(data_folder, "train.csv")
    val_path = os.path.join(data_folder, "val.csv")
    test_path = os.path.join(data_folder, "test.csv")

    print(f"Loading data from {data_folder}")

    # Create datasets
    train_dataset = BEACONDataset(
        train_path,
        tokenizer,
        max_length,
        task_cfg['problem_type'],
        num_labels,
        label_column=label_column,
        sequence_columns=sequence_columns,
    )

    # Update num_labels if auto-detected
    if num_labels is None:
        num_labels = train_dataset.get_num_labels()

    label_encoder = train_dataset.label_encoder

    val_dataset = BEACONDataset(
        val_path,
        tokenizer,
        max_length,
        task_cfg['problem_type'],
        num_labels,
        label_encoder,
        label_column=label_column,
        sequence_columns=sequence_columns,
    )
    test_dataset = BEACONDataset(
        test_path,
        tokenizer,
        max_length,
        task_cfg['problem_type'],
        num_labels,
        label_encoder,
        label_column=label_column,
        sequence_columns=sequence_columns,
    )

    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val: {len(val_dataset):,}")
    print(f"  Test: {len(test_dataset):,}")
    print(f"  Num labels: {num_labels}")

    # Select appropriate collate function
    if task_cfg['problem_type'] == 'token_classification':
        collate_fn = collate_fn_token_classification
    else:
        collate_fn = collate_fn_sequence_level

    # Load model
    print(f"\nLoading model from {model_cfg['path']}...")
    if task_cfg['problem_type'] == "token_classification":
        model = AutoModelForTokenClassification.from_pretrained(
            model_cfg['path'],
            num_labels=num_labels,
            trust_remote_code=model_cfg.get('trust_remote_code', False),
            ignore_mismatched_sizes=True,
            attn_implementation=args.attn_implementation,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg['path'],
            num_labels=num_labels,
            problem_type=task_cfg['problem_type'],
            trust_remote_code=model_cfg.get('trust_remote_code', False),
            ignore_mismatched_sizes=True,
            attn_implementation=args.attn_implementation,
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("attn_implementation:", getattr(model.config, "attn_implementation", None))
    print("torch_dtype:", next(model.parameters()).dtype)
    print("device:", next(model.parameters()).device)
    print("attn_implementation:", args.attn_implementation)
    print("Model class:", model.__class__)
    print("Config class:", model.config.__class__)
    print("Config dict has attn_implementation?:", "attn_implementation" in model.config.to_dict())

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d")
    run_output_dir = os.path.join(
        args.output_dir,
        args.task,
        f"{args.model}_{timestamp}_seed{args.seed}_lr{args.lr}"
    )

    # Training arguments (aligned with BEACON paper Appendix A.1 Table 6)
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        num_train_epochs=task_cfg['epochs'],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=task_cfg['metric'],
        greater_is_better=task_cfg['greater_is_better'],
        save_total_limit=2,
        logging_dir=f"{run_output_dir}/logs",
        logging_steps=100,
        dataloader_num_workers=args.num_workers,
        max_grad_norm=1.0,
        fp16=True,
        # bf16=True,
        seed=args.seed,
        report_to="none",
    )

    # Create Trainer with dynamic padding
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn(task_cfg),
        data_collator=collate_fn,  # Use dynamic padding
    )

    # Train
    print(f"\nTraining for {task_cfg['epochs']} epochs...")
    trainer.train()

    # Evaluate on test set
    print(f"\n{'='*60}")
    print("Test Set Evaluation")
    print(f"{'='*60}")
    test_results = trainer.evaluate(test_dataset)

    for metric, value in sorted(test_results.items()):
        if metric.startswith('eval_'):
            print(f"  {metric.replace('eval_', '')}: {value:.4f}")

    # Save results
    os.makedirs(run_output_dir, exist_ok=True)

    results_file = os.path.join(run_output_dir, "results.csv")
    results_df = pd.DataFrame([{
        'model': args.model,
        'task': args.task,
        'problem_type': task_cfg['problem_type'],
        'primary_metric': task_cfg['metric'],
        'seed': args.seed,
        'epochs': task_cfg['epochs'],
        'batch_size': args.batch_size,
        'lr': args.lr,
        'max_length': max_length,
        'num_labels': num_labels,
        'dynamic_padding': True,
        **{k.replace('eval_', ''): v for k, v in test_results.items()}
    }])
    results_df.to_csv(results_file, index=False)

    # Save config
    config_file = os.path.join(run_output_dir, "config.json")
    config_data = {
        'args': vars(args),
        'task_config': task_cfg,
        'model_config': model_cfg,
        'test_results': {
            k: float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in test_results.items()
        },
    }
    if label_encoder:
        config_data['label_encoder'] = {str(k): v for k, v in label_encoder.items()}

    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_file}")
    print(f"Config saved to {config_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()