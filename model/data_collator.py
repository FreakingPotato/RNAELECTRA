import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random
import re
from .tokenizer import NucEL_Tokenizer  # Relative import within model package

class GenomeDataset(Dataset):
    def __init__(self, dataset, tokenizer, sequence_length=6000, token_length=2048):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_length = token_length
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        sequence = self.dataset[idx]['sequence']
        
        default_length = self.sequence_length + 200

        # Handle full-length sequences (6200bp or longer)
        if len(sequence) >= default_length:
            # Calculate number of complete segments
            num_segments = default_length // self.sequence_length
            
            # Randomly select a segment index (0 to num_segments-1)
            segment_idx = random.randint(0, num_segments - 1)
            
            # Calculate base starting position
            base_start_pos = segment_idx * self.sequence_length
            
            # Add a small random shift (0-4 bp)
            small_shift = random.randint(0, 4)
            
            # Calculate final starting position with shift
            start_pos = min(base_start_pos + small_shift, default_length - self.sequence_length)
            
            # Calculate end position
            end_pos = start_pos + self.sequence_length
            
            # Extract the sequence fragment
            sequence = sequence[start_pos:end_pos]
        else:
            # Fallback for shorter sequences
            sequence = sequence[:self.sequence_length]
        
        # Replace N with A and convert to uppercase
        sequence = sequence.upper().replace('N', 'A')
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.token_length,
            padding="max_length",
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

class RNAGenomeDataset(Dataset):
    """Dataset for RNA sequences from HuggingFace datasets"""
    def __init__(self, dataset, tokenizer, sequence_length=8192, token_length=8192):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.token_length = token_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get sequence from the dataset
        sequence = self.dataset[idx]['sequence']
        
        # Ensure uppercase and already in ATCG format (no U->T conversion needed)
        sequence = sequence.upper()
        
        # Replace any non-standard bases with N, then replace N with A
        sequence = re.sub(r'[^ACGT]', 'N', sequence)
        sequence = sequence.replace('N', 'A')
        
        # Handle sequence length
        if len(sequence) > self.sequence_length:
            # Randomly sample a segment of the desired length
            max_start = len(sequence) - self.sequence_length
            start_pos = random.randint(0, max_start)
            sequence = sequence[start_pos:start_pos + self.sequence_length]
        # If sequence is shorter, it will be padded by the tokenizer
        
        # Tokenize sequence
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.token_length,
            # padding="max_length",
            padding="do_not_pad",
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# # --- ElectraDataCollator with RLMTokenizer ---
# class ElectraDataCollator:
#     def __init__(self, tokenizer, mlm_probability=0.15):
#         self.tokenizer = tokenizer
#         self.mlm_probability = mlm_probability
#         self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
    
#     def __call__(self, features):
#         batch = {}
#         # Stack each feature into a batch tensor
#         for key in features[0].keys():
#             batch[key] = torch.stack([f[key] for f in features])
        
#         # Create labels: the original token IDs before any masking
#         labels = batch["input_ids"].clone()

#         # Determine which tokens to mask: 15% probability
#         probability_matrix = torch.full(batch["input_ids"].shape, self.mlm_probability)
        
#         # Don't mask special tokens
#         special_tokens_mask = [
#             self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
#             for val in batch["input_ids"].tolist()
#         ]
#         probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
#         # Also don't mask padding tokens
#         padding_mask = batch["attention_mask"].bool()
#         probability_matrix.masked_fill_(~padding_mask, value=0.0)
        
#         masked_indices = torch.bernoulli(probability_matrix).bool()

#         # Replace selected tokens with the mask token id
#         masked_input = batch["input_ids"].clone()
#         masked_input[masked_indices] = self.mask_token_id

#         # Add masked input, the original tokens, and mask indicator to the batch
#         batch["masked_input_ids"] = masked_input
#         batch["labels"] = labels
#         batch["masked_indices"] = masked_indices
        
#         return batch

class ElectraDataCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def __call__(self, features):
        # Debug: Check what keys are available
        if not features or "input_ids" not in features[0]:
            print(f"ERROR: Features keys: {features[0].keys() if features else 'empty'}")
            raise ValueError(f"Expected 'input_ids' in features, got: {list(features[0].keys())}")
        
        # Find the maximum sequence length in this batch
        max_length = max(f["input_ids"].size(0) for f in features)
        
        # Pad each feature to max_length
        batch = {"input_ids": [], "attention_mask": []}
        
        for f in features:
            seq_len = f["input_ids"].size(0)
            padding_length = max_length - seq_len
            
            # Pad input_ids with pad_token_id
            padded_input_ids = torch.cat([
                f["input_ids"],
                torch.full((padding_length,), self.pad_token_id, dtype=torch.long)
            ])
            
            # Pad attention_mask with 0s
            padded_attention_mask = torch.cat([
                f["attention_mask"],
                torch.zeros(padding_length, dtype=torch.long)
            ])
            
            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(padded_attention_mask)
        
        # Stack into batch tensors
        batch["input_ids"] = torch.stack(batch["input_ids"])
        batch["attention_mask"] = torch.stack(batch["attention_mask"])
        
        # Create labels: the original token IDs before any masking
        labels = batch["input_ids"].clone()
        
        # Determine which tokens to mask: 15% probability
        probability_matrix = torch.full(batch["input_ids"].shape, self.mlm_probability)
        
        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in batch["input_ids"].tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        # Don't mask padding tokens
        padding_mask = batch["attention_mask"].bool()
        probability_matrix.masked_fill_(~padding_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Replace selected tokens with the mask token id
        masked_input = batch["input_ids"].clone()
        masked_input[masked_indices] = self.mask_token_id
        
        # Add masked input, the original tokens, and mask indicator to the batch
        batch["masked_input_ids"] = masked_input
        batch["labels"] = labels
        batch["masked_indices"] = masked_indices
        
        return batch