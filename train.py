# Code adopted from the NucRL pretrain code

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from datetime import datetime
import deepspeed
import torch.distributed as dist
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    ModernBertConfig,
    ModernBertForMaskedLM,
    ModernBertModel,
    ModernBertPreTrainedModel,
    TrainingArguments,

)
import wandb
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import re
from Bio import SeqIO

# Custom imports from model package
from model.tokenizer import NucEL_Tokenizer
from model.data_collator import GenomeDataset, ElectraDataCollator
from model.electra_trainer import (
    ModernBertForTokenClassification,
    ElectraPretrainingModel,
    ElectraTrainer
)

torch._dynamo.config.capture_scalar_outputs = True


class FastaDataset(Dataset):
    """Dataset for RNA sequences with non-overlapping k-mer tokenization"""
    def __init__(self, sequences, tokenizer, max_length=1024, kmer=1):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_seq_length = max_length  # Maximum RNA sequence length
        self.kmer = kmer
        # For non-overlapping k-mers, calculate the maximum number of tokens
        if kmer > 1:
            # For non-overlapping k-mers, divide by k and add some buffer for special tokens
            self.max_token_length = (max_length // kmer)
        else:
            self.max_token_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Dynamically truncate long sequences to max_seq_length
        if len(sequence) > self.max_seq_length:
            max_start = len(sequence) - self.max_seq_length
            start_pos = random.randint(0, max_start)
            sequence = sequence[start_pos:start_pos + self.max_seq_length]
        
        # Tokenize using the tokenizer directly
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_token_length,  # Use calculated token length limit
            # padding='max_length',  # Enable padding to ensure consistent tensor sizes
            padding=False,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0)
        }


def load_fasta_data(fasta_file, max_length=1024):
    """Load RNA sequences from a FASTA file"""
    sequences = []
    long_sequences = 0
    total_sequences = 0
    invalid_sequences = 0
    
    print(f"Loading sequences from {fasta_file}")
    
    for record in tqdm(SeqIO.parse(fasta_file, "fasta"), desc="Processing FASTA file"):
        total_sequences += 1
        
        # Get sequence and convert to uppercase
        seq = str(record.seq).upper()
        
        # Convert U to T for internal representation
        seq = seq.replace("U", "T")
        
        # Replace non-standard bases with N
        seq = re.sub(r'[^ACGT]', 'N', seq)
        
        # Track long sequences
        if len(seq) > max_length:
            long_sequences += 1
        
        sequences.append(seq)
    
    print(f"Total sequences: {total_sequences}, Long sequences: {long_sequences}")
    
    return sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=259680,
                        help="Total optimisation steps. The released checkpoint was trained "
                             "for 259,680 steps at an effective batch of 512")
    parser.add_argument("--save_steps", type=int, default=8656,
                        help="Checkpoint and evaluation interval in steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--kmer", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="./PRETRAIN_MODEL", help="Output directory")
    parser.add_argument("--fasta_file", type=str, default="data/rna_combined.fasta", help="Path to FASTA file")
    parser.add_argument("--load_checkpoint", action='store_true', help="load from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint directory to resume from; required with --load_checkpoint")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--generator_size", type=int, default=256)
    parser.add_argument("--discriminator_size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k sampling")
    parser.add_argument("--weight_factor", type=float, default=50.0, help="Weight factor for discriminator loss")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--max_token_length", type=int, default=1025, help="Maximum sequence length")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--masking_ratio", type=float, default=0.15, help="Masking ratio for MLM")
    parser.add_argument("--train_split", type=float, default=0.9, help="Training split ratio")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize distributed training.
    if args.local_rank != -1:
        deepspeed.init_distributed("nccl")
        rank = dist.get_rank()
    else:
        rank = 0

    # DeepSpeed config.
    ds_config = {
        "zero_optimization": {"stage": 0},
        "train_micro_batch_size_per_gpu": args.batch_size,
    }
    
    # Training setup.
    fasta_file_name = os.path.basename(args.fasta_file).split('.')[0]
    model_name = f"NucEL_RNA_ELECTRA_K{args.kmer}_D{args.discriminator_size}_G{args.generator_size}_M{int(args.masking_ratio * 100)}_S{args.max_steps}_B{args.batch_size}_L{args.max_token_length}_{fasta_file_name}_Gadi"
    timestamp = datetime.now().strftime("%Y%m%d%H")
    output_path = f"./PRETRAIN_MODEL/NucEL_further_RNA_a100/{model_name}_K{args.kmer}_{timestamp}"
    os.makedirs(output_path, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        wandb.init(
            project="RNA_Genome_ELECTRA_ModernBERT",
            name=model_name,
            dir=output_path,
            mode="online"
        )

    # Initialize tokenizer.
    if args.kmer == 1:
        tokenizer = NucEL_Tokenizer(k=args.kmer, model_max_length=args.max_token_length)
    else:
        tokenizer = NucEL_Tokenizer(k=args.kmer, model_max_length=args.max_token_length)
    
    vocab_size = tokenizer.vocab_size

    if args.kmer == 1:
        # vocab_size += 16
        print(tokenizer.vocab_size)

    # Load sequences from FASTA file
    all_sequences = load_fasta_data(args.fasta_file, args.max_seq_length)
    
    # Split into train and validation sets
    train_sequences, val_sequences = train_test_split(
        all_sequences, 
        train_size=args.train_split, 
        random_state=args.seed,
        shuffle=True
    )
    
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

    # If debug mode is enabled, limit the dataset size
    if args.debug:
        train_sequences = train_sequences[:1000]
        val_sequences = val_sequences[:100]
        print(f"Debug mode: Limited to {len(train_sequences)} train and {len(val_sequences)} val sequences")

    # Create datasets using FastaDataset
    train_dataset = FastaDataset(
        train_sequences, 
        tokenizer, 
        max_length=args.max_seq_length, 
        kmer=args.kmer
    )
    val_dataset = FastaDataset(
        val_sequences, 
        tokenizer, 
        max_length=args.max_seq_length, 
        kmer=args.kmer
    )

    # Create data collator
    data_collator = ElectraDataCollator(tokenizer=tokenizer, mlm_probability=args.masking_ratio)    

    if not args.load_checkpoint:
        # Configure generator (smaller model)
        generator_config = ModernBertConfig(
            attn_implementation="flash_attention_2",          # "sdpa" for V100 cluster 
            vocab_size=vocab_size,
            hidden_size=args.generator_size,               
            num_hidden_layers=12,         
            num_attention_heads=8,        
            intermediate_size=int(args.generator_size*4),    
            unknown_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            mask_token_id=tokenizer.mask_token_id,
            layer_norm_eps=1e-12,
            norm_eps=1e-12,
            global_rope_theta=10000,
            global_attn_every_n_layers=1,   # global attention in every layer
            local_attention=512,            # inert while every layer is global
            local_rope_theta=1000,
            tie_word_embeddings=False,  # Disable weight tying to avoid shared tensor issues
        )

        # Configure discriminator (base model)
        discriminator_config = ModernBertConfig(
            attn_implementation="flash_attention_2",          # "sdpa" for V100 cluster 
            vocab_size=vocab_size,
            hidden_size=args.discriminator_size,              # Larger hidden size for discriminator
            num_hidden_layers=22,        
            num_attention_heads=16,       
            intermediate_size=int(args.discriminator_size*4),   
            unknown_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            cls_token_id=tokenizer.cls_token_id,
            mask_token_id=tokenizer.mask_token_id,
            layer_norm_eps=1e-12,
            norm_eps=1e-12,
            global_rope_theta=10000,
            global_attn_every_n_layers=1,   # global attention in every layer
            local_attention=512,            # inert while every layer is global
            local_rope_theta=1000,
            tie_word_embeddings=False,  # Disable weight tying to avoid shared tensor issues
        )

        generator = ModernBertForMaskedLM(generator_config)
        discriminator = ModernBertForTokenClassification(discriminator_config)
    else:
        generator = ModernBertForMaskedLM.from_pretrained(args.checkpoint_path+"generator/", attn_implementation="flash_attention_2")
        discriminator = ModernBertForTokenClassification.from_pretrained(args.checkpoint_path+"discriminator/", attn_implementation="flash_attention_2")

    # Create ELECTRA pretraining model with sampling parameters
    model = ElectraPretrainingModel(
        generator=generator, 
        discriminator=discriminator, 
        tokenizer=tokenizer,
        temperature=args.temperature,  # Use temperature from args
        top_k=args.top_k,              # Use top_k from args
        weight_factor=args.weight_factor  # Use weight_factor from args
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="steps",
        save_steps=args.save_steps,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        eval_strategy="steps",
        eval_steps=args.save_steps,
        greater_is_better=False,
        prediction_loss_only=False,
        logging_dir=f"{output_path}/logs",
        logging_strategy="steps",
        logging_steps=args.save_steps,
        dataloader_num_workers=min(8, os.cpu_count()),  # Reduced workers to avoid memory issues
        dataloader_pin_memory=True,
         # fp16=True,
        bf16=True,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=1000,
        weight_decay=3e-7,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        deepspeed=ds_config,
        local_rank=int(os.getenv("LOCAL_RANK", -1)),
        # report_to=[]
    )

    # Initialize trainer
    trainer = ElectraTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    # Start training
    trainer.train()

    # Save model and tokenizer
    if trainer.is_world_process_zero():
        model_path = os.path.join(output_path, "final_model")
        os.makedirs(model_path, exist_ok=True)
        
        # Save generator and discriminator separately using PyTorch format to avoid shared tensor issues
        generator.save_pretrained(os.path.join(model_path, "generator"), safe_serialization=False)
        discriminator.save_pretrained(os.path.join(model_path, "discriminator"), safe_serialization=False)
        
        # Extract the BERT model from the discriminator and save it
        discriminator.bert.save_pretrained(os.path.join(model_path, "pretrained_model"), safe_serialization=False)
        
        # Save tokenizer
        tokenizer.save_pretrained(model_path)
        
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()