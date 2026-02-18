from typing import List, Dict, Optional, Union, Any, Tuple
import os
from transformers import PreTrainedTokenizer
from itertools import product
import json

class NucEL_Tokenizer(PreTrainedTokenizer):
    """
    KMER Tokenizer for DNA sequences, inheriting from Hugging Face's PreTrainedTokenizer.
    Handles k-mer tokenization with support for special tokens, padding, and truncation.
    """
    
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        k: int = 6,
        model_max_length: int = 2048,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        num_reserved_tokens: int = 16,
        **kwargs
    ):
        """Initialize the KMER tokenizer."""
        self.k = k
        self.nucleotides = ['A', 'C', 'G', 'T']
        self.num_reserved_tokens = num_reserved_tokens
        
        # Define special tokens
        self.special_tokens = {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "mask_token": mask_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
        }
        
        # Build vocabulary (includes special tokens, nucleotides, and k-mers)
        self._init_vocabulary()
        
        # Now initialize the parent class.
        super().__init__(
            model_max_length=model_max_length,
            pad_token=pad_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )

    def _init_vocabulary(self):
        """Initialize the vocabulary with special tokens, nucleotides, and k-mers."""
        # Get special tokens in a specific order
        special_tokens = [
            self.special_tokens["pad_token"],
            self.special_tokens["unk_token"],
            self.special_tokens["cls_token"],
            self.special_tokens["sep_token"],
            self.special_tokens["mask_token"],
            self.special_tokens["bos_token"],
            self.special_tokens["eos_token"]
        ]
        
        # Add individual nucleotides
        nucleotides = self.nucleotides
        
        # Generate all possible k-mers
        kmers = [''.join(p) for p in product(self.nucleotides, repeat=self.k)]
        
        # Add reserved tokens for future use
        reserved_tokens = [f"[RESERVED_{i}]" for i in range(self.num_reserved_tokens)]

        # Combine all tokens in a specific order
        all_tokens = special_tokens + nucleotides + kmers + reserved_tokens
        
        # Create vocabulary: token -> index
        self.vocab = {}
        for idx, token in enumerate(all_tokens):
            self.vocab[token] = idx
            
        # Create reverse mapping: index -> token
        self.ids_to_tokens = {idx: token for token, idx in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Return the size of vocabulary."""
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.vocab.copy()

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize a DNA sequence into k-mers and individual nucleotides.
        
        Args:
            text: DNA sequence to tokenize
            
        Returns:
            List of tokens.
        """
        text = text.upper().strip()
        tokens = [self.cls_token]
        i = 0
        
        while i < len(text):
            # Try to get a k-mer
            if i <= len(text) - self.k:
                kmer = text[i:i+self.k]
                if kmer in self.vocab:
                    tokens.append(kmer)
                    i += self.k
                    continue
            
            # Fallback: tokenize a single nucleotide
            if i < len(text):
                nucleotide = text[i]
                if nucleotide in self.nucleotides:
                    tokens.append(nucleotide)
                else:
                    tokens.append(self.unk_token)
                i += 1
                
        return tokens

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token to its ID in the vocabulary."""
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token in the vocabulary."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """Save the tokenizer vocabulary to a directory."""
        if not filename_prefix:
            filename_prefix = "vocab"

        vocab_file = os.path.join(save_directory, f"{filename_prefix}.json")
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
            
        return (vocab_file,)

    def save_pretrained(self, save_directory: str, legacy_format: bool = True, filename_prefix: Optional[str] = None, **kwargs):
        """
        Save the tokenizer configuration and vocabulary.
        """
        # Save the vocabulary
        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)
        
        # Save the config
        config = {
            'k': self.k,
            'model_max_length': self.model_max_length,
            'padding_side': self.padding_side,
            'truncation_side': self.truncation_side,
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'sep_token': self.sep_token,
                'cls_token': self.cls_token,
                'mask_token': self.mask_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
            }
        }
        
        super().save_pretrained(save_directory, config=config, legacy_format=legacy_format, **kwargs)
        
        return vocab_files

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        """
        Load a tokenizer from a pretrained model.
        """
        from huggingface_hub import hf_hub_download
        
        # Check if it's a local path or HuggingFace repo
        if os.path.isdir(pretrained_model_name_or_path):
            # Local directory
            config_file = os.path.join(pretrained_model_name_or_path, "tokenizer_config.json")
            vocab_file = os.path.join(pretrained_model_name_or_path, "vocab.json")
        else:
            # HuggingFace Hub
            config_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="tokenizer_config.json"
            )
            vocab_file = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="vocab.json"
            )
        
        # Load config
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Load vocab
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        k = config.get('k')

        # Create tokenizer instance - tokens are at top level in tokenizer_config.json
        tokenizer = cls(
            k=k,
            model_max_length=config.get('model_max_length', 2048),
            pad_token=config.get('pad_token', '[PAD]'),
            unk_token=config.get('unk_token', '[UNK]'),
            sep_token=config.get('sep_token', '[SEP]'),
            cls_token=config.get('cls_token', '[CLS]'),
            mask_token=config.get('mask_token', '[MASK]'),
            bos_token=config.get('bos_token', '[BOS]'),
            eos_token=config.get('eos_token', '[EOS]'),
            **kwargs
        )
        
        # Override the vocabulary with the saved one
        tokenizer.vocab = vocab
        tokenizer.ids_to_tokens = {idx: token for token, idx in vocab.items()}
        
        return tokenizer