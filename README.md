# RNAElectra

**Paper:** [RNAElectra: An ELECTRA-Style RNA Foundation Model for RNA Regulatory Inference][preprint] (bioRxiv, 2026)
**Model weights:** [`FreakingPotato/RNAElectra`](https://huggingface.co/FreakingPotato/RNAElectra) on Hugging Face

[preprint]: https://www.biorxiv.org/content/10.64898/2026.03.15.711950v1.full

RNAElectra is a nucleotide-resolution RNA language model pretrained with replaced-token
detection (RTD): a lightweight generator proposes plausible nucleotide substitutions and a
discriminator learns to flag which positions were replaced. Only the discriminator is
released — it is the encoder used for every downstream task.

This repository holds the pretraining code and the corpus-construction pipeline. The
processed pretraining corpus is deposited on Zenodo ([10.5281/zenodo.22761295][zenodo]);
see [`data_processing/`](data_processing/) for the source-file provenance, checksums and
the build recipe if you prefer to rebuild it from the original RNAcentral exports.

[zenodo]: https://doi.org/10.5281/zenodo.22761295

---

## Contents

```
train.py                      pretraining entry point (DeepSpeed, multi-GPU)
model/
├── tokenizer.py              single-nucleotide tokenizer (vocab 27)
├── data_collator.py          FASTA dataset + RTD collator
└── electra_trainer.py        generator/discriminator wrapper and trainer
data_processing/
├── build_corpus.py           build the pretraining FASTA from RNAcentral exports
├── audit_corpus.py           verify record counts and length distribution
└── README.md                 expected input layout, filters, reference statistics
```

## Install

```bash
pip install torch transformers deepspeed datasets biopython scikit-learn wandb tqdm
```

Pretraining was run with `torch 2.5.1+cu121`, `transformers 4.48.2`, `flash-attn
2.7.4.post1` on NVIDIA H200 and A100. FlashAttention-2 needs compute capability ≥ 8.0; on
older cards (e.g. V100, sm_70) set `attn_implementation="sdpa"` in `train.py`.

## Using the released weights

```python
import torch
from transformers import AutoModel
from tokenizer import NucEL_Tokenizer          # from the Hugging Face repo

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained(
    "FreakingPotato/RNAElectra",
    trust_remote_code=True,
).to(device).eval()

tokenizer = NucEL_Tokenizer.from_pretrained(
    "FreakingPotato/RNAElectra", trust_remote_code=True
)

inputs = tokenizer("AUGCAUGCAUGCAUGC", return_tensors="pt").to(device)
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state   # (1, L, 512)
```

## Pretraining

Build the corpus first ([`data_processing/README.md`](data_processing/README.md)), then:

```bash
deepspeed --hostfile hostfile train.py \
    --fasta_file /path/to/rna_combined.fasta \
    --kmer 1 \
    --masking_ratio 0.15 \
    --batch_size 64 \
    --epochs 30
```

`hostfile` is a DeepSpeed host file, one line per node, e.g. `gpu-node-01 slots=4`. For a
single GPU, drop `--hostfile` and run `python train.py ...`. Checkpoints and the final
model land under `PRETRAIN_MODEL/`; `--output_dir` overrides the root.

The final export contains three directories — `generator/`, `discriminator/` and
`pretrained_model/`. **`pretrained_model/` is the one to fine-tune**: it is the
discriminator's encoder with the RTD head stripped, and it is what the Hugging Face
release contains.

### Configuration as published

| | |
|---|---|
| Objective | replaced-token detection, masking ratio 0.15 |
| Generator (not released) | 12 layers, hidden 256, 8 heads, FFN 1024 |
| Discriminator (released) | 22 layers, hidden 512, 16 heads, FFN 2048 |
| Attention | **global in every layer** (`global_attn_every_n_layers=1`) |
| Position encoding | RoPE, θ = 10,000 |
| Encoder parameters | 92,311,552 |
| Tokenizer | single nucleotide, vocab 27, max token length 1025 |
| Sampling | temperature 0.9, top-k 2 |
| Discriminator loss weight | 50 |
| Train/validation split | 0.9 / 0.1 |

Global attention in every layer is a deliberate departure from the windowed local/global
pattern that ModernBERT-style DNA models inherit from NLP. RNA secondary and tertiary
structure couples nucleotides hundreds of positions apart, and the windowed pattern
severs exactly those dependencies; the paper reports the ablation and its cost.

## Citation

```bibtex
@article{rnaelectra2026,
  title   = {RNAElectra: An {ELECTRA}-Style {RNA} Foundation Model for {RNA} Regulatory Inference},
  author  = {Ding, Ke and Liu, Lixinyu and Parker, Brian and Wen, Jiayu},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.03.15.711950},
  url     = {https://www.biorxiv.org/content/10.64898/2026.03.15.711950v1.full}
}
```

## License

Apache 2.0. Source sequence data remains under the licence of the originating databases
(RNAcentral and its member resources).
