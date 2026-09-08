# Pretraining corpus

The corpus is **not redistributed** — every source file is a public RNAcentral export and
is subject to its own licence. This directory contains the code and the exact folder
layout needed to rebuild it.

## 1. Expected input layout

Download the four RNAcentral exports into one directory:

```
RNA_PRETRAIN_DATA/
├── rnacentral_active.fasta     32,700,238,140 B   all active RNAcentral sequences
├── ensembl.fasta                2,466,926,784 B   RNAcentral records with Ensembl xrefs
├── rfam.fasta                     486,261,898 B   RNAcentral records with Rfam xrefs
└── refseq.fasta                    93,969,117 B   RNAcentral records with RefSeq xrefs
```

All four use RNAcentral `URS…` accessions as headers, e.g.

```
>URS000149A9AF rRNA from 1 species
>URS0000000055_9606 Homo sapiens (human) ZNF451 regulatory antisense RNA
```

so the pooled corpus is RNAcentral throughout; the three smaller files are
database-specific slices, not separate resources. Byte sizes are the snapshots used for
the published corpus — a newer RNAcentral release will give different counts.

> **Only `*.fasta` is picked up.** `build_corpus.py` globs `*.fasta`, so a file with a
> `.fa` extension in the same directory is silently skipped. Our working directory also
> held a GENCODE 3′UTR export named `UTR3_extraction.fa`; it is **not** part of the
> corpus, and its `ENST…` headers are absent from the released FASTA. Keep the extension
> convention if you want to match our record count.

## 2. Build

```bash
python data_processing/build_corpus.py \
    --data-dir /path/to/RNA_PRETRAIN_DATA \
    --output   /path/to/rna_combined.fasta \
    --max-length 1024 \
    --seed 42
```

Per sequence, in this order:

1. uppercase
2. truncate to 1024 nt
3. drop if the (already truncated) sequence still contains `N`

then shuffle the pooled set with seed 42 and write.

**Step order matters.** Truncation precedes the `N` check, so a sequence whose only `N`
lies past position 1024 survives — its `N` was cut away. Swapping steps 2 and 3 changes
the record count.

Peak memory is roughly 60–70 GB: the pooled `(header, sequence)` list is held in RAM so
it can be shuffled globally rather than per file.

## 3. Verify

```bash
python data_processing/audit_corpus.py /path/to/rna_combined.fasta
```

Expected, for the corpus described in the paper (measured on NCI Gadi, 2026-08-14):

| | |
|---|---|
| records | **44,613,362** |
| total nucleotides | **19,869,367,674** |
| mean length | 445.4 nt |

| length bin | records | share |
|---|---|---|
| [0, 50) | 928,151 | 2.08 % |
| [50, 100) | 8,528,273 | 19.12 % |
| [100, 200) | 4,259,448 | 9.55 % |
| [200, 400) | 9,738,954 | 21.83 % |
| [400, 600) | 9,061,498 | 20.31 % |
| [600, 1024) | 3,436,369 | 7.70 % |
| [1024, 2048) | 8,660,669 | 19.41 % |
| [2048, ∞) | 0 | 0.00 % |

The `[1024, 2048)` bucket is entirely sequences that hit the truncation limit, so nothing
exceeds 1024 and the final bucket is empty by construction. Header prefixes should be
100 % `UR`.

## 4. Downstream evaluation data

Downstream benchmarks use their publishers' own splits, unmodified:

- **BEACON** — 13 tasks, the benchmark's published splits. We did not alter them.
- **Secondary structure** — RNAStrAlign for training, ArchiveII600 and TS0 for testing,
  following the standard split used in the secondary-structure literature
  (20,923 / 3,966 / 1,305 sequences).

Neither is redistributed here; obtain them from the original releases.
