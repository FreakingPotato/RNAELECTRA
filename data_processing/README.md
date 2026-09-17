# Pretraining corpus

The processed corpus is deposited on Zenodo
([10.5281/zenodo.22761295](https://doi.org/10.5281/zenodo.22761295)). This directory
contains the code, the source-file provenance and the exact folder layout needed to
rebuild it from the original RNAcentral exports.

## 1. Expected input layout

Download the four RNAcentral exports into one directory:

```
RNA_PRETRAIN_DATA/
├── rnacentral_active.fasta     all active RNAcentral sequences
├── ensembl.fasta               RNAcentral records with Ensembl cross-references
├── rfam.fasta                  RNAcentral records with Rfam cross-references
└── refseq.fasta                RNAcentral records with RefSeq cross-references
```

All four use RNAcentral `URS…` accessions as headers, e.g.

```
>URS000149A9AF rRNA from 1 species
>URS0000000055_9606 Homo sapiens (human) ZNF451 regulatory antisense RNA
```

so the pooled corpus is RNAcentral throughout; the three smaller files are
database-specific slices, not separate resources.

### Source snapshot

The published corpus was built from **two** RNAcentral releases: the whole-database export
comes from release 25.0, the three cross-referenced subsets from release 24.0. Rebuilding
from a single release will not reproduce the record counts below.

| File | RNAcentral release | Bytes | Records |
|---|---|---|---|
| `rnacentral_active.fasta` | [25.0](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/25.0/), released 2025-05-14 | 32,700,238,140 | 40,712,942 |
| `ensembl.fasta` | [24.0](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/24.0/), released 2024-03-07 | 2,466,926,784 | 2,265,715 |
| `rfam.fasta` | [24.0](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/24.0/), released 2024-03-07 | 486,261,898 | 2,218,572 |
| `refseq.fasta` | [24.0](https://ftp.ebi.ac.uk/pub/databases/RNAcentral/releases/24.0/), released 2024-03-07 | 93,969,117 | 119,440 |

In the release tree, `rnacentral_active.fasta` is `sequences/rnacentral_active.fasta.gz`
and the three subsets are uncompressed under `sequences/by-database/`.

Save the following as `SHA256SUMS` beside the four files and check them with
`sha256sum -c SHA256SUMS`:

```
950e8d71611c8a730d4f4a2c85af5f6c125d3eae657e92ab436c6e382ca99121  rnacentral_active.fasta
bb9c1cdeef6889d1b8c5c0d54ce9602c14f4538099efbffb06ab92fda185091e  ensembl.fasta
fb386d9047c7847287e99a56a4616add1942e236344d88aee6237fa763eeccb4  rfam.fasta
b782724f47501b228cb9c7729d8d76d8ea717e4f563c5ba5dab5964eae31a543  refseq.fasta
```

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

The four inputs hold 45,316,669 records between them; 703,307 of those (1.55 %) still
contain `N` after truncation and are dropped, leaving exactly the 44,613,362 records
above. Checksums of the build product:

```
f36b80bd3ca3ce86a21bee099a38fd4de87e05b387b162f6aeacbb34e783e700  rna_combined.fasta
3f715c332083dd90c0671444b3a357f1e2e9cb60050070c14413e71380786889  rna_combined.fasta.gz
```

`rna_combined.fasta.gz` is the archive published on Zenodo, reassembled from its 12 parts.

## 4. Downstream evaluation data

Downstream benchmarks use their publishers' own splits, unmodified:

- **BEACON** — 13 tasks, the benchmark's published splits. We did not alter them.
- **Secondary structure** — RNAStrAlign for training, ArchiveII600 and TS0 for testing,
  following the standard split used in the secondary-structure literature
  (20,923 / 3,966 / 1,305 sequences).

Neither is redistributed here; obtain them from the original releases.
