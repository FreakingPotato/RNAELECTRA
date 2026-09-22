#!/usr/bin/env bash
# Find every pretraining-corpus record homologous to a downstream sequence.
#
# The corpus is the QUERY and the downstream set the TARGET. This matters:
# easy-search indexes the target, and indexing a 21 GB corpus exhausts memory.
#
# --search-type 3   nucleotide-nucleotide; both strands are aligned and the better
#                   orientation is reported, so no separate reverse-complement pass
# --cov-mode 1      coverage is measured against the target, i.e. the downstream sequence
# -c 0.8            the alignment must cover >= 80 % of that downstream sequence
# --min-seq-id 0.8  and reach >= 80 % identity
# no -e             no E-value cut-off; the filter is identity and coverage alone
set -euo pipefail

CORPUS=${1:?usage: run_homology_audit.sh <corpus.fasta> <downstream.fasta> <out.m8> [threads]}
DOWNSTREAM=${2:?}
OUT=${3:?}
THREADS=${4:-48}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

mmseqs easy-search "$CORPUS" "$DOWNSTREAM" "$OUT" "$TMP" \
    --search-type 3 \
    --min-seq-id 0.8 \
    -c 0.8 \
    --cov-mode 1 \
    -s 5.7 \
    --max-seqs 50 \
    --threads "$THREADS" \
    --format-output query,target,fident,alnlen,qcov,tcov,evalue,bits

echo "wrote $OUT ($(wc -l < "$OUT") alignments)"
