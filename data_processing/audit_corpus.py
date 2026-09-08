#!/usr/bin/env python3
"""Verify a built corpus against the figures reported in the paper.

Streams the FASTA once and keeps only counters, so memory stays flat on a 20-30 GB file.
Prints record count, total nucleotides, mean length, an exact length histogram, and the
header-prefix breakdown (which source database each record came from).

Reference values for the corpus described in the paper (measured on NCI Gadi, 2026-08-14):

    records   44,613,362
    total nt  19,869,367,674
    mean len  445.4

    [    0,    50)      928,151   2.08 %
    [   50,   100)    8,528,273  19.12 %
    [  100,   200)    4,259,448   9.55 %
    [  200,   400)    9,738,954  21.83 %
    [  400,   600)    9,061,498  20.31 %
    [  600,  1024)    3,436,369   7.70 %
    [ 1024,  2048)    8,660,669  19.41 %      <- sequences truncated to exactly 1024
    [ 2048,   inf)            0   0.00 %

The [1024, 2048) bucket is entirely sequences that hit the truncation limit; nothing
exceeds 1024, so the last bucket is empty by construction.
"""
import collections
import gzip
import sys

BINS = [0, 50, 100, 200, 400, 600, 1024, 2048, 10 ** 9]


def opener(p):
    return gzip.open(p, "rt") if p.endswith(".gz") else open(p, "r", errors="ignore")


def audit(path):
    n = total = 0
    hist = [0] * (len(BINS) - 1)
    src = collections.Counter()
    seqlen = 0

    def bump(L):
        for i in range(len(BINS) - 1):
            if BINS[i] <= L < BINS[i + 1]:
                hist[i] += 1
                return

    with opener(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if seqlen:
                    n += 1
                    total += seqlen
                    bump(seqlen)
                    seqlen = 0
                h = line[1:].split()[0] if len(line) > 1 else ""
                key = h.split("|")[0].split(".")[0].split("_")[0][:12]
                src[key[:4] if key[:4].isalpha() else key[:2]] += 1
            else:
                seqlen += len(line.strip())
    if seqlen:
        n += 1
        total += seqlen
        bump(seqlen)

    print(f"\n=== {path}")
    print(f"  records: {n:,}   total nt: {total:,}   mean len: {total / max(n, 1):.1f}")
    for i in range(len(BINS) - 1):
        hi = "inf" if BINS[i + 1] > 10 ** 8 else str(BINS[i + 1])
        print(f"    [{BINS[i]:>5}, {hi:>5}): {hist[i]:>12,}  ({100 * hist[i] / max(n, 1):5.2f}%)")
    print(f"  header prefixes (top 8): {src.most_common(8)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: audit_corpus.py <corpus.fasta> [more.fasta ...]")
    for p in sys.argv[1:]:
        audit(p)
