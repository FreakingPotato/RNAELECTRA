#!/usr/bin/env python3
"""Build the RNAElectra pretraining corpus from RNAcentral FASTA exports.

Reproduces the corpus described in the paper: 44,613,362 sequences / 19,869,367,674 nt.

Pipeline, per sequence:
  1. uppercase
  2. truncate to --max-length nt
  3. drop the sequence if it still contains 'N'
  4. shuffle the pooled set with --seed, then write

Two details matter for exact reproduction and are deliberately preserved from the
original script:

  * Truncation happens BEFORE the N check. A sequence whose only N lies beyond
    --max-length is therefore KEPT (its N was already cut away). Reordering these two
    steps changes the record count.
  * Input files are collected with glob("*.fasta"). A file named "*.fa" in the same
    directory is silently skipped -- this is why the GENCODE 3'UTR export that ships
    alongside the RNAcentral files is not part of the corpus.

Memory: the pooled (header, sequence) list is held in RAM before shuffling, which peaks
around 60-70 GB for the full corpus. Run it somewhere with that much memory, or shard.
"""
import argparse
import glob
import multiprocessing as mp
import os
import random


def parse_fasta(file_path, max_length):
    """Parse one FASTA file, returning filtered (header, sequence) tuples."""
    sequences = []
    header, chunks = None, []

    def flush():
        if header and chunks:
            seq = "".join(chunks).upper()[:max_length]
            if "N" not in seq:                    # note: after truncation, by design
                sequences.append((header, seq))

    with open(file_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header, chunks = line, []
            else:
                chunks.append(line)
    flush()
    print(f"  {os.path.basename(file_path)}: kept {len(sequences):,}", flush=True)
    return sequences


def count_records(path):
    with open(path, "r") as f:
        return sum(1 for line in f if line.startswith(">"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True,
                    help="directory holding the source *.fasta files")
    ap.add_argument("--output", required=True, help="output FASTA path")
    ap.add_argument("--max-length", type=int, default=1024,
                    help="truncate sequences to this many nt (paper corpus: 1024)")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed (paper corpus: 42)")
    ap.add_argument("--chunk-size", type=int, default=10000, help="records per write batch")
    ap.add_argument("--workers", type=int, default=min(48, mp.cpu_count()))
    args = ap.parse_args()

    random.seed(args.seed)
    fasta_files = sorted(glob.glob(os.path.join(args.data_dir, "*.fasta")))
    if not fasta_files:
        raise SystemExit(f"no *.fasta files in {args.data_dir}")

    print("=" * 70)
    print(f"source dir : {args.data_dir}")
    print(f"output     : {args.output}")
    print(f"max length : {args.max_length} nt      seed: {args.seed}")
    print(f"inputs     : {len(fasta_files)} file(s)")
    for f in fasta_files:
        print(f"             {os.path.basename(f)}")
    print("=" * 70)

    print("\ncounting input records ...", flush=True)
    total_in = sum(count_records(f) for f in fasta_files)
    print(f"input sequences: {total_in:,}")

    print(f"\nparsing with {args.workers} workers ...", flush=True)
    with mp.Pool(processes=args.workers) as pool:
        results = pool.starmap(parse_fasta, [(f, args.max_length) for f in fasta_files])
    all_sequences = [rec for part in results for rec in part]

    kept = len(all_sequences)
    print(f"\ninput      : {total_in:,}")
    print(f"kept       : {kept:,}")
    print(f"filtered   : {total_in - kept:,} ({100 * (total_in - kept) / max(total_in, 1):.2f}%) "
          f"-- sequences still containing N after truncation")
    if not all_sequences:
        raise SystemExit("nothing survived filtering")

    print(f"\nshuffling {kept:,} sequences (seed {args.seed}) ...", flush=True)
    random.shuffle(all_sequences)

    print(f"writing {args.output} ...", flush=True)
    with open(args.output, "w") as out:
        for i in range(0, kept, args.chunk_size):
            for header, seq in all_sequences[i:i + args.chunk_size]:
                out.write(f"{header}\n{seq}\n")

    total_nt = sum(len(s) for _, s in all_sequences)
    print("=" * 70)
    print(f"wrote {kept:,} sequences, {total_nt:,} nt, mean length {total_nt / kept:.1f}")
    print("Verify with:  python data_processing/audit_corpus.py " + args.output)
    print("=" * 70)


if __name__ == "__main__":
    main()
