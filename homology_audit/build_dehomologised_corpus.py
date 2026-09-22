#!/usr/bin/env python3
"""Delete every corpus record homologous to a downstream sequence.

Input is the alignment file from run_homology_audit.sh, whose first column is the
corpus record and whose third is sequence identity. The search already enforced
>= 80 % identity and >= 80 % coverage of the downstream sequence, so by default
every record appearing in the file is dropped; --identity re-thresholds upwards.

The downstream set should contain the training, validation and test sequences of
every benchmark involved, not only the test split: removing test homologs alone
leaves family members that entered through the training split.
"""
import argparse
import json

COLUMNS = ["query", "target", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]


def drop_ids(m8_paths, identity):
    """Corpus record ids with an alignment at or above `identity`."""
    drop, rows = set(), 0
    qi, fi = COLUMNS.index("query"), COLUMNS.index("fident")
    for path in m8_paths:
        with open(path) as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if len(f) < len(COLUMNS):
                    continue
                rows += 1
                if float(f[fi]) >= identity - 1e-12:
                    drop.add(f[qi])
    return drop, rows


def filter_fasta(src, dst, drop):
    """Stream the FASTA, skipping dropped records. Header id is the first token."""
    kept = dropped = 0
    keep = False
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith(">"):
                rid = line[1:].split()[0]
                keep = rid not in drop
                kept += keep
                dropped += not keep
            if keep:
                fout.write(line)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_in")
    ap.add_argument("corpus_out")
    ap.add_argument("m8", nargs="+", help="alignment files from run_homology_audit.sh")
    ap.add_argument("--identity", type=float, default=0.8)
    ap.add_argument("--report", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    drop, rows = drop_ids(a.m8, a.identity)
    print(f"{rows:,} alignments -> {len(drop):,} corpus records to drop")
    kept, dropped = filter_fasta(a.corpus_in, a.corpus_out, drop)
    total = kept + dropped
    print(f"{total:,} records in -> {kept:,} kept, {dropped:,} dropped "
          f"({100 * dropped / total:.2f} %)")

    if a.report:
        json.dump({"corpus_in": a.corpus_in, "corpus_out": a.corpus_out,
                   "identity_threshold": a.identity, "m8_files": a.m8,
                   "alignments": rows, "corpus_records_in": total,
                   "dropped": dropped, "kept": kept,
                   "dropped_pct": round(100 * dropped / total, 4)},
                  open(a.report, "w"), indent=2)


if __name__ == "__main__":
    main()
