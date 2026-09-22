# Corpus de-homologisation

Removes from the pretraining corpus every record homologous to a downstream sequence,
so that a model can be re-pretrained with the homology removed and the benchmark re-run.

```bash
./run_homology_audit.sh corpus.fasta downstream.fasta homology.m8 48

python build_dehomologised_corpus.py \
    corpus.fasta corpus_dehomologised.fasta homology.m8 \
    --identity 0.8 --report dehomologised.json
```

`downstream.fasta` should hold the training, validation and test sequences of every
benchmark involved. Removing test homologs alone leaves family members that entered
through the training split, which is the weaker filter.

Search settings and why each was chosen are documented in `run_homology_audit.sh`.
In short: nucleotide search over both strands, at least 80 % identity over at least
80 % of the downstream sequence, no E-value cut-off, corpus as query so that the
smaller file is the one indexed.

Applied to the released corpus with the 38,308 secondary-structure sequences
(RNAStrAlign, bpRNA TR0/VL0, ArchiveII600, TS0), this removes 16,633,564 of
44,613,362 records (37.3 %), leaving 27,979,798.

Requires MMseqs2; the reported run used release 18-8cc5c.
