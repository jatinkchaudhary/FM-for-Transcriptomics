# Txn_Jatin sparse-panel biomarker pilot

This folder tests the frozen final 20-epoch Txn_Jatin checkpoint on TCGA before
expanding the experiment to other models.

## Experimental order

1. Build a deterministic patient-level 70/30 discovery/test split.
2. Select 2,000- and 1,000-gene observed panels using discovery expression only.
3. Exclude every gene in ten prespecified Hallmark programs from both panels.
4. Run frozen Txn_Jatin inference on the H100 with all non-panel genes masked.
5. Evaluate held-out hidden-gene reconstruction and Hallmark state recovery.
6. Run patient-grouped phenotype probes and hidden tumor-normal effect ranking.

The 2,000-gene panel runs first. The 1,000-gene panel is a more severe stress
test. Results are exploratory internal evidence and must not be described as a
validated clinical biomarker.

All commands, logs, protocol files, hashes, predictions, tables, figures, and
the generated report remain inside this folder.

## Status

Both panels completed on the remote H100. The principal finding is negative:
Txn_Jatin does not reconstruct expression or preserve hidden biomarker rankings
when 86%-93% of genes are masked. Decoder-derived pathway projections remain
cancer-separable, but the observed panel is stronger and individual pathway
values are not reliably calibrated. See `REPORT.md` for the full tables and
claim boundary.
