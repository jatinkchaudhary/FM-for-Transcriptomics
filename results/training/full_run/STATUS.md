# Txn_Jatin 20-Epoch H100 Run Status

Updated: 2026-07-14 17:56 UTC

Remote run directory:

```text
/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238
```

Current state:

- Full 20-epoch training is active in tmux session `txn_jatin_20ep_20260624_015238_resume_20260707_222047`.
- Watchdog session `txn_jatin_20ep_watchdog_20260624_015238` is active.
- No competing benchmark or notebook jobs were found on the remote check.
- Training processes remain active: 22 `train_flash.py` related processes were present.
- Latest observed training progress: epoch 17/20, batch 1,122/59,889.
- Latest observed train averages in epoch 17: `AvgRecon=0.078160`, `AvgTotal=0.079186`, average step ~1.564s.
- Epoch 16 completed and produced a new best checkpoint with validation loss `0.080209`.
- GPU samples during the check: 98%, 100%, 100% utilization, 67,349 MiB / 81,559 MiB used, ~657-680 W.
- Latest saved epoch checkpoint observed: `epoch_15.pt` at 2026-07-14 17:22 UTC, corresponding to the completed epoch 16/20 summary.
- Latest saved epoch checkpoint observed: `epoch_14.pt` at 2026-07-13 12:28 UTC, corresponding to the completed epoch 15/20 summary.
- Acceleration assessment was written to `reports/acceleration_assessment_20260709_2149.md`; no live hyperparameter restart was performed.
- Latest durable checkpoint `latest.pt` updated at 2026-07-09 18:25 UTC with metadata: `checkpoint_kind=batch`, `epoch=11`, `batch_idx=53999`, `num_batches=54000`, `train_loss=0.0844827844047988`, `gene_embedding_shape=[16055,512]`.
- Remote reboot at 2026-07-07 21:42 UTC killed the prior tmux session; training was restarted from the durable checkpoint at epoch 10, batch 22,000.
- Local logs/reports/configs/env/scripts were synced from remote into this directory on 2026-07-09.

Epoch-10 benchmark detour:

- Epoch-10 checkpoint was stored and hash-validated separately under `Txn_Jatin_10epoch_6thJuly`.
- The epoch-10 benchmark produced partial artifacts but did not complete.
- No `BENCHMARK_COMPLETE` marker is present, and no benchmark process is currently active.
- Root cause evidence from `logs/notebook_full.log`: the runner tried to open missing notebook `Txn_Jatin_10epoch_6thJuly/configs/BRIDGE_Benchmark_Mythos.ipynb`, causing `FileNotFoundError` before final reports and curves were created.
- Partial artifacts include benchmark embeddings and `benchmarks/notebook_full/tables/trrust.tsv`.
- Interruption report: `Txn_Jatin_10epoch_6thJuly/reports/benchmark_interrupted_20260707_202902.md`.
- Main training was restarted rather than leaving the H100 idle.

Benchmark readiness:

- The post-training benchmark path uses the live project runner `/media/volume/TrainingData/home_data/benchmarking_run/Benchmarking/execute_txn_jatin_benchmark.py`.
- The notebook required by that runner exists at `/media/volume/TrainingData/home_data/benchmarking_run/Benchmarking/BRIDGE_Benchmark_Mythos.ipynb`.
- `resume_and_finish.sh` is active inside tmux and is configured to validate the final checkpoint, run the full notebook benchmark, run KEGG supplement, and generate the final report after training reaches 20 epochs.

Completed gates:

- Remote H100, storage, Python, CUDA, PyTorch, datasets, and repository state were inspected.
- Timestamped attached-volume run directory was created.
- Preflight validation passed with zero critical findings.
- Smoke training passed and checkpoint reload validated.
- Batch/throughput tests passed; batch size 16 selected for the full run.
- Full ARCHES4 20-epoch training launched with 958,212 train samples and 20,000 validation samples.
- Training pause/resume around the epoch-10 snapshot was validated.

Still pending:

- Complete all 20 epochs.
- Validate the final checkpoint.
- Run full TCGA and OSDR benchmarks.
- Run gene set, GO, and KEGG benchmark outputs.
- Compare against all available baseline models.
- Generate and validate final publication-grade report.
