# Txn_Jatin Training Metrics

Created: 2026-07-21T16:57:21Z

Source run: `/media/volume/AdditionalHeadroom/Txn_Jatin_runs/full_ARCHES4_20epoch_H100_20260624_015238`

Note: the saved `loss_history.csv` records the final resumed segment as epochs 1-10. The presentation matrix adds `inferred_absolute_epoch=11-20` for the full 20-epoch run.

Files:
- `training_metrics_matrix.csv`: epoch-level training metrics with inferred absolute epoch.
- `training_metrics_matrix.md`: markdown table version.
- `training_loss_curve.png/.svg`: presentation-ready train/validation loss curve.
- `training_loss_matrix_heatmap.png/.svg`: compact loss matrix heatmap.
- `remote_loss_plot.png`: original plot saved by the training run.
- `training_metrics_summary.json`: parsed final/best loss summary.

Summary:
- Final train reconstruction loss: 0.0761440911669026
- Final validation reconstruction loss: 0.0780588462948799
- Best validation reconstruction loss: 0.0780588462948799 at inferred epoch 20
