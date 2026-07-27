# Validation record

- `remote_decoder_smoke.json`: live H100 API smoke test for Txn_Jatin,
  Txn_Jatin OSDR LoRA, BRIDGE, and all five BulkFormer decoders.
- `remote_h100_studio.png`: deployed desktop UI after loading a random 50-gene
  TCGA test and completing Txn_Jatin inference.
- `studio_results_desktop.png`: desktop measured-results rendering.
- `studio_results_mobile.png`: 390-pixel responsive rendering.

The five embedding-only controls were separately verified to return HTTP 422
with `imputation_output: "NaN"`.
