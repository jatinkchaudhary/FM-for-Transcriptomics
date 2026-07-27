# Validation record

- `remote_decoder_smoke.json`: live H100 API smoke test for Txn_Jatin,
  Txn_Jatin OSDR LoRA, BRIDGE, and all five BulkFormer decoders.
- `remote_h100_studio.png`: deployed desktop UI after loading a random 50-gene
  TCGA test and completing Txn_Jatin inference.
- `remote_downstream_retrieval.png`: exact current-input retrieval rendered
  from a live Txn_Jatin H100 downstream request.
- `remote_downstream_map.png`: request-scoped UMAP in interactive 3D mode,
  with the active query highlighted.
- `studio_results_desktop.png`: desktop measured-results rendering.
- `studio_results_mobile.png`: 390-pixel responsive rendering.

The five embedding-only controls were separately verified to return HTTP 422
with `imputation_output: "NaN"`.
