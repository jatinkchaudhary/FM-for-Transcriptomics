# Remote Atlas Backend Validation

Validated on 2026-07-30 UTC at
`nvidea-h100-2-of-2.bio260281.projects.jetstream-cloud.org`.

## Environment

- GPU: NVIDIA H100 80GB HBM3
- Production API: port 8000
- Deployment:
  `/media/volume/AdditionalHeadroom/Txn_Jatin_studio_20260726`
- Pre-deployment backup: `backups/20260730T002210Z`

## Results

1. Remote unit suite: 7/7 passed.
2. Controlled atlas HTTP integration:
   - completed matrix returned HTTP 200;
   - human-liver reference ranked first with cosine similarity 1.0;
   - human species evidence weight was 1.0;
   - expected two-gene disease-set overlap was returned;
   - language head reported `disabled`;
   - incomplete matrix returned HTTP 400 with an instruction to impute first.
3. Production atlas endpoint returned HTTP 503 because the configured,
   versioned atlas files are not installed. This is the intended fail-closed
   behavior.
4. Live Txn_Jatin imputation returned HTTP 200 after deployment and loaded the
   16,055-gene checkpoint on the H100.
5. Ollama was not installed and nothing was listening on localhost port
   11434.
6. The isolated test server on port 8012 was stopped. Production port 8000
   remained healthy.

## Remaining Deployment Inputs

- Install a curated, versioned human/mouse expression atlas and companion
  metadata, disease sets, annotations, and orthology table.
- Install Ollama, pull the pinned model, test evidence-only generation, and
  then set `atlas.ollama.enabled` to `true`.

No production biological labels were generated from the controlled fixture.
