# ARCHS4 Imputation Studio frontend

`index.dc.html` is the editable UI and `support.js` is its local runtime.

When served by `app/backend/server.py`, the UI uses the same origin for:

- live H100 checkpoint inference through `POST /api/impute`
- request-scoped sample retrieval, maps, and inspection through
  `POST /api/downstream`
- model capability metadata through `GET /api/models`
- measured study results through `GET /api/experiments`
- downloadable result tables under `/results/`

Use `?api=https://another-host:port` only when the API is hosted separately.
Use `?api=` to deliberately disable the API and expose the labeled browser
surrogate.

The model selector distinguishes native expression decoders from embedding-only
controls. Unsupported expression inference is reported as `NaN`; it never
falls back silently when a live backend is connected.

The Downstream tab separates two evidence types:

- **Live current-request analysis** recomputes exact cohort neighbors,
  PCA/UMAP/t-SNE 2D and 3D maps, sample inspection, and interpretation whenever
  the input matrix or model changes.
- **Benchmark evidence** reads the packaged, pinned study tables and is
  explicitly labeled historical.

The live retrieval scope is the uploaded cohort. It does not claim to search
the external BRIDGE ARCHS4 index, which is unavailable in the upstream
repository and is model-specific.
