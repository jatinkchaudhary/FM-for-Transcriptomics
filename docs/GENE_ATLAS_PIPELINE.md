# Gene Atlas and Language Head

## Pipeline

1. The selected expression decoder completes the uploaded gene-by-sample
   matrix.
2. The atlas runtime aligns input genes to a versioned reference matrix.
3. Each sample is compared with reference expression profiles using
   mean-centered cosine similarity on `log1p` expression.
4. The API returns nearest references, species and tissue evidence, annotated
   high-expression genes, disease-marker overlaps, and mouse-human orthologs.
5. When enabled, Ollama summarizes only this structured evidence.

The language model is not part of imputation and cannot change the expression
matrix or atlas scores.

## Atlas Contract

The remote production deployment uses atlas version
`production_v1_20260730`: GTEx and TCGA recount2 references, balanced ARCHS4
human/mouse references, MyGene.info annotations, Enrichr disease/phenotype
sets, and the official MGI mouse-human homology report. Exact counts and
validation results are recorded in
`validation/remote_production_atlas_ollama_20260730.md`.

Build `atlas_expression.npz` with:

```bash
python scripts/build_atlas_index.py \
  --expression reference_expression.parquet \
  --metadata reference_metadata.csv \
  --output atlas_expression.npz \
  --input-scale raw
```

Expression input has genes in rows and `reference_id` values in columns.
Metadata requires:

```text
reference_id,species,tissue,study,source
```

Optional evidence files:

- `gene_annotations.json`: gene symbol to inspectable annotation object.
- `disease_gene_sets.json`: association name to gene-symbol list.
- `mouse_human_orthologs.csv`: `mouse_symbol,human_symbol,orthology_type,source`.

Every production index should pin dataset releases and include source URLs,
licenses, preprocessing, checksums, and sample exclusion rules.

## Ollama

After installing Ollama and pulling the configured model:

```bash
ollama pull gpt-oss:120b
```

Set `atlas.ollama.enabled` to `true`. Keep Ollama bound to localhost. The
backend sends deterministic, low-temperature prompts and reports the model
name and availability status with every response.

## Interpretation Boundary

- Species weights summarize nearest-reference evidence; they are not a
  calibrated species classifier.
- Tissue scores indicate resemblance to the indexed references and may be
  ambiguous for mixtures, tumors, cell lines, or low-coverage panels.
- Disease-marker overlaps are associations, not patient-level probabilities,
  diagnoses, or treatment recommendations.
- Mouse-human mappings are reported only from the configured orthology table.
- Imputation and atlas errors can compound, so observed and imputed genes
  should remain distinguishable in exported reports.
