#!/usr/bin/env python3
"""Prepare public pretreatment ICI bulk RNA-seq cohorts for frozen inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = ("Gide", "Riaz", "Hugo", "Rose")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_gene(value: object) -> str:
    return str(value).strip().upper()


def fpkm_to_log1p_tpm(frame: pd.DataFrame) -> pd.DataFrame:
    values = np.maximum(frame.to_numpy(dtype=np.float64), 0.0)
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("At least one FPKM sample has no positive expression")
    tpm = values / totals * 1_000_000.0
    return pd.DataFrame(
        np.log1p(tpm).astype(np.float32),
        index=frame.index,
        columns=frame.columns,
    )


def tpm_to_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    values = np.maximum(frame.to_numpy(dtype=np.float64), 0.0)
    return pd.DataFrame(
        np.log1p(values).astype(np.float32),
        index=frame.index,
        columns=frame.columns,
    )


def collapse_genes(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate gene symbols by summing abundance before log1p."""
    frame = frame.copy()
    frame.columns = [clean_gene(x) for x in frame.columns]
    frame = frame.loc[:, [bool(x) and x != "NAN" for x in frame.columns]]
    if frame.columns.duplicated().any():
        frame = frame.T.groupby(level=0, sort=True).sum().T
    return frame


def load_clinical(raw: Path, cohort: str) -> pd.DataFrame:
    clinical = pd.read_csv(raw / f"{cohort}_clinical.tsv", sep="\t")
    required = {"Index", "Sample_id", "response_label", "Timing", "cancer_type"}
    missing = sorted(required - set(clinical.columns))
    if missing:
        raise ValueError(f"{cohort} clinical file lacks {missing}")
    clinical = clinical.loc[
        clinical["Timing"].astype(str).str.casefold().eq("pre")
        & clinical["response_label"].isin(["R", "NR"])
    ].copy()
    clinical["label"] = clinical["response_label"].map({"NR": 0, "R": 1})
    clinical["cohort"] = cohort
    if clinical["Index"].duplicated().any():
        raise ValueError(f"{cohort} has duplicate clinical Index values")
    return clinical


def load_gide(raw: Path, clinical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    expr = pd.read_csv(raw / "compass_gide_tpm.tsv", sep="\t")
    expr = expr.set_index("Index").drop(columns=["cancer_code"], errors="ignore")
    expr = collapse_genes(expr)
    missing = clinical.loc[~clinical["Index"].isin(expr.index), "Index"].tolist()
    if missing:
        raise ValueError(f"Gide expression misses clinical samples: {missing}")
    expr = expr.loc[clinical["Index"]]
    return tpm_to_log1p(expr), {
        "source_scale": "TPM",
        "sample_matching": "clinical Index equals expression Index",
        "expression_samples_available": int(len(expr)),
    }


def load_gene_info(raw: Path) -> tuple[dict[str, str], dict]:
    info = pd.read_csv(
        raw / "Homo_sapiens.gene_info.gz",
        sep="\t",
        dtype=str,
        usecols=["GeneID", "Symbol", "type_of_gene"],
    )
    info["Symbol"] = info["Symbol"].map(clean_gene)
    info = info.loc[info["Symbol"].ne("-") & info["Symbol"].ne("")]
    mapping = dict(zip(info["GeneID"], info["Symbol"]))
    return mapping, {
        "ncbi_gene_info_rows": int(len(info)),
        "ncbi_gene_info_sha256": sha256(raw / "Homo_sapiens.gene_info.gz"),
    }


def load_riaz(
    raw: Path, clinical: pd.DataFrame, entrez_to_symbol: dict[str, str]
) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(raw / "GSE91061_Riaz_fpkm.csv.gz")
    gene_col = source.columns[0]
    source[gene_col] = source[gene_col].astype(str)
    mapped = source[gene_col].map(entrez_to_symbol)
    mapped_rows = mapped.notna()
    expr = source.loc[mapped_rows].drop(columns=[gene_col]).T
    expr.columns = mapped.loc[mapped_rows].tolist()
    expr = collapse_genes(expr)

    def key(value: object) -> str:
        value = re.sub(r"-\d+$", "", str(value))
        return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")

    by_key = {key(x): x for x in expr.index}
    matched = []
    missing = []
    for value in clinical["Index"]:
        source_id = by_key.get(key(value))
        if source_id is None:
            missing.append(value)
        else:
            matched.append(source_id)
    if missing:
        raise ValueError(f"Riaz expression misses clinical samples: {missing}")
    expr = expr.loc[matched]
    expr.index = clinical["Index"].tolist()
    return fpkm_to_log1p_tpm(expr), {
        "source_scale": "FPKM converted per sample to TPM",
        "sample_matching": "clinical Index matched after removing GEO lane suffix",
        "entrez_rows": int(len(source)),
        "entrez_rows_mapped": int(mapped_rows.sum()),
        "entrez_rows_unmapped": int((~mapped_rows).sum()),
        "expression_samples_available": int(source.shape[1] - 1),
    }


def load_hugo(raw: Path, clinical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    source = pd.read_excel(raw / "GSE78220_Hugo_PatientFPKM.xlsx")
    expr = source.set_index("Gene").T
    expr = collapse_genes(expr)

    def patient_id(value: object) -> str:
        match = re.match(r"^(PT\d+)", str(value).upper())
        if not match:
            raise ValueError(f"Cannot parse Hugo sample ID: {value}")
        return match.group(1)

    grouped = expr.groupby([patient_id(x) for x in expr.index]).mean()
    requested = clinical["Sample_id"].astype(str).str.upper()
    missing = requested.loc[~requested.isin(grouped.index)].tolist()
    if missing:
        raise ValueError(f"Hugo expression misses clinical patients: {missing}")
    expr = grouped.loc[requested]
    expr.index = clinical["Index"].tolist()
    return fpkm_to_log1p_tpm(expr), {
        "source_scale": "FPKM converted per sample to TPM",
        "sample_matching": "patient ID; Pt27A/Pt27B averaged before TPM conversion",
        "expression_samples_available": int(source.shape[1] - 1),
        "patient_level_samples_available": int(len(grouped)),
        "technical_replicates_aggregated": int(source.shape[1] - 1 - len(grouped)),
    }


def load_rose(raw: Path, clinical: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(raw / "GSE176307_Rose_tpm.tsv.gz", sep="\t")
    gene_col = source.columns[0]
    expr = source.set_index(gene_col).T
    expr = collapse_genes(expr)
    wanted = clinical["Sample_id"].astype(str)
    missing = wanted.loc[~wanted.isin(expr.index)].tolist()
    if missing:
        raise ValueError(f"Rose expression misses clinical samples: {missing}")
    expr = expr.loc[wanted]
    expr.index = clinical["Index"].tolist()
    return tpm_to_log1p(expr), {
        "source_scale": "TPM",
        "sample_matching": "clinical Sample_id equals expression column",
        "expression_samples_available": int(source.shape[1] - 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "raw"
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)

    entrez_to_symbol, gene_info_qc = load_gene_info(raw)
    loaders = {
        "Gide": lambda c: load_gide(raw, c),
        "Riaz": lambda c: load_riaz(raw, c, entrez_to_symbol),
        "Hugo": lambda c: load_hugo(raw, c),
        "Rose": lambda c: load_rose(raw, c),
    }
    matrices: dict[str, pd.DataFrame] = {}
    clinical_frames = []
    qc: dict[str, object] = {
        "normalization": "log1p(TPM)",
        "response_definition": "R=CR/PR; NR=SD/PD (COMPASS harmonization)",
        "gene_info": gene_info_qc,
        "cohorts": {},
    }
    for cohort in COHORTS:
        clinical = load_clinical(raw, cohort)
        matrix, cohort_qc = loaders[cohort](clinical)
        if list(matrix.index) != clinical["Index"].tolist():
            raise AssertionError(f"{cohort}: expression and clinical ordering differ")
        if not np.isfinite(matrix.to_numpy()).all():
            raise ValueError(f"{cohort}: expression contains non-finite values")
        if (matrix.to_numpy() < 0).any():
            raise ValueError(f"{cohort}: log1p(TPM) contains negative values")
        matrices[cohort] = matrix
        clinical_frames.append(clinical)
        cohort_qc.update(
            {
                "patients_retained": int(len(clinical)),
                "responders": int(clinical["label"].sum()),
                "nonresponders": int((clinical["label"] == 0).sum()),
                "cancer_types": sorted(clinical["cancer_type"].unique().tolist()),
                "genes_after_mapping": int(matrix.shape[1]),
                "zero_fraction": float((matrix.to_numpy() == 0).mean()),
            }
        )
        qc["cohorts"][cohort] = cohort_qc
        np.savez_compressed(
            prepared / f"{cohort.lower()}_log1p_tpm.npz",
            X=matrix.to_numpy(dtype=np.float32),
            sample_ids=np.asarray(matrix.index, dtype=str),
            genes=np.asarray(matrix.columns, dtype=str),
        )

    combined_clinical = pd.concat(clinical_frames, ignore_index=True)
    selected_columns = [
        "Index",
        "Sample_id",
        "cohort",
        "cancer_type",
        "ICI",
        "ICI_target",
        "RECIST",
        "response_label",
        "label",
        "Timing",
    ]
    for optional in ("patient_id", "Overall_survival", "Progression Free Survival (Days)"):
        if optional in combined_clinical.columns:
            selected_columns.append(optional)
    combined_clinical[selected_columns].to_csv(
        prepared / "clinical_harmonized.tsv", sep="\t", index=False
    )

    manifest = []
    for path in sorted(raw.iterdir()):
        if path.is_file():
            manifest.append(
                {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    qc["total_patients"] = int(len(combined_clinical))
    qc["total_responders"] = int(combined_clinical["label"].sum())
    qc["total_nonresponders"] = int((combined_clinical["label"] == 0).sum())
    (prepared / "preparation_qc.json").write_text(
        json.dumps(qc, indent=2) + "\n", encoding="utf-8"
    )
    (prepared / "raw_file_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
