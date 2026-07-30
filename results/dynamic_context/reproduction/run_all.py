"""
mythos.run_all — end-to-end CPU benchmark driver.

Runs the whole pipeline headlessly and writes every score table + figure to
benchmark_outputs/.  The notebook (BRIDGE_Benchmark_Mythos.ipynb) calls the same
functions section by section; this script is the "run it all" entrypoint.

Usage
-----
    python -m mythos.run_all                 # full run (uses MYTHOS_* env knobs)
    MYTHOS_QUICK=1 python -m mythos.run_all   # fast smoke test
    python -m mythos.run_all --quick          # same

CPU-friendly by construction: device=cpu, threads pinned, samples/genes capped
via env (see mythos.common.MythosConfig).  Models that are not installed fall
back gracefully and are flagged; no numbers are fabricated.
"""
from __future__ import annotations

import os
import sys
import argparse

import numpy as np
import pandas as pd

from . import common
from .common import CFG, Council, FIG, TABLES, OUT, BENCH_ROOT, WALTER_ROOT
from . import data as D
from . import explore as E
from . import adapters as A
from . import gene_bench as GB
from . import sample_bench as SB
from . import leaderboard as LB


def main(argv=None):
    ap = argparse.ArgumentParser(description="BRIDGE Mythos benchmark (CPU)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    ap.add_argument("--skip-sample", action="store_true",
                    help="skip the Walter sample benchmark (model forward)")
    args = ap.parse_args(argv)
    if args.quick:
        os.environ["MYTHOS_QUICK"] = "1"
        CFG.__init__()                      # re-read env
    CFG.apply_thread_caps()

    council = Council()
    council.section("S0 — setup & provenance")
    council.info("ReproEng", "paths", f"BENCH_ROOT={BENCH_ROOT} WALTER={WALTER_ROOT}")
    council.info("ReproEng", "config",
                 f"quick={CFG.quick} threads={CFG.threads} tcga_max={CFG.tcga_max} "
                 f"osdr_max_acc={CFG.osdr_max_acc} bench_max_genes={CFG.bench_max_genes}")

    # ---- S1/S3 ingest ----------------------------------------------------- #
    council.section("S1/S3 — ingest")
    tcga = D.load_tcga(council)
    osdr = D.load_osdr("Mus musculus", council=council)
    E.summarize_metadata(tcga, "HUMAN/TCGA", council)
    E.summarize_metadata(osdr, "MOUSE/OSDR", council)

    tcga_pp, hrep = D.preprocess(tcga, name="HUMAN/TCGA", council=council)
    osdr_pp, mrep = D.preprocess(osdr, name="MOUSE/OSDR", council=council)

    # ---- S4/S5 orthologs + cross-species ---------------------------------- #
    council.section("S4/S5 — orthologs & cross-species")
    orth = D.load_orthologs(council)
    ens2sym = D.load_mouse_ens_to_human_symbol(council)
    mouse_hsym = D.map_mouse_bundle_to_human_symbols(osdr_pp, ens2sym, council)
    cross = E.cross_species_correlation(tcga_pp, mouse_hsym, orth, council=council,
                                        out=FIG / "cross_species_corr.png") \
        if (tcga_pp is not None and mouse_hsym is not None and len(orth)) else None

    # ---- S7 frozen gene embeddings ---------------------------------------- #
    council.section("S7 — frozen gene-embedding extraction")
    canon = D.load_canonical_genes(council)
    hallmark = GB.load_hallmark(council)
    # benchmark vocabulary: ortholog-backed human symbols within the canonical set
    common_syms = sorted(set(orth["human_symbol"].str.upper()) & set(g.upper() for g in canon)) \
        if len(orth) else [g.upper() for g in canon]
    bench_genes = (common_syms or [g.upper() for g in canon])[:CFG.bench_max_genes]
    council.info("CompBio", "benchmark gene vocabulary", f"{len(bench_genes)} symbols")

    adapters = A.build_adapters(council)
    gene_emb = A.extract_gene_embeddings(adapters, bench_genes, council)
    common_genes, finite = A.harmonised_gene_sets(gene_emb, bench_genes)
    council.check("Methodologist", f"shared (common) gene set across {len(adapters)} models",
                  len(common_genes) > 0, f"{len(common_genes)} genes")

    # ---- S8 Zhong gene-functional benchmark ------------------------------- #
    council.section("S8 — Zhong gene-functional benchmark (common & full)")
    go = GB.load_go_bp(council)
    dgn = GB.load_disgenet(council)
    frames = []
    for variant in ("common", "full"):
        cg = common_genes if variant == "common" else None
        if go:
            d = GB.single_gene(gene_emb, bench_genes, go, variant, cg,
                               tag="GO-BP", council=council); d["track"] = "single-gene-GO"; frames.append(d)
        if dgn:
            d = GB.single_gene(gene_emb, bench_genes, dgn, variant, cg,
                               tag="DisGeNET", council=council); d["track"] = "single-gene-disease"; frames.append(d)
        # shuffle control (real-vs-chance) on GO, full only
        if go and variant == "full":
            d = GB.single_gene(gene_emb, bench_genes, go, variant, cg, tag="GO-shuffled",
                               shuffle_control=True, council=council)
            d["track"] = "single-gene-GO"; frames.append(d)
        # Paired interactions may run as a separate supplement that reuses
        # these exact exported matrices, avoiding a second full contextual pass.
        if os.environ.get("MYTHOS_SKIP_PAIRED", "0").strip().lower() not in {
            "1", "true", "yes", "on"
        }:
            for pairs, nm in [(GB.load_pairs_trrust(council), "TRRUST-TF"),
                              (GB.load_pairs_synlethdb(council), "SynLethDB-SL")]:
                if pairs:
                    d = GB.paired_genes(gene_emb, bench_genes, pairs, name=nm,
                                        variant=variant, common_genes=cg, council=council)
                    if len(d):
                        d["track"] = "paired"; frames.append(d)
        # gene-set matching
        if hallmark:
            d = GB.gene_set_matching(gene_emb, bench_genes, hallmark, variant, cg,
                                     council=council)
            if len(d):
                d["track"] = "gene-set"; frames.append(d)
    gene_scores = LB.collect_gene_scores(*frames)
    if len(gene_scores):
        gene_scores.to_csv(TABLES / "gene_scores.csv", index=False)

    # temporal GO holdout (needs dated snapshots GO_OLD/GO_NEW; else WARN+skip)
    GB.single_gene_temporal(gene_emb, bench_genes, {}, {}, council=council)

    # ---- S9 Walter sample benchmark --------------------------------------- #
    sample_scores = pd.DataFrame()
    if not args.skip_sample:
        council.section("S9 — Walter sample benchmark")
        bridge = adapters["BRIDGE"]
        def sample_emb(bundle):
            if bundle is None:
                return {}
            out = {}
            for name, ad in adapters.items():
                try:
                    e = ad.sample_embeddings(bundle.X)
                except Exception:
                    e = None
                if e is not None:
                    out[name] = e
            return out
        s_tcga = SB.run_sample_benchmark(tcga_pp, sample_emb(tcga_pp),
                                         "TCGA-cancer-type", council)
        s_osdr = SB.run_sample_benchmark(mouse_hsym if mouse_hsym is not None else osdr_pp,
                                         sample_emb(mouse_hsym if mouse_hsym is not None else osdr_pp),
                                         "OSDR-spaceflight", council)
        sample_scores = pd.concat([d for d in (s_tcga, s_osdr) if len(d)],
                                  ignore_index=True) if any(len(d) for d in (s_tcga, s_osdr)) else pd.DataFrame()
        if len(sample_scores):
            sample_scores.to_csv(TABLES / "sample_scores.csv", index=False)
            SB.plot_sample_benchmark(sample_scores, out=FIG / "sample_benchmark.png")

    # ---- S10 leaderboard + cross-model ------------------------------------ #
    council.section("S10 — leaderboard & cross-model analyses")
    LB.plot_leaderboard(gene_scores, out=FIG / "leaderboard.png")
    LB.leaderboard_table(gene_scores)
    LB.intermodel_similarity(gene_emb, bench_genes, council=council,
                             out=FIG / "intermodel_similarity.png")
    LB.performance_correlation(gene_scores, council=council,
                               out=FIG / "performance_correlation.png")
    summ = LB.summary_table(gene_scores, sample_scores)

    # ---- S12 council report ----------------------------------------------- #
    council.section("S12 — final Council report")
    council.report(save_csv=TABLES / "council_ledger.csv")
    print(f"\nAll outputs under: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
