from __future__ import annotations
import os, urllib.request
from pathlib import Path
import pandas as pd
ROOT = Path(os.environ['MYTHOS_BENCH_ROOT'])
os.chdir(ROOT)
from mythos.common import Council
from mythos import data as D, adapters as A, gene_bench as GB
out = Path(os.environ['KEGG_OUT']); out.mkdir(parents=True, exist_ok=True)
council = Council()
text = urllib.request.urlopen('https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human', timeout=60).read().decode()
kegg = GB.parse_enrichr_gmt(text)
canon = D.load_canonical_genes(council)
orth = D.load_orthologs(council)
bench_genes = sorted(set(orth['human_symbol'].str.upper()) & set(g.upper() for g in canon)) if len(orth) else [g.upper() for g in canon]
bench_genes = bench_genes[:int(os.environ.get('MYTHOS_BENCH_MAX_GENES','200000'))]
adapters = A.build_adapters(council)
gene_emb = A.extract_gene_embeddings(adapters, bench_genes, council)
common_genes, _ = A.harmonised_gene_sets(gene_emb, bench_genes)
frames=[]
for variant in ('common','full'):
    cg = common_genes if variant == 'common' else None
    d = GB.gene_set_matching(gene_emb, bench_genes, kegg, variant, cg, n_pairs=int(os.environ.get('MYTHOS_MAX_PAIRS','200000')), council=council)
    if len(d):
        d['track']='KEGG-gene-set'; frames.append(d)
res = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
res.to_csv(out / 'kegg_gene_set_scores.csv', index=False)
council.report(save_csv=out / 'kegg_council_ledger.csv')
print(out / 'kegg_gene_set_scores.csv')
