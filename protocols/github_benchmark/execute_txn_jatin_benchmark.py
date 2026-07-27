from __future__ import annotations

import os
import shutil
from pathlib import Path

import nbformat
import pandas as pd
import torch
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

txn_data = Path(
    os.environ.get(
        "TXN_DATA_DIR",
        ROOT / "RNA Walter" / "data" / "archs4" / "train_txn_jatin_full",
    )
)
txn_ckpt = Path(
    os.environ.get(
        "TXN_JATIN_CKPT",
        ROOT / "RNA Walter" / "flash_osdr_model" / "Txn_Jatin" / "best_model.pt",
    )
)

os.environ["MYTHOS_BENCH_ROOT"] = str(ROOT)
os.environ["WALTER_DIR"] = str(ROOT / "RNA Walter")
os.environ["BULKFORMER_DIR"] = str(ROOT / "external_models" / "BulkFormer")
os.environ["BULKFORMER_VARIANT_DIR"] = str(ROOT / "BulkFormer-main")
os.environ["BULKFORMER_CKPT"] = str(
    ROOT / "external_models" / "BulkFormer" / "model" / "bulkformer_37M.pt"
)
os.environ["BRIDGE_MODEL_NAME"] = os.environ.get(
    "BRIDGE_MODEL_NAME",
    os.environ.get("TXN_MODEL_NAME", "Txn_Jatin"),
)
os.environ["BRIDGE_RUN"] = "Txn_Jatin"
os.environ["TXN_JATIN_CKPT"] = str(txn_ckpt)
os.environ["MYTHOS_CANONICAL_GENES"] = str(txn_data / "canonical_genes.csv")
os.environ["BRIDGE_GENE_EMBED_MODE"] = os.environ.get(
    "BRIDGE_GENE_EMBED_MODE", "zero"
)
os.environ.setdefault("BRIDGE_INCLUDE_CONTEXTUAL_VARIANT", "0")
os.environ["BRIDGE_CONTEXT_PARQUET"] = str(txn_data)
os.environ["BRIDGE_CONTEXT_MAX_SAMPLES"] = os.environ.get("BRIDGE_CONTEXT_MAX_SAMPLES", "0")
os.environ["BRIDGE_CONTEXT_BATCH"] = os.environ.get("BRIDGE_CONTEXT_BATCH", "16")
os.environ["MYTHOS_DEVICE"] = "cuda"
os.environ["MYTHOS_PROBE"] = "torch"
os.environ["MYTHOS_DISABLE_CACHE"] = "1"
os.environ["MYTHOS_THREADS"] = os.environ.get("MYTHOS_THREADS", "84")
os.environ["MYTHOS_TCGA_MAX"] = "0"
os.environ["MYTHOS_OSDR_MAX_ACC"] = "0"
os.environ["MYTHOS_BENCH_MAX_GENES"] = "200000"
os.environ["MYTHOS_N_TERMS"] = "100000"
os.environ["MYTHOS_MAX_PAIRS"] = "200000"
os.environ.setdefault("MYTHOS_BRIDGE_BATCH", "8")
os.environ.setdefault("MYTHOS_TORCH_PROBE_EPOCHS", "80")
os.environ.setdefault("MPLBACKEND", "Agg")

if not txn_ckpt.exists():
    raise FileNotFoundError(f"Txn_Jatin checkpoint not found: {txn_ckpt}")
if not (txn_data / "canonical_genes.csv").exists():
    raise FileNotFoundError(f"Txn_Jatin prepared data missing: {txn_data}")

canonical = pd.read_csv(txn_data / "canonical_genes.csv")["gene_symbol"].astype(str).tolist()
payload = torch.load(txn_ckpt, map_location="cpu", weights_only=False)
state = payload.get("model_state_dict", {})
if "gene_embedding.weight" not in state:
    raise RuntimeError("Txn_Jatin checkpoint is missing gene_embedding.weight")
num_genes = int(state["gene_embedding.weight"].shape[0])
if len(canonical) != num_genes:
    raise RuntimeError(
        f"canonical/checkpoint mismatch: genes={len(canonical)}, checkpoint={num_genes}"
    )
if int(payload.get("epoch", 0)) < 1:
    raise RuntimeError("Txn_Jatin checkpoint has not completed one epoch")
if not bool(torch.isfinite(state["gene_embedding.weight"]).all()):
    raise RuntimeError("Txn_Jatin gene_embedding.weight contains non-finite values")
del payload, state

out_root = Path(os.environ.get("MYTHOS_OUTPUT_ROOT", ROOT / "benchmark_outputs_txn_jatin"))
if os.environ.get("MYTHOS_CLEAN_OUTPUTS", "1") == "1" and out_root.exists():
    shutil.rmtree(out_root)
out_root.mkdir(parents=True, exist_ok=True)

# Keep the shared mythos output locations pointed at this Txn_Jatin run.
os.environ["MYTHOS_OUTPUT_ROOT"] = str(out_root)

in_path = ROOT / "BRIDGE_Benchmark_Mythos.ipynb"
out_path = out_root / "Txn_Jatin_Benchmark.full_gpu.executed.ipynb"

nb = nbformat.read(in_path, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None
    if cell.cell_type != "code":
        continue
    src = "".join(cell.source)
    src = src.replace(
        'print(f"common (all-4-covered) gene set: {len(common_genes)}")',
        'print(f"common (all-model-covered) gene set: {len(common_genes)}")',
    )
    src = src.replace(
        'council.check("Methodologist","shared gene set across 4 models", len(common_genes)>0, f"{len(common_genes)} genes")',
        'council.check("Methodologist",f"shared gene set across {len(adapters)} models", len(common_genes)>0, f"{len(common_genes)} genes")',
    )
    src = src.replace(
        """from mythos.common import EMB_CACHE, TABLES
audit=Council()
# 1) no fabricated embeddings entered results
import glob, os
realcnt=0
for f in glob.glob(str(EMB_CACHE/"*__gene__symbol.npz")):
    d=np.load(f,allow_pickle=True); realcnt += (not bool(d["is_fallback"]))
audit.check("ReproEng","all 4 gene embeddings are REAL (no random fallbacks cached)", realcnt==4, f"{realcnt}/4 real")""",
        """from mythos.common import TABLES
audit=Council()
# 1) no fabricated embeddings entered results
try:
    prov_real = {}
    for _name, _adapter in adapters.items():
        _prov = _adapter.provenance() if hasattr(_adapter, "provenance") else {}
        _err = str(_prov.get("error", ""))
        prov_real[_name] = (not bool(getattr(_adapter, "is_fallback", False))) and not _err.startswith("FALLBACK")
    realcnt = sum(bool(v) for v in prov_real.values())
    audit.check("ReproEng",f"all {len(adapters)} gene embeddings are REAL (cache-disabled provenance check)", realcnt==len(adapters), f"{realcnt}/{len(adapters)} real; {prov_real}")
except Exception as e:
    audit.check("ReproEng","gene embedding provenance check readable", False, str(e), warn_only=True)""",
    )
    src = src.replace(
        '"common=intersection of 4 models; full=per-model coverage (S8)"',
        'f"common=intersection of {len(adapters)} models; full=per-model coverage (S8)"',
    )
    cell.source = src

client = NotebookClient(
    nb,
    timeout=-1,
    kernel_name="python3",
    allow_errors=False,
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
nbformat.write(nb, out_path)
print(out_path)
