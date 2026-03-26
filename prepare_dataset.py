"""
prepare_dataset.py
------------------
Downloads RedPajama-Data-1T-Sample (~2.9 GB) from HuggingFace and prepares it
in all three formats needed for llm-data-pipeline-bench:

  1. Parquet  — native HuggingFace format, used by Ray Data directly
  2. WebDataset .tar shards — used by WebDataset loader
  3. MosaicML .mds shards  — used by StreamingDataset loader

Output layout
-------------
data/
  parquet/
    train-00000-of-00004.parquet   # raw HF parquet files, just re-saved locally
    train-00001-of-00004.parquet
    ...
  webdataset/
    shard-000000.tar
    shard-000001.tar
    ...
  mds/
    index.json
    shard.00000.mds
    shard.00001.mds
    ...

Usage
-----
  # Full download + all formats (recommended first run)
  python prepare_dataset.py

  # Skip re-downloading if you already have the HF cache
  python prepare_dataset.py --skip-download

  # Only prepare specific formats
  python prepare_dataset.py --formats parquet webdataset
  python prepare_dataset.py --formats mds

Requirements
------------
  pip install datasets huggingface_hub pyarrow webdataset streaming tqdm

Note on streaming package:
  pip install mosaicml-streaming
  (package name on PyPI is 'mosaicml-streaming', import name is 'streaming')
"""

import argparse
import io
import json
import os
import struct
import tarfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATASET_ID = "togethercomputer/RedPajama-Data-1T"
DATASET_DIR = Path("data")
OUTPUT_DIR = Path("data-output")
SHARD_SIZE = 5_000          # samples per WebDataset / MDS shard
HF_CACHE_DIR = Path(".hf_cache")


# ---------------------------------------------------------------------------
# Step 1 — Download from HuggingFace
# ---------------------------------------------------------------------------

def download_dataset(skip_download: bool = False):
    """
    Load the dataset from HuggingFace (uses local cache if already downloaded).
    Returns a HuggingFace Dataset object.
    """
    from datasets import load_dataset

    print(f"\n{'='*60}")
    print(f"  Step 1: Loading {DATASET_ID}")
    print(f"  Cache dir: {HF_CACHE_DIR.resolve()}")
    print(f"{'='*60}")

    if skip_download:
        print("  --skip-download set, loading from cache only...")

    t0 = time.time()
    ds = load_dataset(f"{DATASET_DIR.resolve()}")

    print(f"  Loaded {len(ds):,} samples in {time.time()-t0:.1f}s")
    print(f"  Columns: {ds.column_names}")
    print(f"  {ds['train']}")
    ds_data = ds['train'][0]

    print(f"  Sample record 0 info: type:{type(ds['train'][0])} ds[0].keys:{ds['train'][0].keys()}")
    print(f"  Sample record 0 ds['train'][0].keys():{ds_data.keys()}")

    print(f"  Sample record 0 len(ds['train'][0]['text']): {len(ds_data['text'])}")
    print(f"  Sample record 0 ds['train'][0]['meta']: {ds_data['meta']}")
    return ds['train']


# ---------------------------------------------------------------------------
# Step 2 — Save as local Parquet
# ---------------------------------------------------------------------------

def save_parquet(ds, output_dir: Path, shard_size: int = 50_000):
    """
    Save the dataset as local Parquet shards.
    Ray Data reads Parquet natively — just point it at this directory.
    """
    out = output_dir / "parquet"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 2: Saving Parquet → {out}")
    print(f"{'='*60}")

    n = len(ds)
    n_shards = max(1, n // shard_size)
    shard_rows = n // n_shards

    written = 0
    for i in range(n_shards):
        start = i * shard_rows
        end = start + shard_rows if i < n_shards - 1 else n
        shard = ds.select(range(start, end))

        # Add a globally unique __id__ column for shuffle quality tracking
        ids = list(range(start, end))
        shard = shard.add_column("__id__", ids)

        fname = out / f"shard-{i:05d}-of-{n_shards:05d}.parquet"
        shard.to_parquet(str(fname))
        written += end - start
        print(f"  [{i+1}/{n_shards}] {fname.name}  ({end-start:,} rows)")

    print(f"  Done. {written:,} rows written to {out}")
    return out


# ---------------------------------------------------------------------------
# Step 3 — Convert to WebDataset .tar shards
# ---------------------------------------------------------------------------

def save_webdataset(ds, output_dir: Path, shard_size: int = SHARD_SIZE):
    """
    Convert dataset to WebDataset .tar shards.

    Each sample in the tar contains:
      {key}.txt   — the text content
      {key}.json  — the meta dict + __id__

    WebDataset reads these as:
      batch["__key__"]  → shard_id/sample_id  (used for shuffle tracking)
      batch["txt"]      → text bytes
      batch["json"]     → meta bytes
    """
    out = output_dir / "webdataset"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 3: Saving WebDataset shards → {out}")
    print(f"  Shard size: {shard_size:,} samples/shard")
    print(f"{'='*60}")

    n = len(ds)
    n_shards = max(1, (n + shard_size - 1) // shard_size)
    shard_idx = 0
    sample_idx = 0

    with tqdm(total=n, unit="samples", desc="  Writing") as pbar:
        for shard_start in range(0, n, shard_size):
            shard_end = min(shard_start + shard_size, n)
            fname = out / f"shard-{shard_idx:06d}.tar"

            with tarfile.open(fname, "w") as tar:
                for i in range(shard_start, shard_end):
                    row = ds[i]
                    key = f"{shard_idx:06d}/{i - shard_start:06d}"

                    # text file
                    text_bytes = row["text"].encode("utf-8")
                    _tar_add_bytes(tar, f"{key}.txt", text_bytes)

                    # meta + __id__ as json
                    meta = json.loads(row["meta"]) if isinstance(row["meta"], str) else row["meta"]
                    meta["__id__"] = sample_idx
                    meta_bytes = json.dumps(meta).encode("utf-8")
                    _tar_add_bytes(tar, f"{key}.json", meta_bytes)

                    sample_idx += 1
                    pbar.update(1)

            shard_idx += 1

    print(f"  Done. {n_shards} shards written to {out}")

    # Write a simple manifest for easy glob loading
    manifest = sorted(str(p.name) for p in out.glob("*.tar"))
    (out / "manifest.txt").write_text("\n".join(manifest))
    print(f"  Manifest written: {out / 'manifest.txt'}")
    return out


def _tar_add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add a bytes buffer as a file entry to an open tarfile."""
    buf = io.BytesIO(data)
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, buf)


# ---------------------------------------------------------------------------
# Step 4 — Convert to MosaicML StreamingDataset .mds shards
# ---------------------------------------------------------------------------

def save_mds(ds, output_dir: Path, shard_size: int = SHARD_SIZE):
    """
    Convert dataset to MosaicML StreamingDataset .mds format.

    Requires: pip install mosaicml-streaming
    """
    out = output_dir / "mds"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 4: Saving MDS shards → {out}")
    print(f"  Shard size: {shard_size:,} samples/shard")
    print(f"{'='*60}")

    try:
        from streaming import MDSWriter
    except ImportError:
        print("  ERROR: mosaicml-streaming not installed.")
        print("  Run: pip install mosaicml-streaming")
        print("  Skipping MDS conversion.")
        return None

    columns = {
        "__id__": "int",
        "text": "str",
        "meta": "str",
        "red_pajama_subset": "str",
    }

    with MDSWriter(out=str(out), columns=columns, size_limit=shard_size * 2048) as writer:
        for i, row in enumerate(tqdm(ds, desc="  Writing", unit="samples")):
            meta = row["meta"] if isinstance(row["meta"], str) else json.dumps(row["meta"])
            writer.write({
                "__id__": i,
                "text": row["text"],
                "meta": meta,
                "red_pajama_subset": row.get("red_pajama_subset", ""),
            })

    shards = list(out.glob("*.mds"))
    print(f"  Done. {len(shards)} shards written to {out}")
    return out


# ---------------------------------------------------------------------------
# Step 5 — Print usage summary
# ---------------------------------------------------------------------------

def print_usage_summary(output_dir: Path):
    print(f"\n{'='*60}")
    print("  Dataset prepared. Usage in your benchmark:")
    print(f"{'='*60}")

    parquet_dir = output_dir / "parquet"
    wds_dir = output_dir / "webdataset"
    mds_dir = output_dir / "mds"

    if parquet_dir.exists():
        size = _dir_size_gb(parquet_dir)
        print(f"\n  [Parquet — Ray Data]  ({size:.2f} GB)")
        print(f"""
    import ray
    ds = ray.data.read_parquet("{parquet_dir.resolve()}")
    # __id__ column already present for shuffle tracking
""")

    if wds_dir.exists():
        size = _dir_size_gb(wds_dir)
        shards = sorted(wds_dir.glob("*.tar"))
        print(f"  [WebDataset]  ({size:.2f} GB, {len(shards)} shards)")
        print(f"""
    import webdataset as wds
    dataset = (
        wds.WebDataset("{wds_dir.resolve()}/shard-{{000000..{len(shards)-1:06d}}}.tar")
        .shuffle(1000)
        .decode()
        .to_tuple("__key__", "txt", "json")
    )
    # __key__ = "shard_idx/sample_idx" — use as shuffle tracking ID
""")

    if mds_dir.exists():
        size = _dir_size_gb(mds_dir)
        print(f"  [MosaicML StreamingDataset]  ({size:.2f} GB)")
        print(f"""
    from streaming import StreamingDataset
    dataset = StreamingDataset(
        local="{mds_dir.resolve()}",
        shuffle=True,
        shuffle_algo="py1s",
    )
    # __id__ column present for shuffle tracking
""")

    print(f"{'='*60}\n")


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Download and prepare RedPajama-1T-Sample for benchmarking"
    )
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HuggingFace download, use cached version",
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["parquet", "webdataset", "mds"],
        # default=["parquet", "webdataset", "mds"],
        default=[],
        help="Which output formats to generate (default: parquet)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Samples per shard for WebDataset and MDS (default: {SHARD_SIZE})",
    )
    p.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=["common_crawl", "c4", "github", "arxiv", "wikipedia", "stackexchange"],
        help="Only keep samples from one RedPajama subset (default: all)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    t_start = time.time()
    print(f"\nllm-data-pipeline-bench — dataset preparation")
    print(f"Output dir : {args.output_dir.resolve()}")
    print(f"Formats    : {args.formats}")
    if args.subset:
        print(f"Subset     : {args.subset}")

    # 1. Download
    ds = download_dataset(skip_download=args.skip_download)

    # 2. Optional subset filter
    if args.subset:
        before = len(ds)
        ds = ds.filter(
            lambda x: x["red_pajama_subset"] == args.subset,
            desc=f"Filtering to {args.subset}",
        )
        print(f"  Filtered: {before:,} → {len(ds):,} samples")

    # 3. Convert to requested formats
    if "parquet" in args.formats:
        save_parquet(ds, args.output_dir)

    if "webdataset" in args.formats:
        save_webdataset(ds, args.output_dir, shard_size=args.shard_size)

    if "mds" in args.formats:
        save_mds(ds, args.output_dir, shard_size=args.shard_size)

    # 4. Summary
    print_usage_summary(args.output_dir)
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
