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

------------
  Sample usage after formatting for each format:
  [Parquet — Ray Data]  (0.52 GB)

    import ray
    ds = ray.data.read_parquet("/home/user/llm-data-bench/data-output/parquet")
    # __id__ column already present for shuffle tracking

  [WebDataset]  (1.16 GB, 5 shards)

    import webdataset as wds
    dataset = (
        wds.WebDataset("/home/user/llm-data-bench/data-output/webdataset/shard-{000000..000004}.tar")
        .shuffle(1000)
        .decode()
        .to_tuple("__key__", "txt", "json")
    )
    # __key__ = "shard_idx/sample_idx" — use as shuffle tracking ID

  [MosaicML StreamingDataset]  (1.13 GB, 111 shards)

    from streaming import StreamingDataset
    dataset = StreamingDataset(
        local="/home/user/llm-data-bench/data-output/mds",
        shuffle=True,
        shuffle_algo="py1s",
    )
    # __id__ column present for shuffle tracking

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

TEXT_DATASET_ID = "togethercomputer/RedPajama-Data-1T"
IMAGE_DATASET_ID = "Maysee/tiny-imagenet"
CC3M_DATASET_ID = "pixparse/cc3m-wds"
DATASET_DIR = Path("data")
OUTPUT_DIR = Path("data-output")
CACHE_DIR = Path(".hf_cache")
SHARD_SIZE = 5_000          # samples per WebDataset / MDS shard
MULTIMODAL_MAX_ROWS = 10_000


# ---------------------------------------------------------------------------
# Step 1 — Download from HuggingFace
# ---------------------------------------------------------------------------

def load_text_dataset():
    from datasets import load_dataset
    print(f"\n{'='*60}")
    print(f"  Step 1: Loading {TEXT_DATASET_ID} for text modality")
    print(f"{'='*60}")

    t0 = time.time()
    ds = load_dataset(f"{DATASET_DIR.resolve()}", split='train', cache_dir=str(CACHE_DIR))

    print(f"  Loaded {len(ds):,} samples in {time.time()-t0:.1f}s")
    print(f"  Columns: {ds.column_names}") # ['text', 'meta']
    print(f"  {ds}")
    ds_data = ds[0]

    print(f"  Sample record 0 info: type:{type(ds[0])} ds[0].keys:{ds[0].keys()}")
    print(f"  Sample record 0 ds[0].keys():{ds_data.keys()}")

    print(f"  Sample record 0 len(ds[0]['text']): {len(ds_data['text'])}")
    print(f"  Sample record 0 ds[0]['meta']: {ds_data['meta']}")
    return ds

def load_image_dataset():
    from datasets import load_dataset
    print(f"\n{'='*60}")
    print(f"  Step 1: Loading {IMAGE_DATASET_ID}")
    print(f"{'='*60}")

    t0 = time.time()
    ds = load_dataset(IMAGE_DATASET_ID, split='train', cache_dir=str(CACHE_DIR))

    print(f"  Loaded {len(ds):,} samples in {time.time()-t0:.1f}s")
    print(f"  Columns: {ds.column_names}") # ['image', 'label']
    print(f"  {ds}")
    ds_data = ds[0]

    print(f"  Sample record 0 info: type:{type(ds[0])} ds[0].keys:{ds[0].keys()}")
    print(f"  Sample record 0 ds[0].keys():{ds_data.keys()}")

    print(f"  Sample record 0 type(ds[0]['image']): {type(ds_data['image'])}")
    print(f"  Sample record 0 ds[0]['label']: {ds_data['label']}")
    return ds

def load_image_text_dataset():
    from datasets import load_dataset
    print(f"\n{'='*60}")
    print(f"  Step 1: Loading {CC3M_DATASET_ID} in streaming mode")
    print(f"{'='*60}")

    t0 = time.time()
    ds = load_dataset(CC3M_DATASET_ID, split='train', cache_dir=str(CACHE_DIR), streaming=True)

    print(f"  Loaded Dataset (Lazy) in {time.time()-t0:.1f}s")
    print(f"  Columns: {ds.column_names}") # ['image', 'label']
    print(f"  {ds}")
    ds_data = next(iter(ds))

    print(f"  Sample record 0 info: type:{type(ds_data)} ds[0].keys:{ds_data.keys()}")
    print(f"  Sample record 0 ds[0].keys():{ds_data.keys()}")
    return ds

def download_dataset(modality: str = "text"):
    """
    Load the dataset from HuggingFace (uses local cache if already downloaded).
    Returns a HuggingFace Dataset object.
    """

    if modality == "text":
        return load_text_dataset()
    elif modality == "images":
        return load_image_dataset()
    elif modality == "image+text":
        return load_image_text_dataset()
    else:
        raise ValueError(f"Invalid modality: {modality}")


# ---------------------------------------------------------------------------
# Step 2 — Save as local Parquet
# ---------------------------------------------------------------------------

def save_parquet(ds, output_dir: Path, modality: str = "text", shard_size: int = 50_000):
    """
    Save the dataset as local Parquet shards.
    Ray Data reads Parquet natively — just point it at this directory.
    """
    import datasets
    iterable_dataset = isinstance(ds, datasets.iterable_dataset.IterableDataset)
    ds_iter = iter(ds)

    out = output_dir / modality / "parquet"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 2: Saving Parquet → {out}")
    print(f"{'='*60}")

    n = len(ds) if not iterable_dataset else MULTIMODAL_MAX_ROWS
    n_shards = max(1, n // shard_size)
    shard_rows = n // n_shards

    written = 0

    def prepare_row(start, end):
        rows = []
        nonlocal ds_iter
        with tqdm(total=end-start, desc="  Preparing rows") as pbar:
            for j in range(start, end):
                row = next(ds_iter)
                pbar.update(1)
                jpg = row["jpg"]
                if isinstance(jpg, dict):           # HF image dict {"bytes": ..., "path": ...}
                    jpg = jpg.get("bytes") or jpg.get("path")
                if hasattr(jpg, "read"):            # file-like
                    jpg = jpg.read()
                if hasattr(jpg, "tobytes"):         # PIL Image
                    buf = io.BytesIO()
                    jpg.save(buf, format="JPEG", quality=85)
                    jpg = buf.getvalue()
                rows.append({
                    "__id__" : j,
                    "jpg" : jpg,
                    "txt" : row["txt"]
                    })
        return rows

    for i in range(n_shards):
        start = i * shard_rows
        end = start + shard_rows if i < n_shards - 1 else n
        fname = out / f"shard-{i:05d}-of-{n_shards:05d}.parquet"
        if iterable_dataset:
            rows = prepare_row(start, end)
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, fname)
        else:
            shard = ds.select(range(start, end))
            # Add a globally unique __id__ column for shuffle quality tracking
            ids = list(range(start, end))
            shard = shard.add_column("__id__", ids)
            shard.to_parquet(str(fname))

        written += end - start
        print(f"  [{i+1}/{n_shards}] {fname.name}  ({end-start:,} rows)")

    print(f"  Done. {written:,} rows written to {out}")
    return out


# ---------------------------------------------------------------------------
# Step 3 — Convert to WebDataset .tar shards
# ---------------------------------------------------------------------------

def save_webdataset(ds, output_dir: Path, modality: str = "text", shard_size: int = SHARD_SIZE):
    """
    Convert dataset to WebDataset .tar shards using wds.ShardWriter.

    Each sample in the tar contains:
      {key}.txt  / {key}.jpg  — the text or image content
      {key}.json              — the meta dict + __id__

    WebDataset reads these as:
      batch["__key__"]  → sample index string
      batch["txt"]      → text bytes   (text modality)
      batch["jpg"]      → jpeg bytes   (images modality)
      batch["json"]     → meta bytes
    """
    import datasets
    import webdataset as wds
    import io as _io

    out = output_dir / modality / "webdataset"
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Step 3: Saving WebDataset shards → {out}")
    print(f"  Shard size: {shard_size:,} samples/shard")
    print(f"{'='*60}")

    shard_pattern = str(out / "shard-%06d.tar")
    iterable_dataset = isinstance(ds, datasets.iterable_dataset.IterableDataset)

    with wds.ShardWriter(shard_pattern, maxcount=shard_size) as sink:
        with tqdm(total=len(ds) if not iterable_dataset else MULTIMODAL_MAX_ROWS, \
            unit="samples", desc="  Writing", disable=iterable_dataset) as pbar:
            '''
            for image: ds.column_names
                ['image', 'label']
            for text: ds.column_names
                ['text', 'meta']
            '''
            for sample_idx, row in enumerate(ds):
                if modality == "text": # we aren't going to use 'meta' columns for GPT-2 forward pass
                    sample = {
                        "__key__": f"{sample_idx:08d}",
                        "txt":     row["text"].encode("utf-8"),
                    }
                elif modality == "images":
                    buf = io.BytesIO()
                    row["image"].save(buf, format="JPEG", quality=95)
                    sample = {
                        "__key__": f"{sample_idx:08d}",
                        "jpg":     buf.getvalue(),
                    }
                elif modality == "image+text":
                    if sample_idx == MULTIMODAL_MAX_ROWS:
                        break
                    buf = io.BytesIO()
                    row["jpg"].save(buf, format="JPEG", quality=95)
                    sample = {
                        "__key__": f"{sample_idx:08d}",
                        "jpg":     buf.getvalue(),
                        "txt":     row["txt"],
                    }

                sink.write(sample)
                pbar.update(1)

    n_shards = len(sorted(out.glob("*.tar")))
    print(f"  Done. {n_shards} shards written to {out}")

    manifest = sorted(str(p.name) for p in out.glob("*.tar"))
    (out / "manifest.txt").write_text("\n".join(manifest))
    print(f"  Manifest written: {out / 'manifest.txt'}")
    return out


# ---------------------------------------------------------------------------
# Step 4 — Convert to MosaicML StreamingDataset .mds shards
# ---------------------------------------------------------------------------

def save_mds(ds, output_dir: Path, modality: str = "text", shard_size: int = SHARD_SIZE):
    """
    Convert dataset to MosaicML StreamingDataset .mds format.

    Requires: pip install mosaicml-streaming

    shard size is sharding size limit in bytes (in here its actually shar_size*2048 check later code)
    
    MDSWriter just keeps writing rows into the current shard until it hits the size_limit in bytes, 
    then starts a new shard. Simple byte-based chunking.
    """
    out = output_dir / modality / "mds"
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

    columns = {}
    if modality == "text":
        columns = {
            "__id__": "int",
            "text": "str",
            "meta": "str",
        }
    elif modality == "images":
        columns = {
            "__id__": "int",
            "image": "bytes",
            "label": "int",
        }
    elif modality == "image+text":
        columns = {
            "__id__": "int",
            "jpg": "bytes",
            "txt": "str",
        }
    else:
        raise ValueError(f"Invalid modality: {modality}")

    with MDSWriter(out=str(out), columns=columns, size_limit=shard_size * 2048) as writer:
        for i, row in enumerate(tqdm(ds, desc="  Writing", unit="samples")):
            meta = row["meta"] if isinstance(row["meta"], str) else json.dumps(row["meta"], default=str)
            if modality == "text":
                writer.write({
                    "__id__": i,
                    "text": row["text"],
                    "meta": meta,
                })
            elif modality == "images":
                buf = io.BytesIO()
                row["image"].save(buf, format="JPEG", quality=95)
                writer.write({
                    "__id__": i,
                    "image": buf.getvalue(),  # proper JPEG bytes, PIL can decode
                    "label": row["label"],
                })
            elif modality == "image+text":
                if i == MULTIMODAL_MAX_ROWS:
                    break
                buf = io.BytesIO()
                row["jpg"].save(buf, format="JPEG", quality=95)
                writer.write({
                    "__id__": i,
                    "jpg": buf.getvalue(),
                    "txt": row["txt"],
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
        description="Download and prepare RedPajama-1T for benchmarking"
    )
    p.add_argument(
        "--formats",
        nargs="+",
        choices=["parquet", "webdataset", "mds"],
        default=["parquet", "webdataset", "mds"],
        help="Which output formats to generate (default: parquet, webdataset, mds)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    p.add_argument(
        "--modality",
        type=str,
        default="text",
        help=f"Modality 'text' or 'images' or 'image+text'(default: text)",
    )
    p.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Samples per shard for WebDataset and MDS (default: {SHARD_SIZE})",
    )
    return p.parse_args()


def main():
    args = parse_args()

    t_start = time.time()
    print(f"\nllm-data-pipeline-bench — dataset preparation")
    print(f"Output dir : {args.output_dir.resolve()}")
    print(f"Formats    : {args.formats}")
    print(f"Modality   : {args.modality}")

    # 1. Load downloaded dataset
    ds = download_dataset(modality=args.modality)

    # 2. Convert to requested formats
    if "parquet" in args.formats:
        save_parquet(ds, args.output_dir, modality=args.modality)

    if "webdataset" in args.formats:
        save_webdataset(ds, args.output_dir, modality=args.modality, shard_size=args.shard_size)

    if "mds" in args.formats:
        save_mds(ds, args.output_dir, modality=args.modality, shard_size=args.shard_size)

    # 4. Summary
    print_usage_summary(args.output_dir)
    print(f"Total time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
