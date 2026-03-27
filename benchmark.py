"""
benchmark.py  —  Task 1.3: Unified Benchmark Harness
------------------------------------------------------
Benchmarks three data loaders against identical training loops.

Loaders  : webdataset | streaming | ray
Datasets : text (Parquet/tar/mds of C4)  |  images (JPEG/tar/mds of CIFAR-100)
Model    : GPT-2 small (text)  |  ResNet-18 (images)  — forward pass only

Output
------
  results/raw/<loader>_<dataset>_bs<B>_w<W>_epoch<E>.json
  {
    "loader":         "webdataset",
    "dataset":        "text",
    "batch_size":     64,
    "num_workers":    4,
    "epoch":          0,
    "samples_per_sec": 1234.5,
    "gpu_util_pct":   78.3,
    "gpu_mem_gb":     4.2,
    "elapsed_sec":    42.1,
    "batches":        782,
    "total_samples":  50048,
    "data_stall_pct": 12.4,   # % of time GPU was idle waiting for data
    "hostname":       "...",
    "gpu_name":       "Tesla V100-SXM2-16GB",
    "timestamp":      "2026-03-26T10:00:00"
  }

Usage
-----
  # text with webdataset, 3 epochs
  python3 benchmark.py --loader webdataset --dataset text --epochs 3

  # images with ray, custom batch size + workers
  python3 benchmark.py --loader ray --dataset images --batch-size 128 --num-workers 8

  # run all loaders x all datasets (full matrix)
  python3 benchmark.py --all

  # quick smoke test
  python3 benchmark.py --loader webdataset --dataset text --max-batches 20 --epochs 1

Requirements
------------
  pip install torch torchvision transformers webdataset mosaicml-streaming ray[data] \
              nvidia-ml-py3 pillow tqdm
"""

import argparse
import datetime
import json
import os
import platform
import socket
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — match what prepare_datasets.py wrote
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/workspace/llm-data-bench/data")
RESULTS_DIR = Path("/workspace/llm-data-bench/results/raw")

TEXT_PARQUET_DIR  = DATA_ROOT / "text/parquet"
TEXT_WDS_DIR      = DATA_ROOT / "text/webdataset"
TEXT_MDS_DIR      = DATA_ROOT / "text/mds"

IMAGE_PARQUET_DIR = DATA_ROOT / "images/parquet"
IMAGE_WDS_DIR     = DATA_ROOT / "images/webdataset"
IMAGE_MDS_DIR     = DATA_ROOT / "images/mds"

# ---------------------------------------------------------------------------
# GPU utilisation poller  (background thread, nvidia-smi)
# ---------------------------------------------------------------------------

class GPUPoller:
    """
    Polls GPU utilisation and memory every 500ms in a background thread.
    Call .start() before the training loop, .stop() after.
    .summary() returns mean util% and peak mem GB.
    """

    def __init__(self, device_idx: int = 0, interval: float = 0.5):
        self.device_idx = device_idx
        self.interval   = interval
        self._utils: list = []
        self._mems:  list = []
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._nvml_ok = False
        self._init_nvml()

    def _init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_idx)
            self._pynvml = pynvml
            self._nvml_ok = True
        except Exception:
            # fallback: parse nvidia-smi output
            self._nvml_ok = False

    def _poll(self):
        while not self._stop.is_set():
            util, mem = self._query()
            if util is not None:
                self._utils.append(util)
                self._mems.append(mem)
            time.sleep(self.interval)

    def _query(self):
        if self._nvml_ok:
            try:
                rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem   = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return float(rates.gpu), float(mem.used) / 1e9
            except Exception:
                pass
        # fallback: subprocess
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits",
                 f"--id={self.device_idx}"],
                text=True, timeout=2,
            ).strip()
            util_s, mem_s = out.split(",")
            return float(util_s.strip()), float(mem_s.strip()) / 1024
        except Exception:
            return None, None

    def start(self):
        self._utils.clear()
        self._mems.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> Dict[str, float]:
        if not self._utils:
            return {"gpu_util_pct": 0.0, "gpu_mem_gb": 0.0, "gpu_util_min": 0.0}
        import statistics
        return {
            "gpu_util_pct": round(statistics.mean(self._utils), 2),
            "gpu_util_min": round(min(self._utils), 2),
            "gpu_mem_gb":   round(max(self._mems), 3),
        }


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------

def make_text_model(device: torch.device) -> nn.Module:
    """GPT-2 small — forward pass only, no gradient."""
    try:
        from transformers import GPT2Model, GPT2Config
        cfg   = GPT2Config(n_layer=6, n_head=8, n_embd=512)
        model = GPT2Model(cfg).to(device).eval()
        print(f"  Model: GPT-2 small (6L/8H/512D)  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        return model
    except ImportError:
        # fallback: simple transformer
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
            num_layers=4,
        ).to(device).eval()
        print("  Model: fallback TransformerEncoder (transformers not installed)")
        return model


def make_image_model(device: torch.device) -> nn.Module:
    """ResNet-18 — forward pass only, no gradient."""
    try:
        from torchvision.models import resnet18
        model = resnet18(weights=None).to(device).eval()
        print(f"  Model: ResNet-18  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        return model
    except ImportError:
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 100),
        ).to(device).eval()
        print("  Model: fallback mini-CNN (torchvision not installed)")
        return model


# ---------------------------------------------------------------------------
# Dummy forward passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def text_forward(model: nn.Module, batch, device: torch.device) -> int:
    """
    Tokenise text batch and run GPT-2 forward.
    Returns number of samples processed.
    """
    texts = _extract_text(batch)
    if not texts:
        return 0
    # simple char-level tokenisation — avoids transformers tokenizer overhead
    # (we want to stress the DATA LOADER, not the tokenizer)
    max_len = 128
    ids = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
    for i, t in enumerate(texts):
        chars = [ord(c) % 50257 for c in t[:max_len]]
        ids[i, :len(chars)] = torch.tensor(chars, dtype=torch.long)
    try:
        model(ids)
    except Exception:
        # GPT2Model expects input_ids kwarg
        try:
            model(input_ids=ids)
        except Exception:
            pass
    return len(texts)


@torch.no_grad()
def image_forward(model: nn.Module, batch, device: torch.device) -> int:
    """
    Decode image batch and run ResNet-18 forward.
    Returns number of samples processed.
    """
    images = _extract_images(batch, device)
    if images is None or images.shape[0] == 0:
        return 0
    model(images)
    return images.shape[0]


# ---------------------------------------------------------------------------
# Batch extraction helpers  (each loader returns different formats)
# ---------------------------------------------------------------------------

def _extract_text(batch) -> list:
    """Extract list of text strings from any loader's batch format."""
    # WebDataset: tuple of (keys, text_bytes, json_bytes)  or dict
    if isinstance(batch, (tuple, list)):
        for item in batch:
            if isinstance(item, (list, tuple)) and isinstance(item[0], (str, bytes)):
                return [t.decode() if isinstance(t, bytes) else t for t in item]
        return []
    # dict (Ray Data, MosaicML)
    for key in ("text", "txt", "__text__"):
        if key in batch:
            val = batch[key]
            if isinstance(val, (list, tuple)):
                return [v.decode() if isinstance(v, bytes) else str(v) for v in val]
            if hasattr(val, "tolist"):
                return [str(v) for v in val.tolist()]
    return []


def _extract_images(batch, device: torch.device) -> Optional[torch.Tensor]:
    """Extract [B, 3, H, W] float tensor from any loader's batch format."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    def _decode_one(item):
        if isinstance(item, bytes):
            import io
            return Image.open(io.BytesIO(item)).convert("RGB")
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        return None

    imgs = []

    if isinstance(batch, (tuple, list)):
        # WebDataset tuple: (keys, jpg_bytes, json_bytes)
        for item in batch:
            if isinstance(item, (list, tuple)) and isinstance(item[0], bytes):
                for b in item:
                    pil = _decode_one(b)
                    if pil:
                        imgs.append(TF.to_tensor(TF.resize(pil, [224, 224])))
                break
    elif isinstance(batch, dict):
        # Ray Data / MosaicML
        for key in ("jpeg_bytes", "image", "img", "pixel_values"):
            if key in batch:
                val = batch[key]
                items = val.tolist() if hasattr(val, "tolist") else list(val)
                for item in items:
                    pil = _decode_one(item)
                    if pil:
                        imgs.append(TF.to_tensor(TF.resize(pil, [224, 224])))
                break

    if not imgs:
        return None
    return torch.stack(imgs).to(device)


# ---------------------------------------------------------------------------
# Loader factories
# ---------------------------------------------------------------------------

def make_webdataset_loader(dataset: str, batch_size: int, num_workers: int):
    import webdataset as wds

    if dataset == "text":
        pattern = str(TEXT_WDS_DIR / "shard-{000000..999999}.tar")
        shards  = sorted(TEXT_WDS_DIR.glob("*.tar"))
        pattern = f"pipe:cat {' '.join(str(s) for s in shards)}" if shards else pattern
        ds = (
            wds.WebDataset(sorted(str(s) for s in shards), shardshuffle=True)
            .shuffle(1000)
            .decode()
            .to_tuple("txt", "json")
            .batched(batch_size, partial=True)
        )
    else:  # images
        shards = sorted(IMAGE_WDS_DIR.glob("*.tar"))
        ds = (
            wds.WebDataset(sorted(str(s) for s in shards), shardshuffle=True)
            .shuffle(500)
            .decode("rgb8")
            .to_tuple("jpg", "json")
            .batched(batch_size, partial=True)
        )

    loader = wds.WebLoader(ds, batch_size=None, num_workers=num_workers,
                           pin_memory=True)
    return loader


def make_streaming_loader(dataset: str, batch_size: int, num_workers: int):
    from streaming import StreamingDataset
    from torch.utils.data import DataLoader

    mds_dir = str(TEXT_MDS_DIR if dataset == "text" else IMAGE_MDS_DIR)

    ds = StreamingDataset(
        local=mds_dir,
        shuffle=True,
        shuffle_algo="py1s",
        batch_size=batch_size,
    )
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    return loader


def make_ray_loader(dataset: str, batch_size: int, num_workers: int):
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(num_cpus=max(num_workers, 2), ignore_reinit_error=True,
                 log_to_driver=False)

    parquet_dir = str(TEXT_PARQUET_DIR if dataset == "text" else IMAGE_PARQUET_DIR)
    ds = ray.data.read_parquet(parquet_dir)
    ds = ds.random_shuffle()

    # wrap as an iterable that yields batches
    class RayLoader:
        def __init__(self, ray_ds, batch_size):
            self._ds         = ray_ds
            self._batch_size = batch_size

        def __iter__(self):
            return self._ds.iter_batches(
                batch_size=self._batch_size,
                prefetch_batches=4,
            )

    return RayLoader(ds, batch_size)


LOADER_FACTORIES = {
    "webdataset": make_webdataset_loader,
    "streaming":  make_streaming_loader,
    "ray":        make_ray_loader,
}


# ---------------------------------------------------------------------------
# Core benchmark loop
# ---------------------------------------------------------------------------

def run_epoch(
    loader,
    model:       nn.Module,
    dataset:     str,
    device:      torch.device,
    epoch:       int,
    max_batches: Optional[int],
    poller:      GPUPoller,
) -> Dict:
    """Run one epoch, return metrics dict."""

    forward_fn = text_forward if dataset == "text" else image_forward

    poller.start()
    t_epoch_start = time.perf_counter()

    total_samples  = 0
    total_batches  = 0
    t_data_wait    = 0.0   # time spent waiting for next batch (stall time)

    t_prev = time.perf_counter()

    with tqdm(desc=f"  Epoch {epoch}", unit="batch", leave=False) as pbar:
        for batch in loader:
            t_got_batch = time.perf_counter()
            t_data_wait += t_got_batch - t_prev   # time since last batch end

            n = forward_fn(model, batch, device)
            total_samples += n
            total_batches += 1

            t_prev = time.perf_counter()
            pbar.update(1)
            pbar.set_postfix(samples=total_samples,
                             sps=f"{total_samples/(t_prev-t_epoch_start):.0f}")

            if max_batches and total_batches >= max_batches:
                break

    t_elapsed = time.perf_counter() - t_epoch_start
    poller.stop()
    gpu_stats = poller.summary()

    t_compute = t_elapsed - t_data_wait
    data_stall_pct = round(100.0 * t_data_wait / t_elapsed, 2) if t_elapsed > 0 else 0.0

    return {
        "epoch":           epoch,
        "batches":         total_batches,
        "total_samples":   total_samples,
        "elapsed_sec":     round(t_elapsed, 3),
        "samples_per_sec": round(total_samples / t_elapsed, 2) if t_elapsed > 0 else 0,
        "data_stall_pct":  data_stall_pct,
        **gpu_stats,
    }


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    loader_name: str,
    dataset:     str,
    batch_size:  int,
    num_workers: int,
    epochs:      int,
    max_batches: Optional[int],
    output_dir:  Path,
) -> list:

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Loader   : {loader_name}")
    print(f"  Dataset  : {dataset}")
    print(f"  Batch    : {batch_size}  Workers: {num_workers}  Epochs: {epochs}")
    print(sep)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"  Device   : {device}  ({gpu_name})")

    # Build model
    model = make_text_model(device) if dataset == "text" else make_image_model(device)

    # Build loader
    print(f"  Building {loader_name} loader...")
    loader = LOADER_FACTORIES[loader_name](dataset, batch_size, num_workers)

    poller = GPUPoller(device_idx=0)

    all_results = []

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")

        # Rebuild loader each epoch (important for shuffle testing)
        if epoch > 0:
            loader = LOADER_FACTORIES[loader_name](dataset, batch_size, num_workers)

        metrics = run_epoch(
            loader=loader,
            model=model,
            dataset=dataset,
            device=device,
            epoch=epoch,
            max_batches=max_batches,
            poller=poller,
        )

        metrics.update({
            "loader":      loader_name,
            "dataset":     dataset,
            "batch_size":  batch_size,
            "num_workers": num_workers,
            "gpu_name":    gpu_name,
            "hostname":    socket.gethostname(),
            "timestamp":   datetime.datetime.now().isoformat(),
        })

        all_results.append(metrics)

        print(f"  samples/sec  : {metrics['samples_per_sec']:.1f}")
        print(f"  GPU util     : {metrics['gpu_util_pct']:.1f}%  "
              f"(min {metrics['gpu_util_min']:.1f}%)")
        print(f"  GPU mem      : {metrics['gpu_mem_gb']:.2f} GB")
        print(f"  Data stall   : {metrics['data_stall_pct']:.1f}%")
        print(f"  Elapsed      : {metrics['elapsed_sec']:.1f}s  "
              f"({metrics['total_samples']:,} samples)")

        # Save per-epoch result immediately
        output_dir.mkdir(parents=True, exist_ok=True)
        fname = (output_dir /
                 f"{loader_name}_{dataset}"
                 f"_bs{batch_size}_w{num_workers}"
                 f"_epoch{epoch}.json")
        fname.write_text(json.dumps(metrics, indent=2))
        print(f"  Saved → {fname.name}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="llm-data-pipeline-bench — unified data loader benchmark"
    )
    p.add_argument("--loader",
                   choices=["webdataset", "streaming", "ray"],
                   help="Data loader to benchmark")
    p.add_argument("--dataset",
                   choices=["text", "images"],
                   help="Dataset to use")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap batches per epoch (for smoke testing)")
    p.add_argument("--output-dir",  type=Path, default=RESULTS_DIR)
    p.add_argument("--all",         action="store_true",
                   help="Run full matrix: all loaders x all datasets")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick run: 20 batches, 1 epoch, webdataset+text only")
    return p.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        print("SMOKE TEST — 20 batches, 1 epoch, webdataset + text")
        run_benchmark(
            loader_name="webdataset",
            dataset="text",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            epochs=1,
            max_batches=20,
            output_dir=args.output_dir,
        )
        return

    if args.all:
        loaders  = ["webdataset", "streaming", "ray"]
        datasets = ["text", "images"]
        print(f"FULL MATRIX: {len(loaders)} loaders x {len(datasets)} datasets "
              f"x {args.epochs} epochs x 3 batch sizes")
        for loader in loaders:
            for dataset in datasets:
                for bs in [32, 64, 128]:
                    run_benchmark(
                        loader_name=loader,
                        dataset=dataset,
                        batch_size=bs,
                        num_workers=args.num_workers,
                        epochs=args.epochs,
                        max_batches=args.max_batches,
                        output_dir=args.output_dir,
                    )
        return

    if not args.loader or not args.dataset:
        print("ERROR: specify --loader and --dataset, or use --all / --smoke-test")
        parse_args().print_help()
        return

    run_benchmark(
        loader_name=args.loader,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        max_batches=args.max_batches,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()