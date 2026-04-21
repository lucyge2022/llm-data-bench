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
    "shm_used_gb":    8.3,    # mean /dev/shm usage during epoch (Ray object store)
    "shm_peak_gb":    11.2,   # peak /dev/shm usage during epoch
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
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths — match what prepare_datasets.py wrote
# ---------------------------------------------------------------------------

DATA_ROOT = Path("/home/user/llm-data-bench/data-output")
RESULTS_DIR = Path("/home/user/llm-data-bench/results/raw")

TEXT_PARQUET_DIR  = DATA_ROOT / "text/parquet"
TEXT_WDS_DIR      = DATA_ROOT / "text/webdataset"
TEXT_MDS_DIR      = DATA_ROOT / "text/mds"

IMAGE_PARQUET_DIR = DATA_ROOT / "images/parquet"
IMAGE_WDS_DIR     = DATA_ROOT / "images/webdataset"
IMAGE_MDS_DIR     = DATA_ROOT / "images/mds"

IMAGE_TEXT_PARQUET_DIR = DATA_ROOT / "image+text/parquet"
IMAGE_TEXT_WDS_DIR     = DATA_ROOT / "image+text/webdataset"
IMAGE_TEXT_MDS_DIR     = DATA_ROOT / "image+text/mds"

# ---------------------------------------------------------------------------
# GPU utilisation poller  (background thread, nvidia-smi)
# ---------------------------------------------------------------------------

class GPUPoller:
    """
    Polls GPU utilisation, GPU memory, CPU utilisation and RAM every interval
    seconds in a background thread.
    Call .start() before the training loop, .stop() after.
    .summary() returns mean/peak stats for all four metrics.
    Optionally pushes live readings to a Prometheus Pushgateway.
    """

    def __init__(self, device_idx: int = 0, interval: float = 1.0,
                 pushgateway: Optional[str] = None, job_labels: Optional[Dict] = None):
        self.device_idx   = device_idx
        self.interval     = interval
        self._pushgateway = pushgateway
        self._job_labels  = job_labels or {}
        self._gpu_utils:     list = []
        self._gpu_mems:      list = []
        self._cpu_utils: list = []
        self._ram_used:  list = []
        self._shm_used:  list = []

        import psutil
        self._proc = psutil.Process(os.getpid())
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._nvml_ok = False
        self._prom_ok = False
        self._init_nvml()
        self._init_prom()

    def _init_prom(self):
        if not self._pushgateway:
            return
        try:
            from prometheus_client import CollectorRegistry, Gauge
            self._prom_registry = CollectorRegistry()
            label_names = list(self._job_labels.keys())
            self._g_util     = Gauge("gpu_util_pct",  "GPU utilisation %",    label_names, registry=self._prom_registry)
            self._g_mem      = Gauge("gpu_mem_gb",    "GPU memory used GB",   label_names, registry=self._prom_registry)
            self._g_cpu_util = Gauge("cpu_util_pct",  "CPU utilisation %",    label_names, registry=self._prom_registry)
            self._g_ram_used = Gauge("ram_used_gb",   "RAM used GB",          label_names, registry=self._prom_registry)
            self._g_shm_used = Gauge("shm_used_gb",   "SHM used GB",          label_names, registry=self._prom_registry)
            self._prom_ok = True
        except ImportError:
            pass

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

    @staticmethod
    def _query_shm() -> float:
        """Return /dev/shm used space in GB."""
        import shutil
        try:
            usage = shutil.disk_usage("/dev/shm")
            return usage.used / 1e9
        except Exception:
            return 0.0

    def _poll(self):
        while not self._stop.is_set():
            gpu_util, gpu_mem = self._query()
            cpu_util = self._proc.cpu_percent(interval=None)
            ram_used = self._proc.memory_info().rss / 1e9
            shm_used = self._query_shm()
            if gpu_util is not None:
                self._gpu_utils.append(gpu_util)
                self._gpu_mems.append(gpu_mem)
            self._cpu_utils.append(cpu_util)
            self._ram_used.append(ram_used)
            self._shm_used.append(shm_used)
            self._push(gpu_util, gpu_mem, cpu_util, ram_used, shm_used)
            time.sleep(self.interval)

    def _push(self, gpu_util: Optional[float], gpu_mem: Optional[float],
              cpu_util: float, ram_used: float, shm_used: float):
        if not self._prom_ok:
            print(f"[ERROR] Prometheus not initialized")
            return
        try:
            from prometheus_client import push_to_gateway
            import math
            label_values = list(self._job_labels.values())
            self._g_util.labels(*label_values).set(gpu_util if gpu_util is not None else math.nan)
            self._g_mem.labels(*label_values).set(gpu_mem if gpu_mem is not None else math.nan)
            self._g_cpu_util.labels(*label_values).set(cpu_util)
            self._g_ram_used.labels(*label_values).set(ram_used)
            self._g_shm_used.labels(*label_values).set(shm_used)
            push_to_gateway(self._pushgateway, job="llm_data_bench", registry=self._prom_registry)
        except Exception as e:
            print(f"[ERROR] Exception in _push:{e}")
            pass

    def __del__(self):
        if getattr(self, "_prom_ok", False):
            try:
                from prometheus_client import push_to_gateway
                import math
                label_values = list(self._job_labels.values())
                self._g_util.labels(*label_values).set(math.nan)
                self._g_mem.labels(*label_values).set(math.nan)
                self._g_cpu_util.labels(*label_values).set(math.nan)
                self._g_ram_used.labels(*label_values).set(math.nan)
                push_to_gateway(self._pushgateway, job="llm_data_bench", registry=self._prom_registry)
            except Exception:
                pass

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
        self._gpu_utils.clear()
        self._gpu_mems.clear()
        self._cpu_utils.clear()
        self._ram_used.clear()
        self._shm_used.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def summary(self) -> Dict[str, float]:
        import statistics
        result = {
            "gpu_util_pct": 0.0, "gpu_mem_gb": 0.0, "gpu_util_min": 0.0,
            "cpu_util_pct": 0.0, "ram_used_gb": 0.0,
            "shm_used_gb": 0.0, "shm_peak_gb": 0.0,
        }
        if self._gpu_utils:
            result["gpu_util_pct"] = round(statistics.mean(self._gpu_utils), 2)
            result["gpu_util_min"] = round(min(self._gpu_utils), 2)
            result["gpu_mem_gb"]   = round(max(self._gpu_mems), 3)
        if self._cpu_utils:
            result["cpu_util_pct"] = round(statistics.mean(self._cpu_utils), 2)
        if self._ram_used:
            result["ram_used_gb"]  = round(max(self._ram_used), 3)
        if self._shm_used:
            result["shm_used_gb"]  = round(statistics.mean(self._shm_used), 3)
            result["shm_peak_gb"]  = round(max(self._shm_used), 3)
        return result


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

def make_multimodal_model(device: torch.device) -> nn.Module:
    """Multimodal model — forward pass only, no gradient."""
    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        print(f"  Model: CLIP (ViT-Large/14)  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        return model
    except ImportError as e:
        print(f"  Model: CLIP (ViT-Large/14) not installed: {e}")
        raise e

# ---------------------------------------------------------------------------
# Dummy forward passes
# ---------------------------------------------------------------------------

@torch.no_grad()
def text_forward(model: nn.Module, batch, device: torch.device) -> int:
    """
    Tokenise text batch and run GPT-2 forward.
    Returns number of samples processed.
    """
    texts = _extract_text(batch) # texts [=] a list of text strings
    if not texts:
        return 0
    # simple char-level tokenisation — avoids transformers tokenizer overhead
    # (we want to stress the DATA LOADER, not the tokenizer)
    max_len = 128
    ids = torch.zeros(len(texts), max_len, dtype=torch.long, device=device) # ids [=] a tensor of shape (len(texts), max_len) if device is cuda, then this tensor already on the GPU VRAM
    # print(f"[DEBUG] ids shape:{ids.shape}, ids residence device:{ids.device}")
    for i, t in enumerate(texts):
        chars = [ord(c) % 50257 for c in t[:max_len]] # we simply map each character to a number between 0 and 50257, in actuality, we should use a more sophisticated tokenizer
        ids[i, :len(chars)] = torch.tensor(chars, dtype=torch.long) # this is thru H2D copy (host to device) AKA cpu RAM to gpu VRAM copy thru PCIe
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
def image_forward(model: nn.Module, batch, device: torch.device, metrics_data_points: Dict) -> int:
    """
    Decode image batch and run ResNet-18 forward.
    Returns number of samples processed.
    """ 
    t_extract_images_start = time.perf_counter()
    images = _extract_images(batch, device, metrics_data_points)
    t_extract_images_end = time.perf_counter()
    metrics_data_points["t_extract_images_time"] = t_extract_images_end - t_extract_images_start
    if images is None or images.shape[0] == 0:
        return 0
    model(images)
    return len(images)


@torch.no_grad()
def multimodal_forward(model: nn.Module, batch, device: torch.device, metrics_data_points: Dict) -> int:
    """
    Decode image and text batch and run CLIP forward.
    Returns number of samples processed.
    """
    from transformers import CLIPProcessor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Extract images
    t_extract_images_start = time.perf_counter()
    images = _extract_images(batch, device, metrics_data_points)
    t_extract_images_end = time.perf_counter()
    metrics_data_points["t_extract_images_time"] = t_extract_images_end - t_extract_images_start
    print(f"[DEBUG] len images:{len(images)} type images:{type(images[0])}")
    
    # Extract text
    texts = _extract_text(batch)
    print(f"[DEBUG] len texts:{len(texts)} type texts:{type(texts[0])}")
    
    if images is None or images.shape[0] == 0 or \
        texts is None or len(texts) == 0:
        return 0
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    model(**inputs)
    return len(images)


# ---------------------------------------------------------------------------
# Batch extraction helpers  (each loader returns different formats)
# ---------------------------------------------------------------------------

def _extract_text(batch) -> list:
    """Extract list of text strings from any loader's batch format."""
    # WebDataset
    if isinstance(batch, (tuple, list)):
        # for Image+Text multimodal dataset with text list
        if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
            return [text_str.decode() if isinstance(text_str, bytes) else text_str for text_str in batch[1]]
        # for Text-only dataset
        for item in batch:
            if isinstance(item, (list, tuple)) and len(item) > 0 \
                    and isinstance(item[0], (str, bytes)):
                return [t.decode() if isinstance(t, bytes) else t for t in item]
        return []
    # dict (Ray Data, MosaicML streaming)
    for key in ("text", "txt", "__text__"):
        if key in batch:
            val = batch[key]
            if isinstance(val, (list, tuple)):
                return [v.decode() if isinstance(v, bytes) else str(v) for v in val]
            if hasattr(val, "tolist"):  # numpy array column
                return [str(v) for v in val.tolist()]
    return []

def _extract_images(batch, device: torch.device, metrics_data_points: Dict) -> Optional[torch.Tensor]:
    """Extract [B, 3, H, W] float tensor from any loader's batch format.

    Fast path (worker-decoded):
      Workers have already decoded + resized to uint8 [3,224,224] numpy arrays.
      DataLoader/WDS batching stacks them into [B,3,224,224].
      We just normalise to float32 and move to device — no PIL work here.

    Slow path (fallback):
      Raw bytes or variable-size arrays — decode + resize in main process.
      Used when num_workers=0 and no upstream .map() decode step.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image

    # ── Fast path: images pre-decoded by workers into stacked [B,3,H,W] ──────
    # WDS:       batch = (numpy [B,3,224,224], ...)  or  (tensor [B,3,224,224], ...)
    # Streaming: batch = {"jpeg_bytes": tensor [B,3,224,224], ...}
    # Ray:       batch = {"jpeg_bytes": numpy  [B,3,224,224], ...}
    if isinstance(batch, (tuple, list)):
        for item in batch:
            if isinstance(item, (torch.Tensor, np.ndarray)) \
                    and hasattr(item, "ndim") and item.ndim == 4 and item.shape[1] == 3:
                t = item if isinstance(item, torch.Tensor) else torch.from_numpy(item)
                return (t.float() / 255.0).to(device)
    elif isinstance(batch, dict):
        for key in ("jpeg_bytes", "image", "img", "pixel_values", "jpg"):
            if key not in batch:
                continue
            val = batch[key]
            if isinstance(val, (torch.Tensor, np.ndarray)) \
                    and hasattr(val, "ndim") and val.ndim == 4 and val.shape[1] == 3:
                t = val if isinstance(val, torch.Tensor) else torch.from_numpy(val)
                return (t.float() / 255.0).to(device)
            break  # key found but not pre-decoded — fall through to slow path

    # ── Slow path: raw bytes / variable-size arrays ───────────────────────────
    def _decode_one(item):
        if isinstance(item, bytes):
            import io
            return Image.open(io.BytesIO(item)).convert("RGB")
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        return None

    imgs = []

    t_decode_time = 0.0
    t_resize_time = 0.0

    if isinstance(batch, (tuple, list)):
        # WebDataset batch is a tuple of per-field lists/arrays:
        # Case 1:  image-only(.decode('rgb8') applied upstream, tensor as output):       (numpy[B,H,W,3],)
        # Case 2:  image-only(no .decode('rgb8') applied upstream, raw bytes as output): ([bytes,...],)
        # Case 3:  image+text:     batch is a tuple of (list_of_text_with_batch_size, list_of_images_with_batch_size)) 
        for item in batch:
            # Skip text lists — strings/bytes that are captions, not image data
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], str): # Case 3 with text list
                continue
            # Raw JPEG bytes path (no .decode() upstream)
            if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], bytes): # Case 2 with raw bytes list or Case 3 with image list
                for b in item:
                    t_decode_st = time.perf_counter()
                    pil = _decode_one(b)
                    t_decode_time += time.perf_counter() - t_decode_st
                    if pil:
                        t_resize_start = time.perf_counter()
                        imgs.append(TF.to_tensor(TF.resize(pil, [224, 224])))
                        t_resize_time += time.perf_counter() - t_resize_start
                break
            # [LUCY] with .decode('rgb8') applied upstream, faster than raw bytes
            elif isinstance(item, (torch.Tensor, np.ndarray)) and item.ndim >= 3: # Case 1 with tensor list
                # .decode('rgb8') applied upstream means the batch is already a tensor of shape [B, H, W, C]
                # if not, its a tensor of shape [B, C, H, W] B is the batch size, C is the number of channels, H is the height, W is the width, each number in the tensor is a num (e.g if its RGB, then its [0, 255])
                t = ( item.float() if isinstance(item, torch.Tensor) else torch.from_numpy(item).float() ) / 255.0
                '''
                item.ndim == 2   # [H, W]          — grayscale, no channel dim
                 #  e.g. (32, 32)

                item.ndim == 3   # [H, W, C]       — single image with channels
                 #  e.g. (32, 32, 3)
                 #             ↑
                 #             3 = RGB (Red, Green, Blue channels)

                t.ndim == 4   # [B, H, W, C]    — batch of images
                 #  e.g. (64, 32, 32, 3)
                 #             ↑
                 #             3 = RGB channels, same meaning
                '''
                if t.ndim == 4:       # [B, H, W, C] → split into individual images
                    t = t.permute(0, 3, 1, 2)   # → [B, C, H, W]
                    for i in range(t.shape[0]):
                        # t[i] shape: [C, H, W] = [3,64,64]
                        t_resize_start = time.perf_counter()
                        imgs.append(TF.resize(t[i], [224, 224]))
                        t_resize_time += time.perf_counter() - t_resize_start
                break
    elif isinstance(batch, dict):
        # Ray Data / MosaicML
        for key in ("jpeg_bytes", "image", "img", "pixel_values", "jpg"):
            if key in batch:
                val = batch[key]
                items = val.tolist() if hasattr(val, "tolist") else list(val)
                for item in items:
                    pil = _decode_one(item['bytes'] if isinstance(item, dict) else item)
                    if pil:
                        imgs.append(TF.to_tensor(TF.resize(pil, [224, 224])))
                break

    if not imgs:
        return None
    metrics_data_points["t_decode_time"] = t_decode_time
    metrics_data_points["t_resize_time"] = t_resize_time
    return torch.stack(imgs).to(device)


# ---------------------------------------------------------------------------
# Worker-safe decode helpers
# These are top-level (not nested) so they can be pickled by DataLoader workers
# and Ray actor processes.
# ---------------------------------------------------------------------------

def _decode_resize_bytes(raw) -> np.ndarray:
    """Decode JPEG/PNG bytes or PIL Image → uint8 numpy [3, 224, 224] (C,H,W)."""
    import io as _io
    from PIL import Image as _Image
    if isinstance(raw, _Image.Image):
        pil = raw.convert("RGB")
    else:
        pil = _Image.open(_io.BytesIO(bytes(raw))).convert("RGB")
    pil = pil.resize((224, 224), _Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8).transpose(2, 0, 1)  # [3, 224, 224]


def _wds_decode_sample(sample: dict) -> dict:
    """WebDataset .map() transform — runs inside DataLoader worker processes.

    Input sample["jpg"] may be:
      • numpy [H,W,3] uint8  — produced by upstream .decode('rgb8')
      • raw bytes            — no upstream .decode()
    Output sample["jpg"] = uint8 numpy [3, 224, 224]
    """
    import torchvision.transforms.functional as _TF
    for key in ("jpg", "jpeg", "png", "img"):
        if key not in sample:
            continue
        raw = sample[key]
        if isinstance(raw, np.ndarray):
            # Already decoded by .decode('rgb8') → [H,W,3] uint8 numpy
            t = torch.from_numpy(raw).permute(2, 0, 1)   # [3,H,W]
            sample[key] = np.asarray(
                _TF.resize(t, [224, 224]).numpy(), dtype=np.uint8
            )
        else:
            # Raw JPEG bytes — decode + resize
            sample[key] = _decode_resize_bytes(raw)
        break
    return sample


def _ray_decode_row(row: dict) -> dict:
    """Ray Data .map() transform — runs inside Ray actor pool."""
    for key in ("jpeg_bytes", "image", "img", "jpg"):
        if key not in row:
            continue
        raw = row[key]
        if isinstance(raw, dict):           # HF Image feature dict
            raw = raw.get("bytes") or raw.get("path")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            row[key] = _decode_resize_bytes(raw)
        break
    return row


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
            .to_tuple("txt")
            .batched(batch_size, partial=True)
        )
    elif dataset == "images":
        shards = sorted(IMAGE_WDS_DIR.glob("*.tar"))
        ds = (
            wds.WebDataset(sorted(str(s) for s in shards), shardshuffle=True)
            .shuffle(500)
            .decode('rgb8')           # JPEG → numpy [H,W,3] uint8  (in worker)
            .map(_wds_decode_sample)  # resize → numpy [3,224,224]  (in worker)
            .to_tuple("jpg")
            .batched(batch_size, partial=True)
        )
    elif dataset == "image+text":
        shards = sorted(IMAGE_TEXT_WDS_DIR.glob("*.tar"))
        ds = (
            wds.WebDataset(sorted(str(s) for s in shards), shardshuffle=True)
            .shuffle(1000)
            # No .decode() upstream — _wds_decode_sample handles raw bytes directly.
            # After .map(), all images are [3,224,224] so .batched() can stack them.
            .map(_wds_decode_sample)  # decode + resize → numpy [3,224,224]  (in worker)
            .to_tuple("jpg", "txt")
            .batched(batch_size, partial=True)
        )

    loader = wds.WebLoader(ds, batch_size=None, num_workers=num_workers,
                           pin_memory=True)
    return loader


class _StreamingDecodeDataset:
    """Wraps StreamingDataset and decodes images in __getitem__.

    Because __getitem__ is called inside DataLoader worker processes
    (when num_workers > 0), the decode+resize runs in parallel, fully
    overlapping with the GPU forward pass on the previous batch.
    """
    _IMAGE_KEYS = ("jpeg_bytes", "image", "img", "jpg")

    def __init__(self, mds_dir: str, **kwargs):
        from streaming import StreamingDataset
        self._ds = StreamingDataset(local=mds_dir, **kwargs)

    # Delegate all dataset protocol methods to the inner StreamingDataset
    def __len__(self):      return len(self._ds)
    def __iter__(self):     return iter(self._ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self._ds[idx] # directly return the idx-th sample from the StreamingDataset
        for key in self._IMAGE_KEYS:
            if key not in sample:
                continue
            raw = sample[key]
            if isinstance(raw, bytes):
                # Decode + resize in the worker → returns uint8 [3,224,224]
                # DataLoader's default collate will stack these into [B,3,224,224]
                sample[key] = _decode_resize_bytes(raw)
            break
        return sample


def make_streaming_loader(dataset: str, batch_size: int, num_workers: int):
    from torch.utils.data import DataLoader

    mds_dir = ""
    if dataset == "text":
        mds_dir = TEXT_MDS_DIR
    elif dataset == "images":
        mds_dir = IMAGE_MDS_DIR
    elif dataset == "image+text":
        mds_dir = IMAGE_TEXT_MDS_DIR
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    kwargs = dict(shuffle=True, shuffle_algo="py1s", batch_size=batch_size)
    ds = _StreamingDecodeDataset(str(mds_dir), **kwargs)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        pin_memory=True, drop_last=True)
    return loader


def make_ray_loader(dataset: str, batch_size: int, num_workers: int):
    import ray
    import ray.data

    if not ray.is_initialized():
        ray.init(
            num_cpus=max(num_workers, 2),
            ignore_reinit_error=True,
            log_to_driver=False,
        )

    parquet_dir = ""
    if dataset == "text":
        parquet_dir = TEXT_PARQUET_DIR
    elif dataset == "images":
        parquet_dir = IMAGE_PARQUET_DIR
    elif dataset == "image+text":
        parquet_dir = IMAGE_TEXT_PARQUET_DIR
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    ds = ray.data.read_parquet(parquet_dir, parallelism=num_workers) # parallelism determines num of blocks for entire dataset

    # 1. first shuffle the data
    ds = ds.random_shuffle()         # total bytes (decompressed, in memory)

    # 2. then decode the images (order of shuffle/map matters!)
    # Decode + resize images in Ray actor pool before shuffling/materialising.
    # _ray_decode_row runs across Ray's parallel task pool — completely offloaded
    # from the main training process.
    if dataset in ("images", "image+text"):
        ds = ds.map(
            _ray_decode_row,
            compute=ray.data.TaskPoolStrategy(size=num_workers),
            )

    return ds.iter_batches(
        batch_size=batch_size,
        prefetch_batches=4,
    )


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
    profile_dir: Optional[Path] = None,
) -> Dict:
    """Run one epoch, return metrics dict."""

    poller.start()
    t_epoch_start = time.perf_counter()

    total_samples  = 0
    total_batches  = 0
    t_data_wait    = 0.0   # time spent waiting for next batch (stall time)
    t_extract_images = 0.0 # time spent extracting images
    t_decode_time = 0.0
    t_resize_time = 0.0
    metrics_data_points = {}

    t_prev = time.perf_counter()

    def _run_loop(prof=None):
        nonlocal total_samples, total_batches, t_data_wait, t_prev, t_extract_images, t_decode_time, t_resize_time
        with tqdm(desc=f"  Epoch {epoch}", unit="batch", leave=False) as pbar:
            for i, batch in enumerate(loader): # only data batch, no labels or metadata, just for simple dummy forward pass
                t_got_batch = time.perf_counter()
                t_data_wait += t_got_batch - t_prev

                if dataset == "text":
                    n = text_forward(model, batch, device)
                elif dataset == "images":
                    n = image_forward(model, batch, device, metrics_data_points) # metrics dict to record image extraction related metrics
                elif dataset == "image+text":
                    n = multimodal_forward(model, batch, device, metrics_data_points)
                total_samples += n
                total_batches += 1

                t_prev = time.perf_counter()
                pbar.update(1)
                pbar.set_postfix(samples=total_samples,
                                 sps=f"{total_samples/(t_prev-t_epoch_start):.0f}")

                if prof:
                    prof.step()

                if max_batches and total_batches >= max_batches:
                    break

    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profile_dir / f"epoch{epoch}.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,   # stack traces balloon file size significantly
            schedule=torch.profiler.schedule(
                wait=2,         # skip first 2 batches (warmup noise)
                warmup=1,       # 1 warmup batch (traced but discarded)
                active=5,       # capture only 5 batches
                repeat=1,
            ),
            # profile_memory=True,  uncomment to profile memory usage
        ) as prof:
            _run_loop(prof)
        prof.export_chrome_trace(str(trace_path))
        print(f"  Profiler trace → {trace_path}  (open in chrome://tracing)")
    else:
        _run_loop()

    t_elapsed = time.perf_counter() - t_epoch_start
    poller.stop()
    gpu_stats = poller.summary()

    t_compute = t_elapsed - t_data_wait
    data_stall_pct = round(100.0 * t_data_wait / t_elapsed, 2) if t_elapsed > 0 else 0.0
    extract_images_pct = round(100.0 * t_extract_images / t_elapsed, 2) if t_elapsed > 0 else 0.0
    decode_time_pct = round(100.0 * t_decode_time / t_elapsed, 2) if t_elapsed > 0 else 0.0
    resize_time_pct = round(100.0 * t_resize_time / t_elapsed, 2) if t_elapsed > 0 else 0.0
    print(f"[DEBUG] decode_time_pct:{decode_time_pct}, resize_time_pct:{resize_time_pct}")

    return {
        "epoch":           epoch,
        "batches":         total_batches,
        "total_samples":   total_samples,
        "elapsed_sec":     round(t_elapsed, 3),
        "samples_per_sec": round(total_samples / t_elapsed, 2) if t_elapsed > 0 else 0,
        "data_stall_pct":  data_stall_pct,
        "extract_images_pct": extract_images_pct,
        **gpu_stats,
    }


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    loader_name:  str,
    dataset:      str,
    batch_size:   int,
    num_workers:  int,
    epochs:       int,
    max_batches:  Optional[int],
    output_dir:   Path,
    pushgateway:  Optional[str] = None,
    profile:      bool = False,
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
    model = None
    if dataset == "text":
        model = make_text_model(device)
    elif dataset == "images":
        model = make_image_model(device)
    elif dataset == "image+text":
        model = make_multimodal_model(device)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # Build loader
    print(f"  Building {loader_name} loader...")
    loader = LOADER_FACTORIES[loader_name](dataset, batch_size, num_workers)

    poller = GPUPoller(
        device_idx=0,
        interval=1.0,
        pushgateway=pushgateway,
        job_labels={"loader": loader_name, "dataset": dataset, "batch_size": str(batch_size)},
    )

    all_results = []

    for epoch in range(epochs):
        print(f"\n  Epoch {epoch+1}/{epochs}")

        # Rebuild loader each epoch (important for shuffle testing)
        if epoch > 0:
            loader = LOADER_FACTORIES[loader_name](dataset, batch_size, num_workers)

        profile_dir = (output_dir / "profiles" /
                       f"{loader_name}_{dataset}_bs{batch_size}") if profile else None
        metrics = run_epoch(
            loader=loader,
            model=model,
            dataset=dataset,
            device=device,
            epoch=epoch,
            max_batches=max_batches,
            poller=poller,
            profile_dir=profile_dir,
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
        print(f"  CPU util     : {metrics['cpu_util_pct']:.1f}%")
        print(f"  RAM used     : {metrics['ram_used_gb']:.2f} GB")
        print(f"  /dev/shm     : {metrics['shm_used_gb']:.2f} GB avg  "
              f"(peak {metrics['shm_peak_gb']:.2f} GB)")
        print(f"  Data stall   : {metrics['data_stall_pct']:.1f}%")
        if dataset == "images":
            print(f"  Extract images: {metrics['extract_images_pct']:.1f}%")
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
                   choices=["text", "images", "image+text"],
                   help="Dataset to use")
    p.add_argument("--batch-size",  type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    '''
    num-workers
    Loader	    What num_workers controls:
    WebDataset	PyTorch DataLoader worker processes:
    Streaming	PyTorch DataLoader worker processes:
    Ray	        CPU budget for Ray's internal execution pool:
    '''
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Cap batches per epoch (for smoke testing)")
    p.add_argument("--output-dir",  type=Path, default=RESULTS_DIR)
    p.add_argument("--all",         action="store_true",
                   help="Run full matrix: all loaders x all datasets")
    p.add_argument("--smoke-test",  action="store_true",
                   help="Quick run: 20 batches, 1 epoch, webdataset+text only")
    p.add_argument("--pushgateway", type=str, default="localhost:9091",
                   help="Prometheus Pushgateway address e.g. localhost:9091")
    p.add_argument("--profile", action="store_true",
                   help="Enable torch.profiler — writes Chrome trace JSON per epoch to output_dir/profiles/")
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
            pushgateway=args.pushgateway,
            profile=args.profile,
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
                        pushgateway=args.pushgateway,
                        profile=args.profile,
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
        pushgateway=args.pushgateway,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()