# LLM Data Pipeline Benchmark Report

**A Deep Dive into WebDataset · MosaicML Streaming · Ray Data**

*April 2025 · Workloads: Text · Image · Multimodal (CC3M)*

---

This report benchmarks three PyTorch-compatible data loading libraries across text, image, and multimodal workloads. Metrics collected include total training time, sample feeding rate (samples/sec), data stall percentage, RAM usage, GPU utilization, and CPU utilization. Each loader was evaluated at multiple batch sizes over three epochs.

## Libraries Under Test

| Library | Format | Primary Strength |
|---|---|---|
| **WebDataset** | `.tar` shards | Sequential streaming, decode offload |
| **MosaicML StreamingDataset** | `.mds` binary | True global shuffle, NVMe-optimized |
| **Ray Data** | Parquet | Distributed multi-node training |

## Key Findings

**Finding 1 — Ray Data shuffle overhead**
Ray's `random_shuffle()` is lazy: it triggers a sort-merge over Arrow tables in shared Object Store memory on first iteration, pushing RAM to ~3 GB for a 700 MB dataset and causing severe first-batch latency spikes.

**Finding 2 — Pre-materializing Ray resolves the bottleneck**
Using `.random_shuffle().materialize()` before iteration brings Ray's throughput and GPU utilization in line with the other two loaders.

**Finding 3 — WebDataset decode offload delivers 2× GPU utilization**
WebDataset's built-in `.decode()` offloads JPEG decoding to worker subprocesses via Pillow C bindings, delivering roughly **2× higher GPU utilization** and **~50% lower training time** versus in-process decoding.

**Finding 4 — MosaicML MDS py1s: true global shuffle at ~200 KB**
The py1s shuffle achieves true global shuffle quality with only ~200 KB of index memory (vs 1 GB+ for Ray), at the cost of random I/O — well-suited to NVMe, less so to HDD or object storage.
