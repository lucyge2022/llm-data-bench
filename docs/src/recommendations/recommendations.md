# 4. Recommendations & Decision Guide

---

## 4.1 When to Use Each Loader

| Scenario | Recommended Loader | Rationale |
|---|---|---|
| Streaming from S3 / HDD / slow NFS | **WebDataset** | Sequential `.tar` reads; no random I/O penalty |
| High-quality shuffle on NVMe / SSD | **MosaicML MDS (py1s)** | True global shuffle at near-zero memory cost; optimal for fast random read storage |
| Multi-node distributed training | **Ray Data** | Native distributed shuffle and execution model across cluster nodes |
| Image/video workloads needing decode offload | **WebDataset** | `.decode()` offloads to worker C libs; clearest GPU utilization gains |
| Pre-tokenized text, large batch sizes | **MosaicML MDS or WebDataset** | Both show lower memory and stall overhead than lazy Ray shuffle |
| Multimodal with variable-resolution images | **Any (with bytes passthrough)** | Disable loader-level decode; manual resize + collate required for all loaders |

## 4.2 Shuffle Strategy Quick Reference

**WebDataset**
Use `shardshuffle=True` + `.shuffle(N)` for cheap local randomization; acceptable for most NLP pre-training where perfect shuffle quality is not critical.

**MosaicML MDS py1s**
Best for single-node NVMe workloads; delivers true global shuffle with negligible memory overhead; degrades on HDD due to random seek pattern.

**Ray Data `random_shuffle`**
Use `.materialize()` before epoch iteration to avoid first-batch latency spike; consider `.randomize_block_order()` for cheaper inter-epoch reshuffling.

## 4.3 Memory Footprint Summary

| Loader | Shuffle Memory Overhead | Notes |
|---|---|---|
| WebDataset | ~50 MB (buffer) | Buffer size controlled by `.shuffle(N)` |
| MosaicML MDS (py1s) | ~200 KB (index array) | Only integer indices stored; sample bytes not loaded until accessed |
| Ray Data (`random_shuffle`) | 1 GB+ (Object Store) | Arrow table immutability forces double-allocation during sort-merge; scales with dataset size |
| Ray Data (pre-materialised) | Same 1 GB+ | Materialization pays the cost upfront; subsequent epochs cheaper with `randomize_block_order()` |
