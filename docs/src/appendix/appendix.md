# Appendix — MDS Format & py1s Shuffle Deep Dive

---

## A.1 MDS Shard Structure

When `prepare_dataset.py` runs `MDSWriter`, it produces the following layout (5,000 samples/shard by default):

```
mds/
  index.json           ←  manifest: shard sizes, per-sample byte offsets
  shard.00000.mds      ←  samples 0 – 4,999
  shard.00001.mds      ←  samples 5,000 – 9,999
  shard.00002.mds      ←  samples 10,000 – 14,999
  ...
```

```python
# Each .mds shard is a flat binary file:
#   [sample_0_bytes] [sample_1_bytes] ... [sample_4999_bytes]
#   with a front-of-file index for O(1) random access by position
```

> **Key Difference from WebDataset:** `.tar` files require sequential reads from the start of the archive. `.mds` files expose an index for O(1) seek to any sample by global position — enabling efficient random-access patterns like py1s.

## A.2 py1s Shuffle Mechanics

**py1s** = Python implementation · single-pass · shuffle

**Algorithm:**

1. Read `index.json` to obtain global sample indices `[0, 1, 2, ..., N-1]`
2. Shuffle the index array in-place: `[23401, 7832, 41205, 891, ..., 15634]`
3. Partition the shuffled index evenly across `num_workers` DataLoader workers
4. Each worker uses its index slice to seek into `.mds` shards and read sample bytes on demand

```python
# py1s worker assignment example (num_workers=4, N=50,000)
Worker 0 indices: [23401, 7832,  41205, 891,  ...]  # 12,500 samples
Worker 1 indices: [33201, 19832,  5205, 2891, ...]  # 12,500 samples
Worker 2 indices: [43401, 27832, 31205, 1891, ...]  # 12,500 samples
Worker 3 indices: [13401, 37832, 11205, 4891, ...]  # 12,500 samples

# Worker 0 resolves index 23401:
# → shard 4  (23401 // 5000 = 4)
# → position 3401 within shard.00004.mds
# → O(1) seek → read sample bytes → decode
```

**Why memory is only ~200 KB:** The shuffle operates entirely on the integer index array — no sample bytes are moved or copied. Only when a worker needs a specific sample does it seek and read from disk.
