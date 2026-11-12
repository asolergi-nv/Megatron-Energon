<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# perf(index): reuse readers and avoid redundant metadata queries

[Draft PR #348](https://github.com/NVIDIA/Megatron-Energon/pull/348) · [Code branch](https://github.com/asolergi-nv/Megatron-Energon/tree/review/startup-07-index-reader-reuse)

Implementation commit: `41c322db694bb56b7d205a472c1223e03026368c`.

- [ ] I, a human, have personally reviewed every line of this PR and take responsibility for it.

## What does this PR do?

Reduce repeated opens and metadata queries when reading JSONL and tar offset indexes.

An offset-cache miss can currently open another reader even when the cache already holds one for the same index file. Reuse that reader instead: the existing lookup logic seeks to the requested offset and refreshes its lookahead. JSONL readers reuse an entry from their single-file cache; tar readers reuse an entry only when its file matches. The existing cache bound and reader cleanup remain in place.

Index readers also obtain the file length from the descriptor they have just opened, using `fstat`, when the stream is an exact MSC POSIX file without a metadata provider. Other streams retain their authoritative path-based size lookup. A failed size lookup closes the newly opened stream.

## Performance

- First-batch wait: **42.60 → 34.84 seconds**, about **7.76 seconds saved (18%)**. Setup plus loader time to the first batch: **50.96 → 42.56 seconds**, about **8.40 seconds saved**.
- Worker storage-client opens: **15,301 → 5,371**. Index path-metadata queries: **12,478 → 0**, with **2,548 descriptor-size checks** instead.

Timings are medians of three fresh-process runs of the validated downstream prototype, not new measurements of this public branch. I/O figures are separately instrumented API calls, not filesystem RPC counts.

## Linked issue

Related to [#331](https://github.com/NVIDIA/Megatron-Energon/issues/331).

## Testing

Validation of this implementation commit:

- `just fix`, `just check`, and `just check-commits`: passed.
- `bwrap --bind / / --dev-bind /dev /dev --tmpfs /tmp --unshare-net just test`: **227 tests passed**. Private temporary and network namespaces isolate the upstream test fixtures.
- The added test exercises random and interleaved offsets, cache bounds, reader closure, metadata-provider and non-POSIX fallbacks, and sizing an open file after its path is replaced.

## Checklist

- [ ] No code was copied from other repositories or projects; the whole change can be contributed under this project's BSD-3-Clause license (human attestation).
- [x] Tests cover the change.
- [x] Documentation in `docs/source` is updated.
- [x] `just check` passes.
- [x] All commits follow Conventional Commits and are signed off.
