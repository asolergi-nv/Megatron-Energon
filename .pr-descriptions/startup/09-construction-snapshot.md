<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# perf(metadata): reuse immutable metadata during loader construction

[Draft PR #342](https://github.com/NVIDIA/Megatron-Energon/pull/342) · [Code branch](https://github.com/asolergi-nv/Megatron-Energon/tree/perf/constructor-metadata-snapshot)

Implementation commit: `861048606a27d8203a6bf6b052933e7d4b46242c`.

- [ ] I, a human, have personally reviewed every line of this PR and take responsibility for it.

## What does this PR do?

Let callers reuse immutable metadata while constructing a dataloader, instead of repeatedly opening small configuration files and querying storage metadata.

`MetadataSnapshot.load` reads one package and verifies its checksum and expected recipe identity. Inside `use_metadata_snapshot`, recorded YAML/JSON reads, file checks, file sizes, and JSONL terminal offsets come from that package. Each open gets an independent stream; unrecorded paths use the normal storage backend. Leave the context before starting workers, so sample reads and checkpoint state remain independent.

The package describes a frozen dataset revision. Its checksum verifies the selected package, not whether the source files have since changed; callers must regenerate it when the inputs change. This PR provides replay and documents the format, with a capture helper for tests. Application-level preparation and selection remain with the caller.

## Performance

| Measured phase | Without snapshot | With snapshot | Time saved |
|---|---:|---:|---:|
| Loader construction | 5.76 s | 0.81 s | approximately **4.96 s** |
| Setup plus loader time to first batch | 36.31 s | 31.33 s | approximately **4.97 s** |

Savings are calculated from the unrounded medians of three fresh-process prototype runs. The improvement comes primarily from construction; the worker's first-batch wait remains around 29 seconds. Snapshot loading and verification are included in the startup total; one-time offline preparation is excluded and amortized across later constructions.

In the separately instrumented constructor, metadata opens fell **2,130 → 0**, file checks **3,746 → 0**, and metadata-info queries **904 → 0**. One snapshot-file read remains. These overlapping API counters must not be added together or treated as filesystem RPC counts. The timings describe the validated downstream prototype, not new measurements of this public branch.

## Linked issue

Related to [#331](https://github.com/NVIDIA/Megatron-Energon/issues/331).

## Testing

Validation of this implementation commit:

- `just fix`, `just check`, and `just check-commits`: passed.
- `bwrap --bind / / --dev-bind /dev /dev --tmpfs /tmp --unshare-net just test`: **229 tests passed**. Private temporary and network namespaces isolate the upstream test fixtures.
- Tests cover checksum/identity rejection, independent streams, avoiding recorded storage calls, and restoring normal access after the context exits. The public JSONL loader test checks sample order and checkpoint restoration with zero and two workers.

## Checklist

- [ ] No code was copied from other repositories or projects; the whole change can be contributed under this project's BSD-3-Clause license (human attestation).
- [x] Tests cover the change.
- [x] Documentation in `docs/source` is updated.
- [x] `just check` passes.
- [x] All commits follow Conventional Commits and are signed off.
