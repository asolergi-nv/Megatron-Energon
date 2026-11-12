<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# perf(io): reuse local index readers and descriptor sizes

[Code branch](https://github.com/asolergi-nv/Megatron-Energon/tree/review/startup-07-index-reader-reuse) · [Review changes](https://github.com/asolergi-nv/Megatron-Energon/compare/c78d47025cd2c3617fc508007353095ce9c883bf...41c322db694bb56b7d205a472c1223e03026368c)

Implementation commit: `41c322db694bb56b7d205a472c1223e03026368c`.

- [ ] I, a human, have personally reviewed every line of this PR and take responsibility for it.

## What does this PR do?

Obtain an eligible local index stream's size from its open descriptor and reuse bounded index readers for repeated offsets in the same file. Other storage backends keep the original metadata lookup.

## Dependencies

No dependency on another pending optimization. The branch starts from the current target base.

## Performance evidence

3 fresh-process trials per configuration; first next() 42.60s -> 34.84s; setup + loader 50.96s -> 42.56s.

These are adjacent-stage measurements from the previously validated, pinned standalone-loader implementation. They are not fresh timings of this branch on the updated target base. Backend comparisons retain all other selected optimizations. No large-scale extrapolation is claimed.

Worker index metadata queries: 12,478 → 0; index reader creations: 12,478 → 2,548; storage-client opens: 15,301 → 5,371. Descriptor fstat calls become 2,548. These are instrumented API counts, not filesystem RPC counts; overlapping categories must not be added.

## Validation

- `just fix`: PASS.
- `just check`: PASS.
- `bwrap --bind / / --dev-bind /dev /dev --tmpfs /tmp --unshare-net just test`: PASS (Ran 227 tests in 646.926s; OK).
- `just check-commits`: PASS (one signed-off Conventional Commit).

Earlier standalone-loader evidence: exact tokens, masks, lengths, source identities and pixels across the selected stack; 64-batch comparisons and fresh-process dataloader restore checks for the final stack. That evidence does not establish full training-process interruption/restart correctness, nor replace branch CI after integration changes.

## Review decisions before merging

Native Energon patch. Only exact eligible MSC POSIX streams without a metadata provider use descriptor sizing. Other streams retain their original authoritative size lookup. No storage-client library change is proposed.

## Linked issue

No new issue has been opened. Link the relevant issue if repository contribution policy requires one before submission.

## Checklist

- [ ] No code was copied from other repositories or projects; the whole change can be contributed under this project's BSD-3-Clause license (human attestation).
- [ ] Tests cover the final agreed API and lifecycle.
- [x] Documentation in `docs/source` is updated.
- [x] `just check` passes.
- [x] All commits follow Conventional Commits and are signed off.
- [ ] Required CI passes on the final submitted revision.
