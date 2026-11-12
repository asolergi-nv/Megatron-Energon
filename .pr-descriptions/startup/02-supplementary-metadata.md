<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# feat(metadata): add opt-in supplementary metadata lookup

[Code branch](https://github.com/asolergi-nv/Megatron-Energon/tree/review/startup-02-supplementary-metadata) · [Review changes](https://github.com/asolergi-nv/Megatron-Energon/compare/c78d47025cd2c3617fc508007353095ce9c883bf...975efcc067010df6a8cf020ca8a0e2cdb3901cc5)

Implementation commit: `975efcc067010df6a8cf020ca8a0e2cdb3901cc5`.

- [ ] I, a human, have personally reviewed every line of this PR and take responsibility for it.

## What does this PR do?

Use a pinned manifest of completed supplementary media-metadata indexes when original prepared metadata is unavailable. Preserve source/key identity, exclude partial indexes and retain the normal fallback for uncovered keys.

## Dependencies

Requires the SQLite URI fix already merged in [Energon #338](https://github.com/NVIDIA/Megatron-Energon/pull/338), included in this branch base. Prepared immutable indexes must be readable by each owner.

## Performance evidence

3 fresh-process trials per configuration; first next() 117.49s -> 117.63s; setup + loader 124.54s -> 124.33s.

These are adjacent-stage measurements from the previously validated, pinned standalone-loader implementation. They are not fresh timings of this branch on the updated target base. Backend comparisons retain all other selected optimizations. No large-scale extrapolation is claimed.

Covered media opens for dimension discovery: 1,549 → 0. Media payload reads during actual sample processing remain. Offline metadata preparation is amortized, not eliminated. No clear first-batch latency improvement was measured at this scale.

## Validation

- `just fix`: PASS.
- `just check`: PASS.
- `bwrap --bind / / --dev-bind /dev /dev --tmpfs /tmp --unshare-net just test`: PASS (Ran 227 tests in 599.118s; OK).
- `just check-commits`: PASS (one signed-off Conventional Commit).

Earlier standalone-loader evidence: exact tokens, masks, lengths, source identities and pixels across the selected stack; 64-batch comparisons and fresh-process dataloader restore checks for the final stack. That evidence does not establish full training-process interruption/restart correctness, nor replace branch CI after integration changes.

## Review decisions before merging

Preserves the tested process-wide lookup adapter. Review the permanent FileStore/config API, initialization in spawned workers, and lifecycle of published immutable indexes before merging. The selected benchmark used complete indexes only; optional observed-key support is retained but was not enabled.

## Linked issue

No new issue has been opened. Link the relevant issue if repository contribution policy requires one before submission.

## Checklist

- [ ] No code was copied from other repositories or projects; the whole change can be contributed under this project's BSD-3-Clause license (human attestation).
- [ ] Tests cover the final agreed API and lifecycle.
- [x] Documentation in `docs/source` is updated.
- [x] `just check` passes.
- [x] All commits follow Conventional Commits and are signed off.
- [ ] Required CI passes on the final submitted revision.
