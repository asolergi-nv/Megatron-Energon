<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# perf(metadata): replay immutable construction metadata

[Code branch](https://github.com/asolergi-nv/Megatron-Energon/tree/review/startup-09-construction-snapshot) · [Review changes](https://github.com/asolergi-nv/Megatron-Energon/compare/c78d47025cd2c3617fc508007353095ce9c883bf...861048606a27d8203a6bf6b052933e7d4b46242c)

Implementation commit: `861048606a27d8203a6bf6b052933e7d4b46242c`.

- [ ] I, a human, have personally reviewed every line of this PR and take responsibility for it.

## What does this PR do?

Capture immutable construction metadata once, then load a compact snapshot for recipe/configuration lookups and index-size checks. This is separate from the media-dimension indexes introduced earlier.

## Dependencies

Independent construction-snapshot candidate for the existing [Energon #342](https://github.com/NVIDIA/Megatron-Energon/pull/342). It has no dependency on another optimization branch. Review this as a replacement revision of the existing PR, not a duplicate PR.

## Performance evidence

3 fresh-process trials per configuration; first next() 29.01s -> 28.78s; setup + loader 36.31s -> 31.33s.

These are adjacent-stage measurements from the previously validated, pinned standalone-loader implementation. They are not fresh timings of this branch on the updated target base. Backend comparisons retain all other selected optimizations. No large-scale extrapolation is claimed.

Constructor metadata opens: 2,130 → 0; file checks: 3,746 → 0; metadata info queries: 904 → 0 in the instrumented scope. One snapshot load is still required. Counters overlap; do not sum them. Offline construction discovery is amortized across owners and restarts.

## Validation

- `just fix`: PASS.
- `just check`: PASS.
- `bwrap --bind / / --dev-bind /dev /dev --tmpfs /tmp --unshare-net just test`: PASS (Ran 229 tests in 653.658s; OK).
- `just check-commits`: PASS (one signed-off Conventional Commit).

Earlier standalone-loader evidence: exact tokens, masks, lengths, source identities and pixels across the selected stack; 64-batch comparisons and fresh-process dataloader restore checks for the final stack. That evidence does not establish full training-process interruption/restart correctness, nor replace branch CI after integration changes.

## Review decisions before merging

Snapshot contents describe immutable input metadata. Digest and caller-supplied recipe identity do not detect later changes to source files; regenerate the snapshot when inputs change. Offline capture is outside the startup timing.

## Linked issue

No new issue has been opened. Link the relevant issue if repository contribution policy requires one before submission.

## Checklist

- [ ] No code was copied from other repositories or projects; the whole change can be contributed under this project's BSD-3-Clause license (human attestation).
- [ ] Tests cover the final agreed API and lifecycle.
- [x] Documentation in `docs/source` is updated.
- [x] `just check` passes.
- [x] All commits follow Conventional Commits and are signed off.
- [ ] Required CI passes on the final submitted revision.
