<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# perf(rng)!: seed only CPU RNGs by default for stateless transforms

[Code branch](https://github.com/asolergi-nv/Megatron-Energon/tree/perf/cpu-only-stateless-rng)

Existing PR: https://github.com/NVIDIA/Megatron-Energon/pull/339

The latest remote branch now makes CPU-only seeding the default and exposes seed_accelerators=True for transforms that use accelerator randomness. This is a breaking default change and must be reviewed as such. The original opt-in-only description is no longer accurate. No changes were made to this branch during review preparation.

Previously measured standalone-loader evidence: first batch 117.63s → 71.68s; setup plus loader 124.33s → 78.94s. Shared-path source-file checks 506,422 → 180 in a separate API audit; not filesystem RPC counts. These are pinned prototype measurements, not a rerun of the current remote revision.

Dependency: none of the other pending performance changes. The internal encoder MR is now an explicit declaration of the default, not necessary to opt into CPU-only behavior.

Validation of the latest remote revision has not been repeated in this task. Preserve accelerator seeding opt-in for transforms that consume that RNG. Full training interruption/restart validation remains unclaimed.

- [ ] Human review of the breaking default and all changed lines completed.
