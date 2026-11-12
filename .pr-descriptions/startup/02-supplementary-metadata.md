<!-- Copyright (c) 2026, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

# Dropped: supplementary metadata lookup adapter

[Energon PR #349](https://github.com/NVIDIA/Megatron-Energon/pull/349) was closed at the author's request. Missing metadata will be handled through dataset preparation rather than a runtime patch in Energon.

The experiment reduced dimension-discovery image opens from 1,549 to zero, but showed no clear first-batch latency improvement at the measured scale. The recorded measurements and experiment artifacts remain available; this adapter is not part of the proposed implementation path.
