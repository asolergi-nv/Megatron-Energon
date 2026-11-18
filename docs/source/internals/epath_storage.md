<!--- Copyright (c) 2025, NVIDIA CORPORATION.
SPDX-License-Identifier: BSD-3-Clause -->

(epath-storage)=

# EPath, Mapped Arrays, and Storage

Energon uses `EPath` so dataset code can work with local files and supported remote protocols through a
common path interface. Readers should preserve that abstraction until an operation specifically requires a
local file.

## Path handling

Accept `EPath` or values convertible to it at API boundaries. Use path composition and file operations
provided by `EPath`; converting to `pathlib.Path` or calling built-in `open` too early silently removes
remote support.

Protocol handlers may also compose an auxiliary path with a primary path, as used by mounted or crude
datasets. Keep the original dataset path in configuration and provenance even when bytes are served from a
cache or auxiliary location.

## Mapped numeric arrays

`map_epath` exposes one-dimensional typed arrays used by indexes. Its backend depends on the path and
options:

- a local path with memory mapping enabled uses `EPathNumpyMappedArray`;
- a remote path, or a request without memory mapping, uses seek/read access;
- `copy_to_local` first creates or reuses a local copy and then memory maps it.

The shape is inferred from file size, byte offset, and dtype. Validate that the remaining byte count is
divisible by the dtype size. The abstraction is one-dimensional; multidimensional interpretation belongs
to the caller.

A stepped slice may read the full byte span between its first and last element and apply the step afterward.
Avoid sparse, extremely wide slices on remote storage when a narrower access pattern is possible.

Mapped arrays and readers are context managers and must be closed when the owner is finished. Do not put an
open file, stream, or memory map into dataset configuration or checkpoint state.

## Frozen construction metadata

For an immutable dataset revision, capture the metadata used by dataset construction
once, validate it once at job startup, and distribute the validated object to the
processes building loaders. Sample/media payloads and binary indexes remain outside
the snapshot. This is a construction cache, not a checkpoint of sampler or packing state.

```python
import json
from pathlib import Path
from megatron.energon import WorkerConfig, get_train_dataset, get_val_dataset
from megatron.energon.epathlib.metadata_snapshot import (
    MetadataSnapshot, capture_metadata, use_metadata_snapshot,
)

recipe = "/data/versioned-dataset/recipe.yaml"
package = Path("/local/metadata.json")  # Parent directory must already exist.
worker = WorkerConfig(rank=0, world_size=1, num_workers=0)

# Offline preparation: include every split the job will construct, without iteration.
with capture_metadata(package, recipe_identity="dataset-release-1"):
    get_train_dataset(recipe, worker_config=worker, batch_size=1,
                      shuffle_buffer_size=None, max_samples_per_sequence=None)
    get_val_dataset(recipe, worker_config=worker, batch_size=1)
audit = json.loads(Path(str(package) + ".audit.json").read_text())

# Run this live preflight once per job, outside capture/replay contexts.
validated = MetadataSnapshot.load(
    package, sha256=audit["snapshot_sha256"], recipe_identity="dataset-release-1",
).validate()

# Applications may serialize validated to their workers through their control plane.
with use_metadata_snapshot(validated) as coverage:
    dataset = get_train_dataset(recipe, worker_config=worker, batch_size=1,
                                shuffle_buffer_size=None, max_samples_per_sequence=None)
print(coverage.hits, coverage.misses, coverage.passthrough_opens)
# Start loader iteration after leaving the construction scope.
```

Capture uses native EPath/JSONL hooks and writes a format-2 package with its audit at
`<complete package filename>.audit.json`. The digest covers the recorded contents,
file-existence checks, sizes, JSONL terminal offsets, recipe identity and DSS root.
An invalid destination fails before construction; a failed constructor publishes
nothing. Each output file is replaced atomically. A crash between the two replacements
can leave an old audit, which the digest check rejects. Version-1 packages must be
recaptured.

`validate()` checks all captured metadata against live storage: recipe and metadata
bytes, file existence and sizes, JSONL terminal offsets, and `.nv-meta/index.uuid`
when available. It binds `dss://` resolution to the effective `NVDATASET_CACHE_DIR`.
It does not read every sample or hash all binary indexes. Same-size edits to sample
payloads, or index changes that preserve the checked offsets/generation, remain outside
this validation. Keep the entire dataset revision immutable from capture through the
job, and regenerate indexes and snapshots when publishing a new revision. A package
checksum alone is not evidence that the dataset is current.

Replay requires the result of preflight and rejects missing metadata keys rather than
silently returning an old or partially cached construction. Capture training and
validation together, using the same paths, storage configuration, Energon version and
constructor options as the job. Different ranks may reuse the same validated object.
Uncached binary/sample opens are counted separately; strict coverage does not forbid
normal sample I/O after construction. The snapshot cannot remain active in forked
children, and leaving the context restores ordinary reads even after an exception.

Energon owns capture, the package format, live validation, scoped replay and worker
isolation. The application owns immutable revision selection, job-level coordination,
distribution, configuration and checkpoint compatibility. A validated object is valid
only for the current job: do not save it in a training checkpoint or reuse it to skip
preflight after an interruption.

## Worker and cache ownership

Open storage resources inside the worker that consumes them. Forked or spawned workers must not rely on a
parent's live file position or remote client object.

When an index is safe to cache locally:

- key the cache by source identity and every option that affects the bytes;
- define freshness checks for source or metadata changes;
- write atomically so another worker cannot observe a partial index;
- treat the cached location as an optimization, not as the dataset's logical identity.

Prepared `.nv-meta` files are part of the dataset configuration contract, while ephemeral cache files are
not.

## Provenance

Every loaded part should retain `SourceInfo` describing where it came from. A `SampleRecord` aggregates
this in `__sources__`, and transformations should propagate it. Provenance is needed for useful error
messages, debugging filters, and tracing samples whose payload came from auxiliary storage.

For a mounted sample, distinguish:

- the logical sample key;
- the primary dataset and shard;
- the physical path used for each part;
- any auxiliary source that supplied the bytes.

Do not replace these with the cache path alone.

## Adding a storage backend

Before adding a new protocol or reader:

1. define seek, range-read, listing, stat, and local-copy capabilities;
2. make unsupported operations fail explicitly;
3. verify path normalization and equality, since paths participate in resume identity;
4. verify cleanup after normal iteration, exceptions, and worker shutdown;
5. test local and remote behavior through the same dataset reader;
6. test empty files, offsets, dtype alignment, slicing, and cache freshness;
7. preserve logical paths and source provenance in configuration and samples.

User-facing remote and auxiliary dataset configuration is documented in
[Remote Datasets](../advanced/remote_dataset.md) and [Crude Datasets](../advanced/crude_datasets.md).
