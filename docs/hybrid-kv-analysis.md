# LMCache Hybrid KV Support for Qwen 3.5: Analysis and Recommendations

Date: 2026-03-23
Branch: `codex/hybrid-kv-offload`

## Context

This document analyzes how the llm-d hybrid KV offloading work (done in
`/home/mbelleau/src/vllm` on branch `codex/hybrid-kv-offload`) relates to LMCache's
Qwen 3.5 support, and whether our vLLM changes help, hinder, or suggest small
improvements that could unblock both paths.

Qwen 3.5 uses a hybrid architecture with:
- **Mamba/linear-attention layers** (groups 0-2): compact recurrent state, very large
  `gpu_block_size` (equal to `max_model_len`, e.g. 98304 tokens)
- **Full-attention layers** (group 3): standard paged KV cache with smaller
  `gpu_block_size` (e.g. 1056 tokens)

## Architecture: Two Independent Connector Paths

A critical finding: **llm-d's `OffloadingConnector` and LMCache's `LMCacheConnectorV1`
are completely independent implementations** of vLLM's `KVConnectorBase_V1` interface.

```
vLLM KVConnectorBase_V1
├── OffloadingConnector  (llm-d path, uses vllm/v1/kv_offload/*)
│   └── SupportsHMA ✓  (works with hybrid KV cache manager)
├── LMCacheConnectorV1   (LMCache path, wraps LMCache engine)
│   └── SupportsHMA ✗  (crashes on hybrid models)
├── NixlConnector
│   └── SupportsHMA ✓
└── ... other connectors
```

They share:
- The same `KVConnectorBase_V1` interface (`save_kv_layer`, `start_load_kv`, etc.)
- The same call sites in vLLM's worker (`gpu_model_runner.py`, `kv_transfer_utils.py`)
- The same `register_kv_caches(kv_caches: dict[str, torch.Tensor])` entry point

They do NOT share:
- Any implementation code
- The `vllm/v1/kv_offload/` framework (offloading-specific)
- Scheduler-side planning logic
- Storage backends

**Conclusion**: Our llm-d vLLM changes don't directly help or hinder LMCache. But the
*knowledge* and *principles* transfer directly, and one small vLLM-side change could
unblock LMCache.

## The Crash: Precise Root Cause

The crash path:

```
vLLM gpu_model_runner.py:6534
  → register_kv_caches(kv_caches)        # dict of ALL layers, including mamba
    → LMCache adapter register_kv_caches()
      → _build_kv_layer_groups()          # correctly groups by shape/dtype ✓
      → ... later, on first request ...
      → _lazy_initialize_buffer(kv_caches)
        → discover_gpu_kv_format(kv_caches[0])  # kv_caches[0] is a mamba list! ✗
          → AttributeError: 'list' has no attribute 'shape'
```

**Key observation**: `save_kv_layer()` in vLLM is called from the
`maybe_transfer_kv_layer` decorator on attention backends
(`kv_transfer_utils.py:56`). **Mamba layers never trigger this decorator** because
they don't use the attention backend. So the per-layer save path naturally skips mamba
layers already.

The crash is only in the initialization/buffer-probing path, not in the steady-state
save/load path.

## Compatibility Assessment

### Our llm-d vLLM changes: compatible with LMCache

| Change | LMCache impact |
|---|---|
| `vllm/v1/kv_offload/planner.py` (HybridOffloadPlanner) | None — offloading-specific |
| `vllm/v1/kv_offload/spec.py` (validation, warnings) | None — offloading-specific |
| `offloading_connector.py` (handle_preemptions fix) | None — different connector |
| `vllm/v1/kv_offload/worker/cpu_gpu.py` (per-group worker) | None — offloading-specific |

**No conflicts.** Both connectors can coexist in the same vLLM build. Selection is by
config string: `kv_connector="OffloadingConnector"` vs `kv_connector="LMCacheConnectorV1"`.

### What would help LMCache on the vLLM side

One change in vLLM's `gpu_model_runner.py` could unblock LMCache without touching
LMCache at all:

```python
# gpu_model_runner.py:6526-6534 (current)
if has_kv_transfer_group():
    kv_transfer_group = get_kv_transfer_group()
    if self.cross_layers_kv_cache is not None:
        kv_transfer_group.register_cross_layers_kv_cache(...)
    else:
        kv_transfer_group.register_kv_caches(kv_caches)  # ALL layers

# Proposed: filter to attention-only for non-HMA connectors
if has_kv_transfer_group():
    kv_transfer_group = get_kv_transfer_group()
    if self.cross_layers_kv_cache is not None:
        kv_transfer_group.register_cross_layers_kv_cache(...)
    else:
        kv_transfer_group.register_kv_caches(kv_caches)
```

Even better: pass `kv_cache_config` alongside `kv_caches` so connectors know the
group structure. But the quickest fix is on the LMCache side (see below).

## What LMCache Already Has (Partial HMA Support)

### `KVLayerGroupsManager` (kv_layer_groups.py) — exists, unused

- Groups layers by `(shape, dtype)` — handles both tensor and list formats
- Built during `register_kv_caches()` via `_build_kv_layer_groups()`
- Has `get_group_by_layer_name()`, `get_group_by_layer_idx()`
- **Never consulted by the GPU connector** — dead infrastructure

### `save_kv_layer` adapter path — almost correct

- `save_kv_layer()` receives a single `kv_layer: torch.Tensor` per attention layer
- Mamba layers never reach this path (vLLM only calls it from attention backends)
- The crash is in `_lazy_initialize_buffer`, not in the per-layer path

## The Minimal LMCache Fix (Phase 1)

Since `save_kv_layer` already naturally skips mamba layers, the fix is narrow:

### 1. Guard `_lazy_initialize_buffer` against heterogeneous kv_caches

In `gpu_connectors.py:642`:
```python
def _lazy_initialize_buffer(self, kv_caches):
    if self.use_gpu and self.gpu_buffer_allocator is None:
        # Filter to only tensor-typed entries (skip mamba list-of-tensors)
        if isinstance(kv_caches, list):
            tensor_kv_caches = [kv for kv in kv_caches if isinstance(kv, torch.Tensor)]
        elif isinstance(kv_caches, dict):
            tensor_kv_caches = [kv for kv in kv_caches.values()
                                if isinstance(kv, torch.Tensor)]
        else:
            tensor_kv_caches = kv_caches

        if not tensor_kv_caches:
            logger.warning("No tensor-typed KV caches found, skipping GPU buffer init")
            return

        self.gpu_kv_format = discover_gpu_kv_format(tensor_kv_caches, EngineType.VLLM)
        ...
```

### 2. Track attention layer count separately

In the adapter, use `KVLayerGroupsManager` to determine `num_layers` for the attention
group only (not all layers). This ensures `store_layer()` allocates the correct number
of memory objects.

### 3. Skip non-attention layers in `start_load_kv`

Filter `kvcaches = list(self.kv_caches.values())` to only attention layers when
passing to the cache engine's retrieval path.

**Estimated effort**: ~50 lines changed across 2-3 files. This gives attention-layer
caching on Qwen 3.5 immediately — which is the high-value path since attention layers
are the expensive ones.

## What llm-d Teaches That LMCache Should Adopt

### Principle 1: Per-group is the right abstraction

llm-d treats each KV group as an independent storage unit. LMCache should evolve
toward this:
- One GPU connector per `KVLayerGroupInfo` (Phase 2)
- Per-group storage allocation in `cache_engine.py`
- Per-group format detection via `discover_gpu_kv_format_for_group()`

### Principle 2: Chunk alignment matters for hybrid models

llm-d's `HybridOffloadPlanner` handles the fundamental asymmetry: mamba state is
monolithic (one block = entire sequence), while attention KV is pageable. The planner
computes `first_hashable_chunk_idx` and `offload_unit_sizes` to find the right
granularity.

LMCache's token database does its own chunking but doesn't account for per-group block
size constraints. For full hybrid support (Phase 3), it would need similar logic.

### Principle 3: max_model_len alignment

`max_model_len` must be a multiple of the chunk size. We added validation warnings for
this in the llm-d spec. LMCache should validate this too, or at least document it.

## Implementation Phases

### Phase 1: Unblock Qwen 3.5 (attention-layer caching only)

**LMCache changes only** — no vLLM changes needed:

| File | Change |
|---|---|
| `gpu_connectors.py:642` | Filter kv_caches to tensors-only in `_lazy_initialize_buffer` |
| `vllm_v1_adapter.py` | Use `KVLayerGroupsManager` to count attention layers; filter kv_caches in `start_load_kv` |

This unblocks the `AttributeError` crash. Attention layers get cached; mamba layers
are skipped (their state is small and cheap to recompute anyway).

### Phase 2: Per-group connectors (full hybrid support)

| File | Change |
|---|---|
| `gpu_connectors.py` | Per-group `GPUConnectorLayerwise` instances |
| `utils.py` | `discover_gpu_kv_format_for_group()` |
| `cache_engine.py` | Per-group shape in `store_layer()` allocation |
| `vllm_v1_adapter.py` | Layer→group→connector dispatch |

### Phase 3: Hybrid chunk planning (optional, port from llm-d)

Port `HybridOffloadPlanner` concepts for chunk-granularity prefix hits across groups.

## Can We Have Both llm-d AND LMCache?

Yes. They're independent connectors selected by `kv_connector` config string. A
deployment could use:
- `OffloadingConnector` for shared-storage KV offloading (llm-d path, multi-node)
- `LMCacheConnectorV1` for local disk/CPU offloading (single-node, faster for local)
- `MultiConnector` wrapping both (vLLM has a `multi_connector.py` that fans out to
  multiple connectors)

The `MultiConnector` at `vllm/.../v1/multi_connector.py` already supports this — it
calls `save_kv_layer` on all child connectors. Having both means: local LMCache for
fast same-node reuse, plus llm-d shared storage for cross-node cache sharing.

## Mapping: llm-d Concepts → LMCache Equivalents

| llm-d concept | LMCache equivalent | Status |
|---|---|---|
| `OffloadingSpec.gpu_block_size` per group | `KVLayerGroupInfo.shape` | EXISTS |
| `HybridOffloadPlanner` | Token database + chunk planning | MISSING |
| Per-group worker handlers | Per-group GPU connectors | MISSING |
| `block_size_factors` per group | `get_shape()` per group | MISSING |
| `group_hash_block_size` | Token database chunk size | PARTIAL (uniform) |
| `requires_partial_group_offload` | N/A in LMCache | NOT NEEDED |
| C++ `TensorCopier` per group | CUDA kernels per format | MISSING for mamba |

## Bottom Line

1. **Our llm-d vLLM changes are fully compatible with LMCache** — no conflicts.
2. **The LMCache crash is narrow** — initialization probes all layers, but the
   steady-state save path already skips mamba layers naturally.
3. **~50 lines of LMCache changes** unblock attention-layer caching on Qwen 3.5.
4. **Having both llm-d and LMCache is viable** and complementary (shared vs local).
5. **The llm-d per-group principle** is the right long-term architecture for LMCache
   too, and `KVLayerGroupsManager` is the existing hook to build on.
