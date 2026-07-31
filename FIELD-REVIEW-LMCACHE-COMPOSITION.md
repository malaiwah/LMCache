# LMCache PR #18/#19/#20 Composition Handoff

## Result

The three LMCache fixes compose successfully when their divergent prerequisite
branches are replayed in the order below. The composed branch passes the
focused CPU, CUDA, outage/recovery, and real LMCache server round-trip tests.
It also adds a lifecycle reconciliation so completion, timeout, and late
responses cannot release retained CUDA resources more than once or overwrite
the first terminal result.

This branch is local-only pending independent review. It has not been pushed
and no upstream PR or issue comment has been posted.

## Exact composition

The local branch is `codex/lmcache-compose-lifecycle` in
`/Users/mbelleau/Projects/lmcache-compose-lifecycle`.

| Order | Source | Upstream SHA | Composed SHA |
| ---: | --- | --- | --- |
| 0 | release base | `9cebd405d0caf4bebe01d694b5a8bf4e3e354314` | same |
| 1 | PR #7 prerequisite | `7b7583aef55e98ba05a6c199e71601d78552e794` | `1c772afc56e40f6a9893474696aa80c7a0824794` |
| 2 | PR #8 prerequisite 1 | `3cf1e8a54558b12dcbda402016d2ff621ebd0652` | `d7c1faa260647386cc4b93b119be23f5010d5b28` |
| 3 | PR #8 prerequisite 2 | `b983bb69339396903f52d9436b29efa600c259ab` | `ba4abb49e9fbd401ded66a24bdfcc099f477d617` |
| 4 | PR #8 prerequisite 3 | `31c4175d2134518e5b43fe8f4a7d072df6043a13` | `afec6065332e2c54a3d1a302bf491ef3357cef3c` |
| 5 | PR #18 | `85abae7d2dab3585be9ad920dc634ea37c905333` | `7b44b2293ab89b17cee466d65058db0eb8a3dcde` |
| 6 | PR #19 | `1d4396c70352764d1fa5c85ef2f27dbe948d6481` | `bf292834aa4bdac325431ce1bf7ee5b2258100d9` |
| 7 | PR #20 responsiveness | `9255d084a6e4e2e38ce3d5efd1ec74380dbd4250` | `0f65a94efa2f17d60c907525ae83dd0df470e98e` |
| 8 | PR #20 outage cleanup | `5aba0dddb37b78496e6ef87a4e14b1840962eb95` | `60769c81db1e9589f555cee554dee8045d3020f1` |
| 9 | PR #20 tests | `9374b2970987a8e6f7027658c802b7835e007b7a` | `676392727e90f69bc8c0b7ca5ff5c01404bebaa4` |
| 10 | lifecycle/race tests | local | `2e4d740476e8120b2720399b0d7adfd14dfc564d` |
| 11 | traceback-retention fix | local | `2014fbb2271eae516b9f2c61dd0098601e99bd19` |
| 12 | composed static cleanup | local | `caf7417be83f225c93e426f0885b935cefcc388c` |

PR #20's outage-cleanup commit required semantic conflict resolution after
PR #18 and PR #19; a mechanical cherry-pick is not sufficient.

## Lifecycle reconciliation

`MessagingFuture` now has one locked `_complete(...)` transition:

- the first result, exception, or timeout owns the terminal state;
- later replies and failures are ignored;
- retained resources are detached while holding the completion lock and
  cleared outside it, exactly once;
- `_expire()` invokes the timeout callback only when expiry wins;
- the raw messaging future and its CUDA wrapper retain independent leases.

The first CPU union exposed a second lifecycle leak: re-raising the stored
timeout exception object attached a traceback that retained the
`CUDAMessagingFuture`, and therefore its exporter event. The final code stores
a traceback-free timeout sentinel and raises a fresh `LMCacheTimeoutError`.
Ordinary remote exceptions retain their prior object-identity behavior.

Focused tests cover timeout/late-reply ordering, success/late-failure ordering,
100 concurrent completion/timeout races, and independent raw/caller CUDA
leases.

## Qualification commands and results

Remote GPU worktree:
`/workspace/field-review-tests/worktrees/lmcache-compose-lifecycle`

Remote evidence:
`/workspace/field-review-tests/artifacts`

Copied local evidence:
`/Users/mbelleau/Documents/GLM-5.2 Turnkey Appliance/field-review-evidence/lmcache-compose-lifecycle`

### CPU union

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$wt" /opt/venv/bin/python -m pytest -q \
  tests/v1/multiprocess/test_mq.py \
  tests/v1/multiprocess/test_futures.py \
  tests/v1/multiprocess/test_engine_driven_transfer.py \
  tests/v1/multiprocess/test_lmcache_driven_event_ordering.py \
  tests/v1/test_vllm_mp_adapter.py \
  tests/cli/commands/bench/test_server_bench.py::TestHandleTransferEvents
```

Result: `124 passed, 13 skipped` in 54.38 seconds.

Evidence: `lmcache-compose-lifecycle-cpu-v2.log`

### CUDA union, physical GPU 3

The same union ran with `CUDA_VISIBLE_DEVICES=3` and an isolated extension
cache at
`/workspace/field-review-tests/cache/lmcache-compose-lifecycle-cold`.

- cold: `137 passed` in 76.45 seconds;
- warm/reused cache: `137 passed` in 76.40 seconds.

Evidence:

- `lmcache-compose-lifecycle-gpu3-cold.log`
- `lmcache-compose-lifecycle-gpu3-warm.log`

### Injected timeout, outage, and recovery

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH="$wt" /opt/venv/bin/python -m pytest -vv -s \
  tests/v1/multiprocess/test_futures.py::test_timeout_releases_resources_once_and_rejects_late_reply \
  tests/v1/multiprocess/test_futures.py::test_first_success_releases_resources_once_and_rejects_late_failure \
  tests/v1/multiprocess/test_futures.py::test_completion_timeout_race_has_one_terminal_owner \
  tests/v1/multiprocess/test_futures.py::test_cuda_timeout_releases_raw_lease_but_not_caller_lease \
  tests/v1/multiprocess/test_mq.py::test_full_dead_client_queue_does_not_block_healthy_client \
  tests/v1/multiprocess/test_mq.py::test_timed_out_future_is_reclaimed_and_late_response_is_ignored \
  tests/v1/multiprocess/test_mq.py::test_connection_reset_discards_stale_pending_and_unsent_work \
  tests/v1/test_vllm_mp_adapter.py::test_heartbeat_retires_outage_session_then_recovers \
  tests/v1/test_vllm_mp_adapter.py::test_dropped_retrieve_reported_once_via_unhealthy_get_finished \
  tests/v1/test_vllm_mp_adapter.py::test_dropped_retrieve_reported_once_via_healthy_get_finished
```

Result: `10 passed` in 15.47 seconds. The log records a saturated dead-client
queue while a healthy client proceeds, and an explicit unhealthy-to-healthy
recovery transition.

Evidence: `lmcache-compose-lifecycle-timeout-outage-recovery.log`

### Real LMCache server/native CUDA round trip

The source checkout used the deployed `lmcache.c_ops` native ABI and started:

```bash
lmcache server \
  --host 127.0.0.1 --port 15558 \
  --http-host 127.0.0.1 --http-port 18081 \
  --chunk-size 256 --l1-size-gb 5 --eviction-policy LRU --max-workers 1
```

The client used:

```bash
lmcache bench server \
  --rpc-url tcp://127.0.0.1:15558 \
  --url http://127.0.0.1:18081 \
  --mode gpu --transfer-mode lmcache_driven \
  --start 0 --end 3 --interval 0.1
```

Result: registration, three 512-token stores, three retrieves, all three
checksum comparisons, and unregistration succeeded. Cold store mean was
2.78 ms and warm retrieve mean was 2.32 ms.

Evidence:

- `lmcache-compose-lifecycle-real-server.log`
- `lmcache-compose-lifecycle-real-gpu-roundtrip.log`

### Static checks

Ruff check, Ruff format check, and `git diff --check` all passed across the 17
changed Python files.

Evidence: `lmcache-compose-lifecycle-static-final.log`

## Remaining warnings and scope

- CUDA tests and the real bench still emit:
  `Producer process has been terminated before all shared CUDA tensors released`.
  This warning was already present in the individual PR #18 evidence. All
  protocol operations and checksum checks pass, but process-shutdown CUDA IPC
  cleanup remains a follow-up risk rather than a resolved claim.
- This subtask did not run a full vLLM plus LMCache serving stack. That belongs
  to the combined turnkey candidate qualification.
- The branch still requires independent review before push or upstream
  comments.
- The remote test server was stopped cleanly; physical GPU 3 returned to
  14 MiB usage.
