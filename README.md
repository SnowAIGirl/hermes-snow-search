# Hermes Snow Search

> [![GitHub](https://img.shields.io/badge/GitHub-mlinquan%2Fhermes--snow--search-blue?logo=github)](https://github.com/mlinquan/hermes-snow-search)
> English | [中文版](README_CN.md)

In-memory parallel search plugin for [Hermes Agent](https://hermes-agent.nousresearch.com).
Loads session history, holographic facts (fact_store), and built-in memory (MEMORY.md / USER.md) into RAM.
Searches all stores in parallel — results in <1ms. Supports full message-body deep search.

## How it works

1. **Eager load** — data is loaded in a background thread right after Hermes starts
2. **Keep in RAM** — sessions, facts, and memory entries live in Python lists, no I/O on search
3. **Parallel search** — `ThreadPoolExecutor` runs all stores concurrently
4. **Incremental updates** — `post_tool_call` hook catches `fact_store add` and `memory add` → appends to cache
5. **Eviction** — `pre_llm_call` hook checks memory usage; evicts oldest/lowest-trust entries when >80% of limit
6. **Deep search** — full message-body index with session_id + timestamp + role. Incremental refresh via `SELECT MAX(id)`

## Installation

```bash
pip install hermes-snow-search
hermes plugins enable hermes-snow-search
# Restart Hermes (/new or re-launch)
```

## Configuration

```yaml
plugins:
  hermes-snow-search:
    memory_limit_mb: 500          # safety cap, not actual usage
    session_max: 7000
    fact_max: 10000
    deep_search_enabled: true     # set false to use lightweight only
    deep_search_load_mode: "ondemand"   # "ondemand" | "startup"
```

| Key | Default | Description |
|-----|---------|-------------|
| `memory_limit_mb` | 500 | Hard memory cap; eviction triggers at 80% |
| `session_max` | 7000 | Max session entries in lightweight cache |
| `fact_max` | 10000 | Max fact entries in cache |
| `deep_search_enabled` | true | Enables full message-body search. Set `false` for lightweight-only mode |
| `deep_search_load_mode` | ondemand | `ondemand` = load on first search, `startup` = background at boot |

> `memory_limit_mb` (500 MB) is a safety cap, not actual usage. One week of real conversation (~230 sessions, ~10,000 messages) fits in ~6 MB. At 500 MB you can store roughly **1-2 years** of heavy daily use — memory won't be the bottleneck.

## Deep Search

Enabled by default (`deep_search_enabled: true`). When active, full message-body search replaces lightweight session summaries automatically. Results include `session_id`, `timestamp`, `role`, and `search_info`.

### Load modes

| Mode | When | Behavior |
|------|------|----------|
| `ondemand` (default) | On first deep search | Blocks until index is built, shows progress |
| `startup` | Background, 2.5s after startup | Non-blocking, prints progress at ~0/50/100% |

Progress is written to stderr:

```
[Hermes Snow Search] Loading deep search index...
[Hermes Snow Search] Session 58/231 | 2,500 messages | 10/500 MB | ~0.6s remaining
[Hermes Snow Search] Deep search ready | 10,229 messages | 7 days (May 13 ~ May 20) | 6 MB
```

Index builds from newest sessions backwards, stops at 85% of `memory_limit_mb`. Subsequent calls use `SELECT MAX(id)` for incremental refresh — cross-process sync is automatic (shared state.db).

### Sort modes

| `sort` | Behavior |
|--------|----------|
| `relevance` (default) | Best match first (recency + keyword score) |
| `oldest` | Earliest timestamp first — answer "when did X first happen" |
| `newest` | Latest timestamp first — answer "when was the last X" |

### Performance

| Mode | Searches | Latency | Memory (1 week) |
|------|----------|---------|-----------------|
| Lightweight | Session summaries | <0.5ms | ~3 MB |
| Deep | Full message bodies | ~1-5ms | ~6 MB |

Lightweight and deep mode never load simultaneously — deep mode skips sessions and loads facts + memory + messages.

## Caveats

- **First use delay (ondemand):** First deep search triggers index building (~1s for ~1 week).
- **Root sessions only:** Deep search indexes user ↔ assistant conversations. Subagent sessions (delegate_task children) are excluded.
- **Tool messages excluded:** Only `user` and `assistant` role messages are stored.

## Usage Tips

- **"Latest" questions match naturally** — snow_search ranks by relevance with recency boost.
- **"First time" questions use `sort="oldest"`** — the earliest hit moves to the top.
- **Specific keywords win** — "database migration schema users" beats "that database thing".
- **Cross-process auto-sync** — no manual reload needed between CLI and Gateway.
- **Trust the result** — snow_search sweeps everything in RAM. If it found nothing, there's no record.

## Author

LinQuan & Snow (AI Girl)
