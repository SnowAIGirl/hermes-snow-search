"""Snow Search Engine — in-memory parallel memory search.

Design
------
All data lives in RAM: session summaries, holographic facts, and built-in
memory entries. Searches traverse Python lists — no SQLite, no I/O, no syscall.

Lifecycle
  1. Lazy full load on first snow_search() call.
  2. Incrementally updated via post_tool_call hook (detect fact_store add,
     memory add).
  3. Eviction check runs on pre_llm_call when usage > 80% of limit.

Config
  plugins:
    hermes-snow-search:
      memory_limit_mb: 500
      session_max: 2000
      fact_max: 10000
      memory_max: 100

If holographic memory (fact_store) is not enabled, facts are silently skipped.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_LIMIT_MB = 20
_DEFAULT_SESSION_MAX = 7000
_DEFAULT_FACT_MAX = 10000
_DEFAULT_MEMORY_MAX = 100

# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

SNOW_SEARCH_SCHEMA = {
    "name": "snow_search",
    "description": (
        "High-speed parallel memory search across ALL available stores (session history, "
        "holographic facts, and built-in memory). All data is cached in RAM — results "
        "return in <1ms. Use this FIRST when the user asks about past conversations, "
        "preferences, or anything you might have discussed before. Falls back to empty "
        "results silently when a store is unavailable. Supports fuzzy keyword matching."
        "\n\n"
        "When to use vs session_search / fact_store:"
        "\n- snow_search: broad recall, \"what do I know about X\" — searches everything at once"
        "\n- session_search: deep drill into a specific session (scroll, bookends, FTS5)"
        "\n- fact_store: structured CRUD (add facts, probe entities, reason across entities)"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query — keywords, partial matches supported.",
            },
            "limit_per_source": {
                "type": "integer",
                "description": "Max results per source (default: 5, max: 20).",
                "default": 5,
            },
            "include_sessions": {
                "type": "boolean",
                "description": "Include session history in search (default: true).",
                "default": True,
            },
            "include_facts": {
                "type": "boolean",
                "description": "Include holographic facts in search (default: true).",
                "default": True,
            },
            "include_memory": {
                "type": "boolean",
                "description": "Include built-in memory entries in search (default: true).",
                "default": True,
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def _tokenize(text: str) -> set[str]:
    """Lowercase keyword tokens."""
    return set(_WHITESPACE.split(text.lower().strip())) if text else set()


def _match_score(query_tokens: set[str], text: str) -> float:
    """Score how many query tokens appear in text."""
    if not query_tokens or not text:
        return 0.0
    lower = text.lower()
    hits = sum(1 for t in query_tokens if t in lower)
    return hits / len(query_tokens) if hits > 0 else 0.0


def _estimate_bytes(obj: Any) -> int:
    """Rough memory estimate for a Python object (dict/list of strings)."""
    try:
        raw = json.dumps(obj, ensure_ascii=False, default=str)
        return len(raw.encode("utf-8"))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SnowSearchEngine:
    """In-memory search engine — lazy loads, incremental updates, eviction."""

    def __init__(self, ctx) -> None:
        self._ctx = ctx
        self._lock = threading.Lock()

        # --- config ---
        config = self._load_config()
        self._memory_limit = config.get("memory_limit_mb", _DEFAULT_LIMIT_MB) * 1024 * 1024
        self._session_max = config.get("session_max", _DEFAULT_SESSION_MAX)
        self._fact_max = config.get("fact_max", _DEFAULT_FACT_MAX)
        self._memory_max = config.get("memory_max", _DEFAULT_MEMORY_MAX)

        # --- data (lazy loaded) ---
        self._ready = False
        self._load_error: str | None = None

        self._sessions: list[dict] = []  # title, session_id, last_active, preview
        self._facts: list[dict] = []  # fact_id, content, trust_score, category, tags
        self._memory_entries: list[dict] = []  # source (MEMORY.md/USER.md), content

        self._current_bytes = 0

        # --- holographic availability (checked once) ---
        self._holographic_available: bool | None = None

    # -- config ---------------------------------------------------------------

    @staticmethod
    def _load_config() -> dict:
        """Read plugin config from ~/.hermes/config.yaml."""
        try:
            from hermes_constants import get_hermes_home
            config_path = get_hermes_home() / "config.yaml"
            if not config_path.exists():
                return {}
            import yaml
            with open(config_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            return all_config.get("plugins", {}).get("hermes-snow-search", {}) or {}
        except Exception:
            return {}

    # -- load -----------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Lazy full load on first search. Thread-safe."""
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            try:
                self._load_all()
                self._ready = True
                logger.info(
                    "snow-search loaded: %d sessions, %d facts, %d memory entries, ~%d MB",
                    len(self._sessions),
                    len(self._facts),
                    len(self._memory_entries),
                    self._current_bytes // (1024 * 1024),
                )
            except Exception as e:
                self._load_error = str(e)
                logger.warning("snow-search initial load failed: %s", e)

    def _load_all(self) -> None:
        """Load all three stores into RAM — silent, no terminal output."""
        total = 0

        memory = self._load_memory()
        total += _estimate_bytes(memory)
        self._memory_entries = memory

        sessions = self._load_sessions()
        total += _estimate_bytes(sessions)
        self._sessions = sessions

        facts = self._load_facts()
        total += _estimate_bytes(facts)
        self._facts = facts
        self._current_bytes = total

    def _load_sessions(self) -> list[dict]:
        """Load session titles + previews from SessionDB."""
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            raw = db.list_sessions_rich(
                limit=self._session_max + 10,
                exclude_sources=["tool"],
                order_by_last_active=True,
            )
            results = []
            for s in raw:
                if s.get("parent_session_id"):
                    continue
                results.append({
                    "session_id": s.get("id", ""),
                    "title": s.get("title", ""),
                    "last_active": s.get("last_active", ""),
                    "preview": s.get("preview", ""),
                    "message_count": s.get("message_count", 0),
                })
                if len(results) >= self._session_max:
                    break
            return results
        except Exception as e:
            logger.debug("snow-search session load failed: %s", e)
            return []

    def _resolve_fact_store_path(self) -> str | None:
        """Resolve the holographic memory DB path from config."""
        try:
            from hermes_constants import get_hermes_home
            from hermes_cli.config import cfg_get
            import yaml

            config_path = get_hermes_home() / "config.yaml"
            db_path = None
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    all_cfg = yaml.safe_load(f) or {}
                mem_cfg = cfg_get(all_cfg, "plugins", "hermes-memory-store", default={}) or {}
                db_path = mem_cfg.get("db_path")
            if not db_path:
                db_path = str(get_hermes_home() / "memory_store.db")
            return db_path
        except Exception:
            return None

    def _load_facts(self) -> list[dict]:
        """Load all facts from holographic memory (if available).

        Combines availability check + data load in one DB open.
        """
        db_path = self._resolve_fact_store_path()
        if not db_path:
            self._holographic_available = False
            return []

        try:
            from plugins.memory.holographic.store import MemoryStore

            store = MemoryStore(db_path=db_path)
            try:
                count = store._conn.execute(
                    "SELECT COUNT(*) FROM facts"
                ).fetchone()[0]
                if count == 0:
                    self._holographic_available = False
                    return []

                rows = store._conn.execute(
                    "SELECT fact_id, content, category, tags, trust_score "
                    "FROM facts ORDER BY trust_score DESC LIMIT ?",
                    (self._fact_max,),
                ).fetchall()
            finally:
                store.close()

            self._holographic_available = True
            return [
                {
                    "fact_id": r[0],
                    "content": r[1],
                    "category": r[2] or "general",
                    "tags": r[3] or "",
                    "trust_score": r[4] if r[4] else 0.5,
                }
                for r in rows
            ]
        except Exception as e:
            logger.debug("snow-search facts load failed: %s", e)
            self._holographic_available = False
            return []

    def _load_memory(self) -> list[dict]:
        """Load built-in memory entries from MEMORY.md and USER.md."""
        results = []
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
            memories_dir = home / "memories"

            for source in ("MEMORY.md", "USER.md"):
                path = memories_dir / source
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                # Entries are delimited by §
                entries = [e.strip() for e in text.split("§") if e.strip()]
                for entry in entries[:self._memory_max]:
                    results.append({
                        "source": source.replace(".md", ""),
                        "content": entry,
                    })
        except Exception as e:
            logger.debug("snow-search memory load failed: %s", e)
        return results

    # -- search ---------------------------------------------------------------

    def handle_search(
        self,
        args: dict,
        **kwargs,
    ) -> str:
        """Tool handler: parallel search across all loaded stores."""
        self._ensure_loaded()

        query = args.get("query", "")
        limit = min(int(args.get("limit_per_source", 5)), 20)
        include_sessions = args.get("include_sessions", True)
        include_facts = args.get("include_facts", True)
        include_memory = args.get("include_memory", True)

        if not query or not query.strip():
            return json.dumps({
                "success": True,
                "query": "",
                "hits": [],
                "total": 0,
                "stores_available": {
                    "sessions": bool(self._sessions),
                    "facts": bool(self._facts),
                    "memory": bool(self._memory_entries),
                },
                "message": "No query provided. Use query= to search across all stores.",
            })

        query_tokens = _tokenize(query)
        if not query_tokens:
            return json.dumps({"success": True, "query": query, "hits": [], "total": 0})

        # Determine which stores to search
        stores = {}
        if include_sessions and self._sessions:
            stores["sessions"] = self._sessions
        if include_facts and self._facts:
            stores["facts"] = self._facts
        if include_memory and self._memory_entries:
            stores["memory"] = self._memory_entries

        if not stores:
            return json.dumps({
                "success": True,
                "query": query,
                "hits": [],
                "total": 0,
                "message": "No stores available — all data sources are empty or disabled.",
            })

        # Parallel search
        hits = []
        with ThreadPoolExecutor(max_workers=len(stores)) as ex:
            future_map = {}
            for store_name, data in stores.items():
                fn = self._make_searcher(store_name, data, query_tokens, limit)
                future_map[ex.submit(fn)] = store_name
            for f in as_completed(future_map):
                store_name = future_map[f]
                try:
                    results = f.result()
                    hits.extend(results)
                except Exception as e:
                    logger.debug("snow-search %s failed: %s", store_name, e)

        # Sort by score desc, then source priority
        hits.sort(key=lambda h: (-h.get("score", 0), h.get("source", "")))
        total = len(hits)
        # Trim to reasonable total
        hits = hits[:limit * 3]

        return json.dumps({
            "success": True,
            "query": query,
            "hits": hits,
            "total": total,
            "stores_available": {
                "sessions": bool(self._sessions),
                "facts": bool(self._facts),
                "memory": bool(self._memory_entries),
            },
        }, ensure_ascii=False)

    def _make_searcher(self, store_name: str, data: list[dict], tokens: set[str], limit: int):
        """Return a callable that searches one store."""
        def _search():
            scored = []
            for item in data:
                score = self._score_item(store_name, tokens, item)
                if score > 0:
                    entry = {"source": store_name, "score": round(score, 3)}
                    entry["content"] = self._format_item(store_name, item)
                    if store_name == "facts":
                        entry["trust_score"] = item.get("trust_score", 0.5)
                        entry["category"] = item.get("category", "general")
                    elif store_name == "sessions":
                        entry["session_id"] = item.get("session_id", "")
                        entry["title"] = item.get("title", "Untitled")
                        entry["last_active"] = item.get("last_active", "")
                    scored.append(entry)

            scored.sort(key=lambda x: -x["score"])
            return scored[:limit]
        return _search

    @staticmethod
    def _score_item(store: str, tokens: set[str], item: dict) -> float:
        """Compute relevance score for one item."""
        if store == "sessions":
            title = item.get("title", "")
            preview = item.get("preview", "")
            score = _match_score(tokens, title) * 3.0
            score += _match_score(tokens, preview) * 1.5
            return score

        elif store == "facts":
            content = item.get("content", "")
            tags = item.get("tags", "")
            score = _match_score(tokens, content) * 2.0
            score += _match_score(tokens, tags) * 3.0
            trust = item.get("trust_score", 0.5)
            score *= trust
            return score

        elif store == "memory":
            content = item.get("content", "")
            return _match_score(tokens, content) * 2.0

        return 0.0

    @staticmethod
    def _format_item(store: str, item: dict) -> str:
        """Short display string for one item."""
        if store == "sessions":
            return item.get("preview", "") or item.get("title", "")
        elif store == "facts":
            return item.get("content", "")
        elif store == "memory":
            return item.get("content", "")
        return ""

    # -- incremental updates via hooks ----------------------------------------

    def on_post_tool_call(
        self,
        tool_name: str = "",
        args: dict | None = None,
        result: str = "",
        **kwargs,
    ) -> None:
        """Detect fact_store/memory writes → update in-memory cache."""
        if not self._ready:
            return
        if not args:
            args = {}

        try:
            if tool_name == "fact_store" and args.get("action") == "add":
                self._on_fact_added(args, result)
            elif tool_name == "memory" and args.get("action") == "add":
                self._on_memory_added(args)
        except Exception:
            pass  # Never let a hook crash the agent loop

    def _on_fact_added(self, args: dict, result: str) -> None:
        """Append new fact to in-memory list."""
        content = args.get("content", "")
        if not content:
            return
        try:
            parsed = json.loads(result)
            fact_id = parsed.get("fact_id")
        except Exception:
            fact_id = None

        entry = {
            "fact_id": fact_id or 0,
            "content": content,
            "category": args.get("category", "general"),
            "tags": args.get("tags", ""),
            "trust_score": 0.5,
        }
        with self._lock:
            self._facts.insert(0, entry)
            self._current_bytes += _estimate_bytes(entry)
        # Don't trigger eviction here — wait for pre_llm_call

    def _on_memory_added(self, args: dict) -> None:
        """Append new memory entry to in-memory list."""
        target = args.get("target", "memory")  # "memory" or "user"
        content = args.get("content", "")
        if not content:
            return

        entry = {
            "source": "MEMORY.md" if target == "memory" else "USER.md",
            "content": content,
        }
        with self._lock:
            self._memory_entries.insert(0, entry)
            self._current_bytes += _estimate_bytes(entry)

    def on_pre_llm_call(self, **kwargs) -> dict | str | None:
        """Eviction check before each LLM call."""
        if not self._ready:
            return None
        limit_mb = self._memory_limit
        if limit_mb <= 0:
            return None
        threshold = int(limit_mb * 0.8)

        with self._lock:
            if self._current_bytes < threshold:
                return None
            self._evict()
        return None  # No context injection needed

    def _evict(self) -> None:
        """Evict least valuable entries until under threshold."""
        target = int(self._memory_limit * 0.6)

        # 1. Sessions: keep most recent
        self._sessions.sort(key=lambda s: float(s.get("last_active", 0) or 0), reverse=True)
        self._sessions = self._sessions[:self._session_max]

        # 2. Facts: keep highest trust
        self._facts.sort(key=lambda f: f.get("trust_score", 0.5), reverse=True)
        self._facts = self._facts[:self._fact_max]

        # 3. Memory entries: keep most recent (inserted at head on update)
        self._memory_entries = self._memory_entries[:self._memory_max]

        self._current_bytes = (
            _estimate_bytes(self._sessions)
            + _estimate_bytes(self._facts)
            + _estimate_bytes(self._memory_entries)
        )

    # -- reload ---------------------------------------------------------------

    def reload(self) -> str:
        """Force a full reload of all stores. Usable from /snow-reload command."""
        with self._lock:
            self._ready = False
            self._holographic_available = None
        self._ensure_loaded()
        if self._load_error:
            return f"snow-search reloaded with error: {self._load_error}"
        return (
            f"snow-search reloaded: "
            f"{len(self._sessions)} sessions, {len(self._facts)} facts, "
            f"{len(self._memory_entries)} memory entries, "
            f"~{self._current_bytes // (1024 * 1024)} MB"
        )
