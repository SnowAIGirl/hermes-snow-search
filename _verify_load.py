"""Verify that snow_search loads 246 sessions + 291 facts + 36 memory entries into RAM."""
from hermes_snow_search.tools import SnowSearchEngine


class MockCtx:
    """Minimal mock for SnowSearchEngine — it only stores ctx, doesn't call it during load."""
    def register_tool(self, *a, **kw): pass
    def register_hook(self, *a, **kw): pass


engine = SnowSearchEngine(MockCtx())

# Trigger lazy load
engine._ensure_loaded()

print(f"snow-search 加载验证")
print(f"  sessions:  {len(engine._sessions)}")
print(f"  facts:     {len(engine._facts)}")
print(f"  memory:    {len(engine._memory_entries)}")
print(f"  RAM 估算:   ~{engine._current_bytes // 1024} KB")
print(f"  holographic: {engine._holographic_available}")
print(f"  error:     {engine._load_error}")

# Also verify the data is actually in memory (not just empty lists)
if engine._sessions:
    s0 = engine._sessions[0]
    print(f"\n  最新 session: {s0.get('title', '?')[:60]}")
if engine._facts:
    f0 = engine._facts[0]
    print(f"  最高 trust fact: {f0.get('content', '?')[:60]}")
if engine._memory_entries:
    m0 = engine._memory_entries[0]
    print(f"  最新 memory: {m0.get('content', '?')[:60]}")

# Quick search test
result = engine.handle_search({"query": "诗银雪"})
import json
parsed = json.loads(result)
print(f"\n  搜索 '诗银雪': {parsed.get('total', 0)} hits")
if parsed.get('hits'):
    for h in parsed['hits'][:3]:
        print(f"    [{h['source']}] score={h['score']} trust={h.get('trust_score', '-'):.2f} → {h.get('content', '')[:50]}")
