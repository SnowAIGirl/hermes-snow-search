"""深挖 memory 计数：看看为什么是 38 不是 36。"""
from hermes_snow_search.tools import SnowSearchEngine


class MockCtx:
    def register_tool(self, *a, **kw): pass
    def register_hook(self, *a, **kw): pass

engine = SnowSearchEngine(MockCtx())
engine._ensure_loaded()

print(f"Memory entries loaded: {len(engine._memory_entries)}")
print()

# Check sources
from collections import Counter
sources = Counter(m["source"] for m in engine._memory_entries)
for src, cnt in sources.most_common():
    print(f"  {src}: {cnt}")

print("\n--- 按源展示 ---")
for m in engine._memory_entries:
    content_short = m["content"][:80].replace("\n", "|")
    print(f"  [{m['source']}] {content_short}")
