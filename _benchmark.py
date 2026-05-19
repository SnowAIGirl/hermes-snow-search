"""性能测试：内存搜索 speed benchmark"""
import time
import json
from hermes_snow_search.tools import SnowSearchEngine


class MockCtx:
    def register_tool(self, *a, **kw): pass
    def register_hook(self, *a, **kw): pass


engine = SnowSearchEngine(MockCtx())

# 1. 冷启动加载时间
print("=== 1. 冷启动加载 ===")
t0 = time.perf_counter()
engine._ensure_loaded()
t1 = time.perf_counter()
print(f"    耗时: {(t1-t0)*1000:.1f} ms")
print(f"    数据: {len(engine._sessions)} sessions + {len(engine._facts)} facts + {len(engine._memory_entries)} memory")

# 预加载后，后面 search 都是纯内存操作
print("\n=== 2. 搜索性能（每轮查 3 次取均值） ===")

queries = [
    ("通用词", "Hermes"),
    ("小雪自己", "诗银雪"),
    ("项目名", "灵台"),
    ("技术相关", "deprecated"),
    ("英文/配置", "config"),
    ("用户", "泉哥"),
    ("事实查询", "trust_score"),
    ("长尾词", "snow_search"),
]

for label, q in queries:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        result = engine.handle_search({
            "query": q,
            "limit_per_source": 5,
            "include_sessions": True,
            "include_facts": True,
            "include_memory": True,
        })
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    parsed = json.loads(result)
    avg = sum(times) / len(times)
    print(f"  [{label}] \"{q}\": avg={avg:.2f}ms, best={min(times):.2f}ms, hits={parsed['total']}")

print("\n=== 3. 压力测试：空查询（无匹配） ===")
t0 = time.perf_counter()
result = engine.handle_search({"query": "zzzznotexist999"})
t1 = time.perf_counter()
parsed = json.loads(result)
print(f"    耗时: {(t1-t0)*1000:.2f}ms, hits={parsed['total']}")

print("\n=== 4. 压力测试：宽查全源 ===")
t0 = time.perf_counter()
result = engine.handle_search({
    "query": "plugin",
    "limit_per_source": 20,
})
t1 = time.perf_counter()
parsed = json.loads(result)
print(f"    耗时: {(t1-t0)*1000:.2f}ms, hits={parsed['total']} (limit_per_source=20)")

print("\n=== 5. 重复调用稳定性（连续 10 次相同查询） ===")
times = []
q = "Hermes"
for _ in range(10):
    t0 = time.perf_counter()
    engine.handle_search({"query": q})
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
print(f"    平均: {sum(times)/len(times):.2f}ms, 最大: {max(times):.2f}ms, 最小: {min(times):.2f}ms")
