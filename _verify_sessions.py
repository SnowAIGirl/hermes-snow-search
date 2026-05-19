"""深挖 session 计数差异：看为什么是 227 不是 246。"""
from hermes_snow_search.tools import SnowSearchEngine


class MockCtx:
    def register_tool(self, *a, **kw): pass
    def register_hook(self, *a, **kw): pass


engine = SnowSearchEngine(MockCtx())

# Skip the cached load — go straight to the raw DB
from hermes_state import SessionDB

db = SessionDB()
raw_all = db.list_sessions_rich(
    limit=9999,
    exclude_sources=["tool"],
    order_by_last_active=True,
)

total_raw = len(raw_all)
excluded = sum(1 for s in raw_all if s.get("parent_session_id"))
child_ids = [s.get("id") for s in raw_all if s.get("parent_session_id")]
root = [s for s in raw_all if not s.get("parent_session_id")]

print(f"DB 原始总数 (exclude_sources=['tool']): {total_raw}")
print(f"  其中 parent_session_id 不为空:      {excluded}")
print(f"  单条 session 总数 (parent=None):     {len(root)}")
print(f"  session_max 配置默认:                {2000}")
print()

# Check what the engine actually loads
engine._ensure_loaded()
print(f"引擎加载 session 数: {len(engine._sessions)}")

# Is the diff because of the session_max filter? The engine uses:
# "SELECT COUNT(*) FROM facts" first to check availability, then fetch all
# Then: results[] loop with break if >= self._session_max
# The default is 2000, so that shouldn't be the issue.

# Let me check the actual loaded IDs vs raw IDs
loaded_ids = {s["session_id"] for s in engine._sessions}
raw_root_ids = {s.get("id") for s in db.list_sessions_rich(
    limit=9999, exclude_sources=["tool"], order_by_last_active=True
) if not s.get("parent_session_id")}

missing = raw_root_ids - loaded_ids
print(f"\n缺失 session 数: {len(missing)}")

if missing:
    # Print details of missing ones
    for s in db.list_sessions_rich(limit=9999, exclude_sources=["tool"]):
        if s.get("id") in missing:
            print(f"  - {s.get('id')}: {s.get('title', '?')[:60]} (parent={s.get('parent_session_id')}, msg_count={s.get('message_count')})")
