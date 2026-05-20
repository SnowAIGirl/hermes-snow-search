# Hermes Snow Search

> [English](README.md) | 中文版

Hermes Agent 的内存级并行搜索插件。全量加载到 RAM，多路并发，毫秒级返回。默认开启深度搜索，自动搜完整消息正文。

## 工作原理

1. **启动加载** — 后台线程自动加载
2. **全程驻内存** — 数据在 Python 列表中，搜索不走磁盘
3. **并行搜索** — ThreadPoolExecutor 多路并发
4. **增量更新** — post_tool_call 钩子捕获写入，追加缓存
5. **自动淘汰** — 超 80% 内存上限时淘汰最旧条目
6. **深度搜索** — 完整消息正文索引，含 session_id + timestamp + role，增量刷新

## 安装

```bash
pip install hermes-snow-search
hermes plugins enable hermes-snow-search
# 重启 Hermes
```

## 配置

```yaml
plugins:
  hermes-snow-search:
    memory_limit_mb: 500          # 安全上限，非实际开销
    session_max: 7000
    fact_max: 10000
    deep_search_enabled: true     # 设为 false 则仅用轻量模式
    deep_search_load_mode: "ondemand"   # "ondemand" | "startup"
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `memory_limit_mb` | 500 | 内存硬上限，达 80% 触发淘汰 |
| `session_max` | 7000 | 轻量缓存最大 session 数 |
| `fact_max` | 10000 | 最大事实条目数 |
| `deep_search_enabled` | true | 开启完整消息正文搜索。false 则仅用轻量模式 |
| `deep_search_load_mode` | ondemand | ondemand = 首次搜索时加载，startup = 启动时后台加载 |

> 500 MB 是安全上限，不是实际开销。一周真实对话（~230 session、~10,000 条消息）仅 ~6 MB。够存 1-2 年重度日常使用。

## 深度搜索

默认开启。激活后自动搜索完整消息正文替代轻量摘要。结果含 `session_id`、`timestamp`、`role`、`search_info`。

### 加载模式

| 模式 | 触发 | 表现 |
|------|------|------|
| `ondemand` | 首次搜索 | 阻塞加载，显示进度 |
| `startup` | 后台 2.5 秒 | 不阻塞，打印 ~0/50/100% 进度 |

```
[Hermes Snow Search] Loading deep search index...
[Hermes Snow Search] Session 58/231 | 2,500 messages | 10/500 MB | ~0.6s remaining
[Hermes Snow Search] Deep search ready | 10,229 messages | 7 days (May 13 ~ May 20) | 6 MB
```

从最新 session 反向加载，到 85% 内存上限停止。后续调用增量刷新，跨进程自动同步。

### 排序模式

| `sort` | 效果 |
|--------|------|
| `relevance` | 最佳匹配优先（相关度 + 近期加分） |
| `oldest` | 最早时间优先 — 回答"第一次" |
| `newest` | 最晚时间优先 — 回答"最近一次" |

### 性能

| 模式 | 搜索范围 | 延迟 | 内存（周数据） |
|------|----------|------|---------------|
| 轻量 | Session 摘要 | <0.5ms | ~3 MB |
| 深度 | 完整消息正文 | ~1-5ms | ~6 MB |

轻量与深度互斥——深度模式跳过 session 摘要，仅加载 facts + memory + messages。

## 注意事项

- **首次延迟：** 首次深度搜索触发索引构建（~1 秒/周数据）。
- **仅根会话：** 只索引顶层对话，子 agent 排除。
- **不索引工具输出：** 仅 user/assistant 角色消息。

## 使用建议

- "最近/上次"类问题天然命中第一条
- "第一次"类问题用 sort="oldest"
- 关键词越具体越好
- 跨进程自动同步，无需手动 reload
- 搜索覆盖全部内存数据，没找到就是没记录

## 作者

LinQuan & Snow (AI Girl)
