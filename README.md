# WeChat Agent

基于 ReAct 的企业微信智能助手示例，支持：

- 本地 CLI 对话（`agent.py`）
- 企业微信回调服务（`wechat_robot_server.py`）
- 本地知识库 RAG 检索问答
- MCP Streamable HTTP 远端工具接入
- Skill 自动发现、检索与注入
- 会话持久化、消息去重、流式状态管理
- 健康检查与 Prometheus 指标

---

## 功能概览

- **ReAct Agent**：模型按 `<thought>/<action>/<observation>/<final_answer>` 循环执行。
- **工具体系**：内置 `get_current_date`、`rag_summarize`，并可扩展 MCP 与 Skill 工具。
- **企业微信流式回复**：`text` 消息返回 `stream_id`，再通过 `stream` 轮询结果。
- **技能系统**：从 `skills/*/SKILL.md` 自动发现技能并注入提示词上下文。
- **可观测性**：内置 `/healthz` 与 `/metrics`。

---

## 项目结构

```text
.
├─ agent.py                         # CLI 入口（ReActAgent）
├─ wechat_robot_server.py           # 企业微信服务入口（Flask）
├─ build_rag_index.py               # 构建/更新 RAG 索引
├─ requirements.txt
├─ pyproject.toml
├─ .env.example
├─ config/
│  └─ mcp_servers.json
├─ app/
│  ├─ agent/                        # runtime / model / parser / prompt
│  ├─ application/conversation/     # ChatStore / Session / Stream / Dedup
│  ├─ channel/wecom/                # 渠道路由、加密适配
│  ├─ contracts/
│  ├─ infrastructure/               # 日志、缓存、指标
│  ├─ mcp/                          # MCP client + registry
│  └─ skills/                       # Skill 解析、检索、注入、执行
├─ tools/                           # 本地工具 + MCP 工具封装
├─ rag/                             # 向量库、检索总结、prompt
├─ wecom/                           # 企微协议构建/解析
├─ wxwork/                          # 企业微信加解密实现
├─ db/
└─ skills/                          # 示例技能
```

---

## 运行前提

- Python `>=3.10`
- 已配置可用的 OpenAI 兼容模型接口（`OPENAI_BASE_URL` + API Key + 模型名）
- 如果要启动企业微信服务，需可用 MySQL（当前实现中为硬依赖）

安装依赖：

```bash
python -m pip install -r requirements.txt
```

---

## 环境变量

先复制模板：

### Linux / macOS
```bash
cp .env.example .env
```

### Windows PowerShell
```powershell
Copy-Item .env.example .env
```

### 1) 最小必填（CLI）

```env
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENROUTER_MODEL=your-model-name
OPENROUTER_API_KEY=your-key
# 或者用 OPENROUTER_API_KEYS=key1,key2

REACT_MAX_STEPS=12
REACT_MAX_HISTORY_MESSAGES=20
REACT_MAX_MEMORY_TURNS=8
REACT_MEMORY_TEXT_LIMIT=240
REACT_PROMPT_MAX_FILES=30
SKILL_SHORTLIST_LIMIT=3
REACT_MODEL_CONNECT_RETRIES=2
```

### 2) 企业微信服务必填（在 CLI 基础上增加）

```env
WECHAT_ROBOT_TOKEN=your-token
WECHAT_ROBOT_ENCODING_AES_KEY=your-encoding-aes-key
WECHAT_ROBOT_RECEIVE_ID=
WECHAT_SERVER_HOST=0.0.0.0
WECHAT_SERVER_PORT=8085
WECHAT_WORKER_MAX=8
WECHAT_MAX_PROCESS_SECONDS=180
WECHAT_USE_WAITRESS=true
WECHAT_ROBOT_WELCOME_TEXT=欢迎进入智能助手。你可以直接提问，我会用流式消息回复你。
WECHAT_REPLY_DEBUG_LOG=false
WECHAT_LOG_MESSAGE_CONTENT=false
LOG_LEVEL=INFO
```

### 3) 会话与存储（服务端强相关）

```env
SESSION_TTL_SECONDS=1800
STREAM_TTL_SECONDS=600
SESSION_LOCK_TTL_SECONDS=180
CHAT_HISTORY_LOAD_LIMIT=10
CHAT_MESSAGE_RETENTION_DAYS=90
CHAT_SESSION_INACTIVE_RETENTION_SECONDS=21600
CHAT_CLEANUP_INTERVAL_SECONDS=300
CHAT_MESSAGE_DEDUP_TTL_SECONDS=600

MYSQL_ENABLED=true
MYSQL_HOST=your-mysql-host
MYSQL_PORT=3306
MYSQL_USER=your-mysql-user
MYSQL_PASSWORD=your-mysql-password
MYSQL_DATABASE=your-mysql-database
MYSQL_CONNECT_TIMEOUT=3
MYSQL_POOL_SIZE=8
```

### 4) 可选能力

```env
# RAG
RAG_ENABLED=true
RAG_TOP_K=4
RAG_DATA_DIR=knowledge
RAG_PERSIST_DIR=.rag_store
RAG_COLLECTION_NAME=knowledge
RAG_CHUNK_SIZE=700
RAG_CHUNK_OVERLAP=120
RAG_ALLOWED_TYPES=.txt,.md,.pdf
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_CHAT_MODEL=your-rag-chat-model

# Redis（可选增强，失败会降级）
REDIS_ENABLED=false
REDIS_KEY_PREFIX=agent_demo
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_SOCKET_TIMEOUT_SECONDS=2
REDIS_CONNECT_TIMEOUT_SECONDS=2

# Skill 脚本执行（默认关闭）
SKILL_ENABLE_SCRIPTS=false
SKILL_SCRIPT_TIMEOUT_SECONDS=20
SKILL_SCRIPT_OUTPUT_LIMIT=4000
```

---

## 启动方式

推荐顺序：

1. 配置 `.env`
2. 安装依赖
3. （可选）构建 RAG 索引
4. 先跑 CLI 验证模型与工具
5. 再跑企业微信服务联调

构建 RAG 索引（可选）：

```bash
python build_rag_index.py
```

启动 CLI：

```bash
python agent.py
```

CLI 命令：

- `/clear`：清空上下文
- `/exit`：退出

启动企业微信服务：

```bash
python wechat_robot_server.py
```

---

## HTTP 接口

- `GET /api/wechat/robot`：企业微信 URL 验证
- `POST /api/wechat/robot`：企业微信主回调（加密协议）
- `POST /api/wechat/robot/chat`：本地明文调试接口
- `GET /healthz`：健康检查
- `GET /metrics`：Prometheus 指标

`POST /api/wechat/robot/chat` 示例：

```json
{
  "user_id": "u1",
  "session_id": "s1",
  "message": "今天几号"
}
```

---

## 企业微信消息流程

1. 企业微信请求进入 `POST /api/wechat/robot`
2. 服务验签并解密消息，统一转换为 `InboundMessage`
3. `text` 消息创建 `stream_id`，立即回占位内容
4. 后台线程执行 Agent（可调用本地/RAG/MCP/Skill 工具）
5. 客户端通过 `stream` 持续拉取增量结果，直到 `finish=true`

---

## MCP 工具接入（仅 Streamable HTTP）

支持两种配置来源：

- 环境变量 `MCP_SERVERS_JSON`
- 文件 `config/mcp_servers.json`

兼容格式：

- 标准 MCP `{"mcpServers": {...}}`
- 项目数组格式 `[{"name":"...","url":"..."}]`

注意：

- 当前仅支持带 `url` 的 Streamable HTTP MCP。
- `command` / `args`（stdio 模式）会被跳过。
- 工具会映射为 `mcp_<server_name>_<tool_name>`。
- 工具缓存位置：`.mcp_cache/tool_cache.json`。

---

## Skill 系统

基于 `skills/<skill-name>/SKILL.md`：

- 启动时自动发现技能
- 每轮请求检索候选技能并注入上下文
- 支持工具访问技能资源：
  - `list_skill_resources(skill_name)`
  - `load_skill_reference(skill_name, reference_path)`
  - `run_skill_script(skill_name, script_path, args)`（需启用脚本执行）

示例技能：

- `skills/knowledge-rag-answer`
- `skills/testcase-generator`

---

## 常见问题

- **服务启动时报 MySQL 配置错误**：企业微信服务会初始化 `ChatStore`，请确保 MySQL 相关变量完整且可连通。
- **缺少 `REACT_*` 导致启动失败**：这些参数在代码里按 `int(os.getenv(...))` 读取，建议全部显式配置。
- **MCP 工具未出现**：确认配置为 `url` 形式，且远端服务可访问。
- **RAG 首次较慢**：首次或文档变化时会进行索引构建，属正常现象。

---

## 开源与合规

- 许可证：MIT（见 `LICENSE`）
- 中文许可证参考译文：见 `LICENSE.zh-CN.md`（仅供阅读，法律效力以英文原文为准）
- 安全漏洞反馈：见 `SECURITY.md`
- 贡献说明：见 `CONTRIBUTING.md`
- 社区行为准则：见 `CODE_OF_CONDUCT.md`
- 第三方实现说明：见 `THIRD_PARTY_NOTICES.md`

---

## 说明

- 当前主链路聚焦企业微信文本与流式问答。
- 内置工具保持精简（时间查询 + RAG）；复杂能力建议通过 MCP/Skill 扩展。
