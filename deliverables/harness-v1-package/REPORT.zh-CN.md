# Data Agent Harness V1：结论、架构与复现报告

生成日期：2026-08-25
项目：KDD Cup 2026 Data Agents Starter Kit
范围：Harness V1、确定性 Semantic Compiler MVP、task 169、task 269、Raw/Semantic 对照与 Replay

## 1. 一句话结论

Harness 值得保留，但 **“要求模型先写 ASQ，再由模型同时给出 SQL”并不能解决 NL2SQL 的核心问题**。真正有效的边界是：

- LLM 负责理解问题，选择受治理的业务维度、指标和过滤值；
- Semantic Model 保存业务名到物理字段、指标表达式和 Join Graph 的映射；
- 确定性编译器独占物理 SQL 的生成权；
- Harness 负责状态、权限、版本、token、延迟、事件和 replay；
- 执行器只执行已批准、带指纹的计划。

这不是“不要 SQL”，而是“不让概率模型从原始 Schema 一步猜出最终 SQL”。

当前实现已在两个定点任务上验证这条链路可行，但覆盖范围仍很小，不能据此推断整个公开集的表现变化。后续应优先解决语义模型误匹配，再评估可复用的交易分析域。

## 2. “superhero 模型”到底是什么

这里的“模型”不是大语言模型，也不是经过训练的机器学习模型。

它是文件 `semantic_models/superhero.yaml` 中的一份 **版本化语义模型配置**，作用类似编译器的符号表、业务词典和 Join 规则库。它声明：

- 业务实体：hero、power；
- 业务维度：`hero.id`、`hero.name`、`power.id`、`power.name`；
- 物理来源：SQLite、CSV、JSON；
- 物理表达式：例如 `hero.name -> T1.superhero_name`；
- Join Graph：
  - `superheroes.id = hero_powers.hero_id`，`one_to_many`；
  - `hero_powers.power_id = superpowers.id`，`many_to_one`；
- 输出契约：`hero.name` 的最终列名必须是 `superhero_name`；
- 版本：`superhero-v1`；
- 匹配条件：任务 context 中必须存在声明的数据文件。

task 269 的 LLM 实际只需要生成：

```json
{
  "version": "1",
  "dimensions": ["hero.name"],
  "metrics": [],
  "filters": [
    {"field": "power.name", "op": "=", "value": "Death Touch"}
  ],
  "group_by": [],
  "order_by": [],
  "output": ["superhero_name"]
}
```

确定性编译器随后生成：

```sql
SELECT T1.superhero_name AS "superhero_name"
FROM superheroes AS T1
INNER JOIN hero_powers AS T2 ON T1.id = T2.hero_id
INNER JOIN superpowers AS T3 ON T2.power_id = T3.id
WHERE T3.power_name = 'Death Touch'
```

因此，superhero 只是验证三源 Join 的测试域，不代表项目要围绕超级英雄构建产品。

## 3. 原始方法给项目的关键启发

输入文章的核心不是某个榜单分数，而是重新分配职责：

| 职责 | 适合 LLM | 适合确定性系统 |
| --- | --- | --- |
| 理解自然语言意图 | 是 | 否 |
| 识别业务维度、指标、过滤条件 | 是 | 通过目录约束 |
| 业务字段绑定物理列 | 不应临场猜测 | 是 |
| 选择基础 Join 和连接键 | 不应临场猜测 | 是 |
| 指标计算口径 | 不应临场发明 | 是 |
| 数据库方言和物理 SQL | 尽量不负责 | 是 |
| 长尾窗口、递归、复杂组合 | 必要时参与 | 编译器逐步扩展 |
| 审计、权限、版本、成本 | 否 | Harness |

对应到本项目，V1 没有直接追求完整 Semantic Layer，而是先建立一个能做科学对照的 Harness，再加入最小确定性编译器。

## 4. 当前 Agent 是怎样构造的

### 4.1 总体结构

```text
自然语言问题
    │
    ▼
ReActAgent ──────────────► OpenAI-compatible Model
    │                           │
    │ 单步 JSON action          │ token / latency / response metadata
    ▼                           ▼
HarnessToolRegistry ◄── InstrumentedModelAdapter
    │
    ├── HarnessPolicy：模式、状态和工具权限
    ├── HarnessSession：统一 session/event/totals/version
    ├── Raw tools：读文件、SQLite、DuckDB、Python、answer
    └── Semantic tools
          ├── get_semantic_catalog
          ├── prepare_asq
          ├── execute_prepared_asq
          └── answer_prepared_asq
                    │
                    ▼
          Semantic Catalog + Compiler
                    │
                    ▼
          task-scoped DuckDB Executor
                    │
                    ▼
              AnswerTable + trace.json
```

### 4.2 ReAct 循环

`ReActAgent` 每轮要求模型返回一个 JSON 对象：

```json
{
  "thought": "本轮判断",
  "action": "一个工具名",
  "action_input": {}
}
```

每轮只执行一个工具，然后把 observation 追加回消息历史。循环直到调用终止工具或达到步数上限。

关键行为：

- raw 使用基础 ReAct system prompt；
- semantic 使用额外的语义协议 prompt；
- compiler-backed 任务要求第一步调用 `get_semantic_catalog`；
- 临近步数上限时，Agent 会收到必须提交终止答案的提示；
- 模型格式错误记录为 `__parse_error__`；
- 权限或工具错误记录为 `__tool_error__`；
- 最终答案统一为 `AnswerTable(columns, rows)`。

### 4.3 模型适配器

`OpenAIModelAdapter` 使用 OpenAI-compatible Chat Completions 接口，因此可连接兼容服务。它返回：

- 文本内容；
- prompt tokens；
- completion tokens；
- total tokens；
- response model；
- response id。

`InstrumentedModelAdapter` 在其外部记录 `model.started`、`model.completed` 或 `model.failed`，并测量墙钟延迟。

### 4.4 Runner 装配

单任务运行时，Runner 依次构造：

1. `PublicTask`；
2. 基础 Tool Registry；
3. `HarnessSession`；
4. `HarnessPolicy(raw|semantic)`；
5. `HarnessToolRegistry`；
6. `InstrumentedModelAdapter`；
7. `ReActAgent`；
8. 运行并把 answer、steps、failure 和 harness 写入 `trace.json`。

## 5. Harness V1 包含什么

用户限定的第一版目标均已落地。

### 5.1 统一 session/event 数据结构

状态集合：

```text
created
discovering
planning
compiling
validating
executing
submitted
failed
```

每个 `HarnessEvent` 包含：

- `sequence`；
- `event_type`；
- `timestamp`；
- `state_before`；
- `state_after`；
- `elapsed_ms`；
- `payload`。

`HarnessSession` 汇总 task、模式、模型、最大步数、起止时间、版本、事件、计划数量、失败原因和总计。

### 5.2 状态机和工具权限

raw 模式：

- 保留原 baseline 的文件、SQL、DuckDB、Python 和手工 answer 自由度；
- 作为对照组。

semantic 且存在确定性模型时：

- 禁止 raw SQLite/DuckDB SQL；
- 禁止任意 Python；
- 禁止手工 `answer`；
- 只允许执行 compiler 生成并注册的计划；
- `answer_prepared_asq` 必须携带已批准 fingerprint；
- session 进入终止态后禁止继续调用工具。

semantic 但没有确定性模型时：

- 暂时保留 legacy registered-plan fallback；
- 模型可以提交 ASQ 和物理 SQL，Harness 只允许之后执行完全相同的 SQL fingerprint；
- 该 fallback 只提供审计与一致性，不提供真正的 Schema Grounding 保证。

### 5.3 token、延迟和版本记录

每个 trace 记录：

- 模型调用数及具有 token usage 的调用数；
- 工具调用数；
- prompt/completion/total tokens；
- 模型总延迟；
- 工具总延迟；
- 端到端延迟；
- Harness 版本；
- Prompt 标签和 SHA-256；
- Tool description SHA-256；
- Semantic Model 标签；
- Policy 版本；
- 每份编译计划自己的 semantic model id/version/fingerprint。

### 5.4 raw 与 semantic 对照

CLI 支持：

```bash
dabench compare-modes --limit N --config CONFIG
dabench compare-runs RAW_RUN SEMANTIC_RUN
```

对照汇总 task 成功数、模型/工具调用、tokens、模型/工具延迟和端到端延迟。

注意：`succeeded_task_count` 只表示 Agent 生成了答案，不能替代官方评测结果。评测应单独通过 `evaluate.py` 判断。

### 5.5 ASQ/SQL replay

ASQ replay：

- 从 trace 提取 ASQ；
- 使用当前相同版本语义模型重新编译；
- 检查语义模型版本；
- 执行新计划；
- 不调用 LLM。

SQL replay：

- 直接使用 trace 中记录的物理计划；
- 不重新进行语义解析；
- 不调用 LLM。

两种 replay 都记录源 trace SHA-256，便于审计“结果来自哪一次运行”。

## 6. 确定性 Semantic Compiler 是怎样构造的

### 6.1 数据结构

- `SourceDefinition`：名称、类型、相对路径、别名、relation、主键；
- `DimensionDefinition`：业务名、来源、物理表达式、数据类型、输出标签、描述；
- `MetricDefinition`：业务名、依赖来源、物理表达式、数据类型、输出标签、描述；
- `JoinDefinition`：左右来源、连接表达式、基数；
- `SemanticModel`：模型 id、版本、匹配签名、sources、dimensions、metrics、joins；
- `AnalysisSemanticQuery`：dimensions、metrics、filters、group_by、order_by、output、limit；
- `CompiledSemanticPlan`：ASQ、模型版本、SQL、来源、Join path、输出列和 fingerprint。

### 6.2 编译步骤

```text
ASQ
 │
 ├─ 结构校验：版本、字段类型、output、limit
 ├─ 语义名校验：维度、指标、过滤和排序字段必须存在
 ├─ 来源推导：从所选字段计算 required sources
 ├─ Join Graph 搜索：自动找到路径并纳入桥表
 ├─ Grain/Fanout 校验：阻止已知会放大指标的基数路径
 ├─ 输出契约校验：列数和列名必须受模型治理
 ├─ SQL 生成：SELECT/FROM/JOIN/WHERE/GROUP/ORDER/LIMIT
 └─ 指纹：model id + version + ASQ + SQL + sources 的 SHA-256
```

task 269 促使 Join 解析从“只连接直接需要的表”升级为图搜索：即使 `hero_power.csv` 不出现在选择或过滤字段中，也会作为受治理桥表自动加入。为保证 replay 和指纹稳定，搜索顺序按语义模型声明顺序确定，不能依赖 Python set 的遍历顺序。

### 6.3 执行器

每个计划在内存 DuckDB 中执行：

- SQLite：通过只读 URI 使用 `sqlite3` 读取 relation，再注册为 pandas DataFrame；
- CSV：DuckDB 自动检测并创建 view；
- JSON：支持顶层 record list，或 `{ "table": ..., "records": [...] }` 包装格式；包装 JSON 经确定性展开后注册；
- SQL：只来自 `CompiledSemanticPlan`；
- 输出：保留治理后的列名，限制最大行数和单元格预览长度。

## 7. 两份现有语义模型

### 7.1 `debit_card.yaml`

用途：task 169。

来源：

- SQLite `customers`；
- CSV `yearmonth`。

定义：

- customer segment、currency、month、year 等维度；
- `debit_card.average_monthly_consumption` 指标；
- CustomerID 一对多 Join；
- Benchmark 要求的精确输出列；
- 明确浮点聚合输入顺序，避免 DuckDB 因并行/输入顺序产生约 1e-12 的差异。

### 7.2 `superhero.yaml`

用途：task 269 的 SQLite + CSV + JSON 三源技术验证。

来源：

- SQLite superhero；
- CSV hero_power 桥表；
- JSON superpower。

它是测试夹具，不是推荐继续投资的业务域。

## 8. 实验结果

### 8.1 第一阶段：只有 Harness 协议，没有真正编译器

样本：公开集前 5 题，run `20260824T111412Z`。

| 指标 | Raw | Legacy Semantic |
| --- | ---: | ---: |
| 生成答案的任务 | 5/5 | 5/5 |
| 模型调用 | 53 | 64 |
| 工具调用 | 43 | 43 |
| Total tokens | 366,845 | 464,943 |
| 端到端时间 | 253.203s | 245.888s |

Semantic tokens 增加 98,098，约 +26.7%。该对照仅用于说明协议开销和行为差异，不作为本项目的公开评测结果。

结论：

- 仅增加 ASQ 结构和“先注册再执行”的权限门禁，不会自动保证答案质量；
- 如果物理 SQL 仍由 LLM 生成，字段、Join、粒度和输出错误仍然存在；
- 协议增加了额外消息和格式负担，可能让 token 和解析错误变多；
- Harness 本身有价值，因为它让这个失败可以被精确量化，而不是凭感觉讨论。

### 8.2 第二阶段：task 169 确定性指标编译

问题：2013 年 SME 客户的平均月消费。

| 指标 | Raw | Compiler-backed Semantic |
| --- | ---: | ---: |
| 技术验证 | 未通过 | 通过 |
| 模型调用 | 10 | 11 |
| 工具调用 | 9 | 7 |
| Total tokens | 55,951 | 53,249 |
| 端到端时间 | 148.532s | 31.950s |

Semantic：

- token 降低 2,702，约 -4.8%；
- 端到端时间降低 116.582s，约 -78.5%；
- 模型调用多 1 次，主要来自格式解析重试；
- 最关键的是物理计划和输出契约均通过了技术验证。

Raw 的主要问题不是 SQL 无法运行，而是指标口径和输出契约不一致；语义层通过显式治理这些约束来降低歧义。

### 8.3 第三阶段：task 269 三源 Join

问题：列出具有 Death Touch 能力的超级英雄名称。

最终有效对照：raw `20260824T115215Z`，semantic `20260824T115530Z`。

| 指标 | Raw | Compiler-backed Semantic |
| --- | ---: | ---: |
| 技术验证 | 通过 | 通过 |
| 模型调用 | 6 | 4 |
| 工具调用 | 6 | 3 |
| Prompt tokens | 23,029 | 12,855 |
| Completion tokens | 694 | 426 |
| Total tokens | 23,723 | 13,281 |
| 端到端时间 | 18.067s | 11.907s |

Semantic：

- total tokens 降低 10,442，约 -44.0%；
- 端到端时间降低 6.160s，约 -34.1%；
- 模型调用减少 2 次；
- 工具调用减少 3 次；
- 核心工具序列为 `get_semantic_catalog -> prepare_asq -> answer_prepared_asq`；
- 中间仍出现 1 次 JSON 格式解析错误，说明未来应考虑原生 tool calling。

最初的 semantic 实跑仍按 raw 习惯读取了四次原始数据，直到 Python 被权限门禁阻止。将“第一步必须读语义目录”前置到 system prompt，并把 semantic tools 放在工具描述开头后，才得到上述结果。这证明 Prompt 顺序也应当作为可版本化实验变量。

### 8.4 Replay

task 169 和 task 269 均完成：

- ASQ replay：完成，模型调用 0；
- SQL replay：完成，模型调用 0。

Replay 的价值不只是省 token，还能区分：

- ASQ 在新语义模型版本下是否仍能编译；
- 旧物理计划是否仍能复现旧结果；
- 结果变化来自 LLM、语义资产、编译器还是底层数据。

## 9. 当前可以确认的结论

1. **Harness 值得加入。** 即使语义路径未带来公开评测结果变化，它也提供可比较的证据、权限边界和可复现性。
2. **ASQ 不是魔法。** 如果 SQL 仍由模型写，ASQ 只是另一层输出格式。
3. **语义资产和编译器负责治理关键约束。** 字段、Join、指标、粒度和输出契约都必须被显式定义。
4. **“模型不直接写 SQL”应通过权限实现，不只靠 Prompt。** compiler-backed 任务已禁止 raw SQL、Python 和手工答案。
5. **输出列也是业务契约。** Benchmark 中正确数值配错误列名仍会失败。
6. **Join Graph 必须包含基数。** 只有连接键而没有 one-to-many/many-to-one 无法做 fanout 风险判断。
7. **桥表必须由编译器自动发现。** 不应要求 LLM 在 ASQ 中显式选择一个纯物理桥表。
8. **版本和 fingerprint 必不可少。** 否则 replay 不能回答“重放的是哪套定义”。
9. **Agent 成功与答案正确必须分开统计。** `succeeded=True` 只说明提交了表格。
10. **单题成功不能证明泛化。** 当前 compiler-backed 证据只有 task 169 和 task 269。
11. **语义工程成本是真成本。** 当前的技术验证依赖人工定义的字段、Join、指标和输出契约，而不是自动涌现。
12. **后续应优先验证可复用业务域。** task 269 已完成跨源技术验证，下一步可转向交易分析等高复用业务域。

## 10. 已知限制和风险

### P0：模型匹配存在误匹配风险

当前 `SemanticCatalog.match_task` 只检查 `required_paths` 是否都存在。

例如 task 287 也包含 task 269 的三个文件，另外还有 gender 数据。当前 catalog 可能匹配 `superhero-v1`，但该模型没有 `hero.gender`，从而出现“匹配成功、能力不足”。

下一步应把匹配升级为 capability-aware：

- 记录任务所需 semantic capabilities；
- 支持 context signature、优先级或更具体模型覆盖；
- 区分 required、optional 和 unsupported source；
- 在目录返回前验证所需维度/指标是否可满足；
- 把“为什么匹配/为什么拒绝”写入 trace。

### ASQ 表达能力有限

当前 V1 支持：

- dimensions、metrics；
- AND filters；
- `= != > >= < <= in between`；
- group_by、order_by、limit。

尚不支持：

- OR / NOT 和嵌套条件；
- distinct 语义；
- 窗口函数；
- 子查询、CTE、递归；
- 多阶段指标和 semi-additive metric；
- 明确时间 grain、时区和财务周期；
- 多条等价 Join path 的成本选择。

### 数据与执行限制

- JSON 只覆盖 record list 和 `records` 包装；
- SQLite 当前先整体读取 relation 到 pandas，再注册 DuckDB，不适合大表；
- 基数来自模型声明，尚未做运行时数据质量验证；
- 没有查询成本预算、扫描量限制和超时分级；
- 没有字段级、行级和指标级真实企业权限；
- 没有 schema drift 检测和自动失效机制。

### Agent 协议限制

- 当前通过 fenced JSON 解析 action，真实运行出现过 parse error；
- 更稳妥的方向是模型原生 tool calling 或严格结构化输出；
- semantic fallback 的安全性和正确性弱于 compiler-backed 路径；
- Prompt 前置能改善行为，但不能代替程序权限。

## 11. 推荐路线图

### 下一小目标（推荐）

**修复 capability-based 模型匹配，并建立 transaction analytics 语义域。**

优先任务：

1. task 173：SQLite transactions + CSV yearmonth + JSON gasstations；
2. task 180：复用交易和月度数据，加入产品单价与消费状态；
3. task 169：作为客户/月度指标回归；
4. task 269：仅作为三源桥表回归。

选择交易域的理由：

- 客户、交易、产品、门店、日期是常见企业实体；
- 能验证事实表、维表、时间粒度和跨源 Join；
- task 173 与 180 能检验同一语义资产是否真正复用；
- 有助于验证可复用的产品能力。

### 随后应做

1. 数据集 coverage scanner：输出每个任务的数据源签名和可用 semantic capabilities；
2. 5~10 题固定回归集，而不是只看单题；
3. correctness、token、latency、tool calls、compile failures 同时作为指标；
4. ASQ/SQL replay 加入 CI；
5. semantic model 变更必须 bump version，并运行影响分析；
6. 引入原生 structured tool calling，减少解析重试；
7. 再逐步支持 OR、distinct、窗口和多阶段查询。

## 12. 如何复现

### 安装与检查

```bash
uv sync
uv run dabench status --config configs/react_baseline.local.yaml
```

本地配置可以使用环境变量，避免把 key 写进文件：

```yaml
agent:
  api_key: ${DABENCH_API_KEY}
```

### 运行单题

```bash
.venv/bin/dabench run-task task_169 --config configs/react_baseline.local.yaml
.venv/bin/dabench run-task task_269 --config configs/react_baseline.local.yaml
```

修改 `harness.mode` 为 `raw` 或 `semantic` 即可切换模式。项目当前默认已恢复为 `raw`。

### 对比已有运行

```bash
.venv/bin/dabench compare-runs \
  artifacts/runs/20260824T115215Z \
  artifacts/runs/20260824T115530Z
```

### Replay

```bash
.venv/bin/dabench replay-task task_269 \
  --trace artifacts/runs/20260824T115530Z/task_269/trace.json \
  --kind asq \
  --config configs/react_baseline.local.yaml

.venv/bin/dabench replay-task task_269 \
  --trace artifacts/runs/20260824T115530Z/task_269/trace.json \
  --kind sql \
  --config configs/react_baseline.local.yaml
```

### 测试与静态检查

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
```

当前结果：11 项测试通过，Ruff 和 compileall 通过。

## 13. 代码地图

| 路径 | 责任 |
| --- | --- |
| `src/data_agent_baseline/agents/react.py` | ReAct 循环、解析、步数与终止 |
| `src/data_agent_baseline/agents/prompt.py` | raw/semantic system prompt |
| `src/data_agent_baseline/agents/model.py` | OpenAI-compatible 模型适配 |
| `src/data_agent_baseline/run/runner.py` | 运行时装配、trace/prediction 写入 |
| `src/data_agent_baseline/harness/events.py` | 状态和事件结构 |
| `src/data_agent_baseline/harness/session.py` | session、版本、totals |
| `src/data_agent_baseline/harness/telemetry.py` | 模型 token/延迟遥测 |
| `src/data_agent_baseline/harness/policy.py` | raw/semantic 工具权限和 SQL fingerprint |
| `src/data_agent_baseline/harness/tools.py` | semantic tools 和编译计划执行 |
| `src/data_agent_baseline/harness/replay.py` | ASQ/SQL replay |
| `src/data_agent_baseline/harness/experiment.py` | raw/semantic 对照汇总 |
| `src/data_agent_baseline/semantic/schema.py` | Semantic Model、ASQ、Plan 类型 |
| `src/data_agent_baseline/semantic/catalog.py` | YAML 加载和任务匹配 |
| `src/data_agent_baseline/semantic/compiler.py` | ASQ 校验、Join 图搜索、SQL 和指纹 |
| `src/data_agent_baseline/semantic/executor.py` | SQLite/CSV/JSON 到 DuckDB 的确定性执行 |
| `semantic_models/debit_card.yaml` | task 169 语义资产 |
| `semantic_models/superhero.yaml` | task 269 三源技术验证资产 |
| `tests/test_harness.py` | Harness 和 legacy replay 测试 |
| `tests/test_harness_config.py` | 配置测试 |
| `tests/test_semantic_compiler.py` | task 169/269 编译、权限和 replay 测试 |

## 14. 最终判断

当前项目已经从“一个能调用工具的 ReAct baseline”变成“一个可以比较、约束、审计和重放的数据 Agent 实验平台”，并拥有最小可工作的业务编译器。

不应将两个定点任务的结果外推为语义层已经泛化。实际状态应描述为：

> Harness V1 已完成；确定性 Semantic Compiler MVP 已在一个指标题和一个三源桥表题上验证；legacy semantic 已被实验证明无效；下一阶段需要 capability-aware matching、可复用交易域和多题回归。

这也是后续决策的基准线。
