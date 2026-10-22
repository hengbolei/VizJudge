# VizJudge

[English](README.md) | **简体中文**

**生成候选图表，判断哪些图表值得关注，并把这些证据转化为机器学习的下一步行动。**

VizJudge 不是又一个自动 EDA 报告生成器。它关心的核心问题不是"能画多少张图"，而是
**"哪些图有用，它们对数据质量、特征工程和建模意味着什么？"**

> 面向机器学习流程的图表价值判断器：少画图、看重点、给出下一步。

## 输出示例

以下是在 Breast Cancer Wisconsin 数据集上的真实输出（569 行、30 个特征，共评判 60 张候选图，
最终渲染 6 张）。完整报告见 [`outputs/sample_report`](outputs/sample_report/summary.md)。

<img src="outputs/sample_report/01-target_box__diagnosis__worst_perimeter__diagnosis.png" alt="worst_perimeter by diagnosis" width="420">

```text
- Signal: The best validated feature beats the permutation null (p = 0.020).

### 1. worst_perimeter by target diagnosis — 92.4/100
- strong group separation by eta (0.78; eta squared 0.61)
- Held-out roc_auc gain over baseline: 0.4730 (positive is better); held-out strength 0.95 vs in-sample relevance 0.78.
- high confidence: positive incremental ridge-baseline gain in every held-out fold.

### 2. worst_concave_points by target diagnosis — 79.4/100
- Held-out roc_auc gain over baseline: 0.4643 (positive is better); held-out strength 0.47 vs in-sample relevance 0.78.
- Selection score: 74.4 (diversity penalty: 5.0): this chart family is already represented.
```

离线复现命令：
`uv run vizjudge agent examples/breast_cancer/breast_cancer.csv --target diagnosis --task classification --seed 42 --output outputs/sample_report`。

## 与同类工具的区别

| 工具 | 侧重点 | VizJudge 的补充 |
| --- | --- | --- |
| [dabl](https://github.com/dabl/dabl) | 基于简单监督启发式、面向目标变量的图表 | 对每张图做固定随机种子的交叉验证*增量*收益评估，加入冗余与多样性惩罚、置信度标签，并输出行动报告（数据质量、特征、建模） |
| [AutoViz](https://github.com/AutoViML/AutoViz) | 内部做特征选择后自动绘图 | 选择依据完全透明：每张图的得分、各折收益和理由都写在 `report.json` 中，而不是黑盒过滤 |
| [LIDA](https://github.com/microsoft/lida) | 由 LLM 生成分析目标、图表代码并评判图表质量 | 统计验证优先；LLM Judge 是可选的，且只能引用确定性流程产生的证据 ID |
| [ydata-profiling](https://github.com/ydataai/ydata-profiling) | 逐列和两两变量的全量画像报告 | 刻意少画图：对候选图排序，只渲染最值得建模者关注的前 k 张 |

这些工具是互补的：需要全面画像时用 profiler，需要决定哪些可视化证据应该改变下一步建模时用 VizJudge。

## 面向宿主 Agent 的统计工具集

VizJudge 的核心是确定性的统计判断：候选图表、信号指标、留出集上的增量收益、多样性选择和稳定的证据 ID。
Claude Code 这类宿主 Agent 本身已经拥有 LLM，以及用户的领域背景和代码，所以 VizJudge 被设计成**由它们来调用**：
[skill](skills/vizjudge/SKILL.md) 让宿主模型担任 Judge，细粒度的 MCP 工具
（`profile_dataset` → `generate_candidates` → `validate_features` → `select_charts` → `render_charts`）
提供宿主必须引用的证据。宿主只能降低统计优先级、不能提高，并负责把保留下来的证据转化为用户项目中的具体改动。

在需要可复现的批量运行和 CI 场景下，`vizjudge agent` 会以确定性方式完整执行
**规划 → 画像 → 分析 → 评判 → 改进 → 渲染** 循环，并带有执行预算和运行轨迹：

```bash
vizjudge agent data.csv --target outcome --task classification --output outputs/agent
python examples/agent_loop/demo.py --output outputs/demo-v04
```

批量循环默认只使用统计判断。LLM Judge 是可选的：可以回放录制好的 JSON 响应，也可以配置自己的 HTTP Judge 端点。
演示使用固定种子的合成数据和模拟的离线 Judge，不需要任何凭据。详见
[Agent 工作流、宿主工具、Judge 约定与预算](docs/agent.md)、[演示](examples/agent_loop/README.md)
和[更新日志](CHANGELOG.md)。

## 分析能力

1. 读取 CSV、JSON/JSONL 或 Parquet 数据。
2. 推断数值、类别、日期时间、标识符、常量和自由文本等列角色。
3. 生成精简的候选图表规格。
4. 结合信号强度、覆盖率、可操作性和固定种子的增量预测验证为候选图打分，并施加多样性与冗余惩罚。
5. 只渲染价值最高的图表。
6. 把结构化观察结果和机器学习建议写入 JSON 和 Markdown。
7. 通过 Python、CLI 和 MCP 提供细粒度的宿主 Agent 工具、直接分析接口和批量 Agent 循环。

VizJudge 的得分是决策辅助，而不是统计证明。它们有意保持透明和确定性，详见 [评分说明](docs/scoring.md)。
置信度标签描述的是用于排序的证据强弱，而不是统计概率。

## 快速开始

推荐使用 [`uv`](https://docs.astral.sh/uv/) 作为包管理器：它能快速创建可复现的环境，并在一个工具里管理项目、锁文件、工具和发布流程。

```bash
uv sync --extra dev
uv run vizjudge analyze data.csv --target survived --output outputs/run
```

传统的 `pip` 也可以：

```bash
python -m venv .venv
python -m pip install -e .
vizjudge analyze data.csv --target survived --output outputs/run
```

读取 Parquet 需要可选依赖 `parquet`：`uv sync --extra parquet`。

输出目录包含 `report.json`、`summary.md` 以及排名靠前的图表 PNG 文件；Agent 运行还会写出 `execution.json`。
使用 `--no-render` 只做结构化分析。`--task regression` 或 `--task classification` 可以消除目标变量类型的歧义；
重复使用 `--role COLUMN=ROLE` 可以按领域知识覆盖列角色。

## 在 Agent 中使用

同一个包同时提供 MCP server 和 skill：

```bash
pip install "vizjudge[mcp] @ git+https://github.com/hengbolei/VizJudge"
vizjudge skill install                  # 把 skill 复制到 ~/.agents/skills/vizjudge
vizjudge mcp-config --client dsh        # 或：--client json | --client claude
vizjudge doctor                         # 检查解释器、mcp 和已安装的 skill
```

`mcp-config` 会针对当前解释器输出注册配置（`python -m vizjudge.mcp.server`），因此宿主无需配置 PATH 或虚拟环境。
`--client json` 输出许多宿主通用的 `mcpServers` 格式。分步接入指南见 [DeepSeek Harness](docs/integrations/deepseek-harness.md)。

## MCP server

MCP 集成基于稳定的 2.x 版 Python SDK。服务通过 stdio 通信，由宿主 Agent 以
`python -m vizjudge.mcp.server`（或 `vizjudge-mcp`）启动。

| 工具 | 用途 |
| --- | --- |
| `profile_dataset` | 列角色、缺失情况、重复记录、数据质量建议 |
| `generate_candidates` | 所有候选图表及其得分、警告、图表 ID 和证据 ID |
| `validate_features` | 指定特征的交叉验证增量收益 |
| `select_charts` | 多样化且经过验证的图表选择；接受宿主给出的评判和优先级，但只能降低优先级 |
| `render_charts` | 在图片预算内渲染选定的图表 ID |
| `analyze_dataset` | 一次性完整分析（保持不变，用于兼容） |
| `run_agent` | 完整的确定性批量循环，带预算和可选的 Judge 适配器 |

每个工具都接受一个 `budget` 对象（行数、候选数、超时、图片数、输入字节数），并在受限的子进程中运行。
请只分析你有权读取的数据集；生成的文件会写入指定的输出目录。

## Agent Skill

skill 位于 [`skills/vizjudge`](skills/vizjudge)，并打包在 wheel 中。
`vizjudge skill install` 会把它复制到 `~/.agents/skills`（也可用 `--target dsh|claude`、
`--project` 安装到当前项目，或 `--dest DIR` 指定目录）。Codex 可以在仓库中以 `$vizjudge` 调用它。
skill 会先检查自身的前置条件，在 MCP 工具未注册时回退到 CLI。
它让宿主模型担任 Judge：规定推荐的工具调用顺序、证据引用规则、基于列语义的数据泄漏审查，以及如何把结果转化为代码改动。

## CLI 参考

```text
vizjudge analyze DATASET [--target COLUMN] [--output DIR] [--top-k N] [--no-render]
vizjudge profile DATASET
vizjudge agent DATASET [--target COLUMN] [--task TASK] [--output DIR]
               [--judge-file FILE | --judge-url URL] [--max-iterations N]
               [--max-tool-calls N] [--max-seconds S] [--python-timeout S]
               [--max-rows N] [--max-candidates N] [--max-rendered-charts N]
vizjudge skill install [--target agents|dsh|claude] [--project] [--dest DIR] [--force]
vizjudge skill path
vizjudge mcp-config [--client dsh|json|claude]
vizjudge doctor
vizjudge --version
```

## 项目状态

这是一个 alpha 质量、本地优先的 v0.5 实现。Agent 循环、预测性检验、基于证据的 Judge 集成和执行预算都已实现。
在线 Judge 的质量和对下游模型效果的影响仍需基准测试；图像/OCR 输入和操作系统级沙箱属于后续工作。
详见 [路线图](docs/roadmap.md) 和 [报告格式迁移指南](docs/report-schema.md)。

## 开发

依赖版本锁定在 [`uv.lock`](uv.lock) 中；CI 会在 Linux 和 Windows 上用 Python 3.11 与 3.12 运行代码检查和测试，然后构建 wheel 并做冒烟测试。

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
```

## 许可证

[MIT](LICENSE)

> 本文档为英文 [README](README.md) 的中文翻译，如有出入以英文版为准。
