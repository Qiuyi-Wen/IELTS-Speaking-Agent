# 🎓 交互式雅思回答陪练与智能评估系统

**Interactive IELTS Answer Practice & Text Assessment System**

基于 **LangGraph + LangChain** 构建的多节点协作系统，为用户提供雅思口语 Part 2 题目练习、语法/词汇分析、文本连贯性评估和分阶段学习建议。

> 当前版本接收键盘输入，因此只评估回答文本。发音与真实口语流利度需要音频证据，本项目不会从文本或打字耗时推断这两个维度，也不会把文本综合分冒充完整 IELTS Speaking 总分。

---

## 📋 目录

- [系统概述](#系统概述)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [自动评测](#自动评测)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [Agent 工作流](#agent-工作流)
- [记忆机制](#记忆机制)
- [工具说明](#工具说明)
- [题库说明](#题库说明)
- [常见问题](#常见问题)

---

## 系统概述

本系统覆盖雅思口语 Part 2 回答文本的练习与评估流程：

1. **Examiner**（考官）从内置题库随机抽取一道 Cue Card 题目
2. 用户在计时器监督下输入英文口语回答
3. **Grammar_Judge**（语法考官）逐条分析语法错误并给出维度评分
4. **Vocab_Judge**（词汇考官）评估词汇使用并提供高级替换词推荐
5. **Head_Coach**（主教练）校准结构化结果，输出文本综合分与分阶段复习建议
6. 练习数据自动保存至本地 JSON 文件，形成长期学习档案

---

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                    LangGraph StateGraph             │
│                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Examiner │──▶│ Human    │──▶│  Timer   │       │
│  │ (出题)   │    │ Input    │    │ (计时器) │        │
│  └──────────┘   │ (interrupt)│   └──────────┘       │
│                   └──────────┘        │             │
│                                       ▼             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  Output  │◀──│  Head    │◀──│  Vocab   │       │
│  │ (报告+保  │    │  Coach  │    │  Judge   │       │
│  │  持久化)  │    │ (校准+   │    │ (词汇    │       │
│  └──────────┘    │  综合评分)│    │  分析)   │       │
│                  └──────────┘    └──────────┘       │
│                        ▲                            │
│                        │                            │
│                   ┌──────────┐                      │
│                   │ Grammar  │                      │
│                   │  Judge   │                      │
│                   │ (语法    │                      │
│                   │  分析)   │                      │
│                   └──────────┘                      │
└─────────────────────────────────────────────────────┘
```

| Agent | 角色 | 核心职责 |
|-------|------|----------|
| 🎯 **Examiner** | 考官 | 从 10 道题库中随机抽题，跨轮次去重，启动计时器 |
| ⌨️ **Human Input** | 人机交互 | LangGraph `interrupt()` 中断挂起，等待用户输入 |
| ⏱️ **Timer** | 计时器 | 记录包含阅读与输入在内的交互耗时，不用于推断口语流利度 |
| 🔍 **Grammar_Judge** | 语法考官 | 结构化语法分析；程序校验每条原文引用 |
| 📚 **Vocab_Judge** | 词汇考官 | 结构化词汇分析；不强制制造替换建议 |
| 🎓 **Head_Coach** | 主教练 | 校准专项评分 → 文本综合分 + 可展示依据 + 分阶段建议 |
| 📄 **Output** | 输出 | 分层打印评估报告 + 长期记忆 JSON 持久化 |

---

## 核心特性

### ✅ 结构化输出与证据校验

三个模型节点均通过 Pydantic schema 返回结构化结果：

1. 所有分数限制在 0-9，并使用 0.5 分档
2. `quote` / `source_quote` 必须是用户原文的精确子串
3. 引文校验失败时自动重试一次，连续失败则终止本轮评估
4. 没有可靠错误时允许返回空 `issues`，避免强迫模型制造问题
5. 长期画像直接从显式 `issues` 更新，不再扫描自然语言关键词
6. 专项分数由子维度算术校准，避免模型返回互相矛盾的总分
7. 主教练的依据与建议必须关联实际 `issue.category`，不允许凭空制造弱项

### 💾 双重记忆机制

| 记忆类型 | 实现方式 | 生命周期 | 作用 |
|----------|----------|----------|------|
| **短期记忆** | `AgentState.messages` + `operator.add` reducer | 单轮内 | 各节点之间传递中间产出，自动累加消息 |
| **长期记忆** | `Qiuyi_ielts_profile.json` 本地文件 | 跨轮次持久化 | 记录每次练习的题目、词数、耗时、薄弱项、分数及全部历史统计 |

### 🛠️ 双工具系统

| 工具 | 类型 | 功能 |
|------|------|------|
| `timer_tool` | 计时工具 | 记录从考官出题到用户回答完毕的真实耗时 |
| `update_profile_tool` | 文件写入工具 | 将评估结果结构化写入 JSON 长记忆文件，自动计算历史均分与累计薄弱项 |

### 🎲 动态题库

内置 10 道雅思 Part 2 标准题目，覆盖人物、地点、事件、物品、活动 5 大类别。采用 `list.remove()` 真正剔除已用题目，确保多轮不重复；题库耗尽后自动重置。

### 👤 Human-in-the-Loop

基于 LangGraph 的 `interrupt()` + `Command(resume=...)` 实现，图在出题后自动挂起等待用户输入，提交后自动恢复后续分析流程。

---

## 环境要求

- **Python**：3.10+
- **API 密钥**：阿里云百炼 DashScope API Key（需开通 qwen-turbo 模型服务）
- **操作系统**：Windows / macOS / Linux

**Python 依赖包**：

| 包名 | 用途 |
|------|------|
| `langchain-openai` | ChatOpenAI 接口（兼容 DashScope） |
| `langgraph` | StateGraph 图编排 + MemorySaver + interrupt |
| `pydantic` | 评估 schema、分数范围与数据校验 |

---

## 快速开始

### 1. 克隆或下载项目

```bash
cd /path/to/
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API 密钥

**Windows (PowerShell)**：

```powershell
$env:DASHSCOPE_API_KEY = '你的DashScope API密钥'
```

**macOS / Linux (Bash)**：

```bash
export DASHSCOPE_API_KEY='你的DashScope API密钥'
```

> 💡 API 密钥获取地址：https://dashscope.console.aliyun.com/billing
>
> 项目不包含默认密钥；请仅通过环境变量提供密钥。

### 4. 运行系统

```bash
python AILab_AgentIELTSTestPreparation.py
```

### 5. 运行本地校验

```bash
python -m unittest discover -s tests -v
```

这些测试不调用模型 API，覆盖分数校准、原文引用、薄弱项来源、建议证据关联和评测集契约。

---

## 自动评测

`evals/cases.json` 提供 12 条固定回答，低、中、高三个质量等级各 4 条。每条记录包含预期文本分数区间；低质量回答还标注专项考官必须召回的问题类别。

真实模型评测是显式运行的，默认只执行 3 条，避免意外产生大量 API 费用：

```powershell
$env:DASHSCOPE_API_KEY = "你的DashScope API密钥"
python -m evals.run_model_evals --limit 3
```

需要保留逐条结果时，可指定输出文件：

```powershell
python -m evals.run_model_evals --limit 3 --output evals/results.json
```

评测器会在每条案例完成后立即打印并更新输出文件。单条模型输出校验失败会被记录为失败结果，后续案例仍会继续运行。

也可以只运行指定样例：

```bash
python -m evals.run_model_evals --case-id low_place_01 --case-id high_place_01
```

GitHub Actions 只运行免费的确定性单元测试，不会读取 API Key 或调用模型。

---

## 使用说明

### 单轮练习流程

1. 系统启动后自动打印架构信息和 Agent 角色说明
2. Examiner 从题库中随机抽取一道 Part 2 题目并展示
3. 系统进入等待状态，显示输入提示框
4. 用户输入英文口语回答（建议 150-250 词，1-2 分钟），按回车提交
5. 系统自动完成：计时 → 结构化语法分析 → 结构化词汇分析 → 结果校准 → 报告输出
6. 练习数据自动保存至 `Qiuyi_ielts_profile.json`

### 多轮练习

每轮结束后系统会询问是否继续。继续则自动进入下一轮，Examiner 会抽取一道与之前不同的题目。

### 交互命令

| 输入 | 效果 |
|------|------|
| 正常英文回答 | 提交并进入分析流程 |
| `quit` / `exit` / `退出` | 退出当前轮次或程序 |
| `y` / `yes` | 确认继续下一轮 / 重置题库 |
| `n` / 其他任意键 | 结束练习 |

### 输出说明

每轮练习结束后，终端会显示分层评估报告：

```
🎯 本轮雅思回答文本评估报告
├── ⏱️ 交互数据（耗时 + 词数，不作为口语流利度证据）
├── 🔎 可展示、可核查的评分依据
├── 📊 回答文本综合分
├── 📝 完整结构化语法诊断
├── 📚 完整结构化词汇诊断
├── 📅 下一步复习建议（🔴短期 / 🟡中期 / 🟢长期）
├── 💪 鼓励语
└── 💾 长期记忆状态（文件路径 + 累计统计）
```

---

## 项目结构

```
AgentLab/
├── AILab_AgentIELTSTestPreparation.py   # 主程序（雅思陪练系统）
├── evaluation_models.py                  # Pydantic 评估 schema 与证据校验
├── evals/                                # 12 条固定样例 + 按需模型评测器
├── tests/                                # 无需调用真实模型的单元测试
├── .github/workflows/tests.yml           # PR / main 自动测试
├── requirements.txt                      # Python 依赖
├── Qiuyi_ielts_profile.json              # 长期记忆文件（运行后自动生成）
├── Agent_Architecture.png                # LangGraph 架构图
├── README.md                             # 本文件
└── .gitignore
```

---

## Agent 工作流

### 节点流转图

```
Examiner → Human_Input → Timer → Grammar_Judge → Vocab_Judge → Head_Coach → Output → END
    │           │                                                      │
    │     [interrupt()]                                          [update_profile_tool]
    │     ← 图挂起 ←                                              ← JSON 文件写入
    │     Command(resume=answer)
    │           │
    └───────────┘
```

### 各节点职责

| 节点 | 函数 | 触发时机 | 输入 | 输出 |
|------|------|----------|------|------|
| `examiner` | `examiner_node` | 图启动 / 新一轮 | — | 随机题目 + 计时启动 |
| `human_input` | `human_input_node` | examiner 完成后 | interrupt 挂起 | 用户回答文本 |
| `timer` | `timer_node` | 用户回答提交后 | start_time | 真实耗时 |
| `grammar_judge` | `grammar_judge_node` | timer 完成后 | 题目 + 回答 | 语法分析报告 |
| `vocab_judge` | `vocab_judge_node` | grammar 完成后 | 题目 + 回答 | 词汇分析报告 |
| `head_coach` | `head_coach_node` | vocab 完成后 | 结构化反馈 + 历史记录 | 校准后的文本综合评估 |
| `output` | `output_node` | coach 完成后 | 全部状态 | 分层报告 + JSON 写入 |

---

## 记忆机制

### 短期记忆：AgentState

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 自动累加，各节点消息持续累积
    current_question: str                     # 本轮题目
    user_answer: str                          # 用户回答原文
    grammar_feedback: str                     # 语法考官反馈
    vocab_feedback: str                       # 词汇考官反馈
    head_coach_feedback: str                  # 主教练综合评估
    grammar_result: dict                       # 通过 schema 校验的语法结果
    vocab_result: dict                         # 通过 schema 校验的词汇结果
    coach_result: dict                         # 通过 schema 校验的综合结果
    elapsed_time: float                       # 答题耗时（秒）
    estimated_score: float                    # 回答文本综合分
    profile_path: str                         # 长期记忆路径；自动评测时可隔离
    phase: str                                # 当前阶段标识
    available_questions: list                 # 剩余题库索引（跨轮次传递）
```

### 长期记忆：Qiuyi_ielts_profile.json

```json
{
  "student_name": "Qiuyi",
  "created_at": "2026-07-19T00:00:00",
  "practice_sessions": [
    {
      "session_id": 1,
      "timestamp": "2026-07-19T12:00:00",
      "question": "Describe a person who has influenced your life greatly.",
      "answer_word_count": 180,
      "elapsed_seconds": 95.3,
      "weaknesses": ["时态", "介词"],
      "estimated_score": 6.0
    }
  ],
  "total_sessions": 1,
  "average_score": 6.0,
  "cumulative_weaknesses": ["时态", "介词"],
  "last_updated": "2026-07-19T12:00:00"
}
```

---

## 工具说明

### timer_tool

```python
timer_tool(action="start")                             # 开始计时
timer_tool(action="stop", start_time=t1, end_time=t2)  # 结束计时，输出耗时
timer_tool(action="elapsed", start_time=t1, end_time=t2)  # 查看当前已用时间
```

- `start` 操作记录时间节点，返回人类可读的时间戳
- `stop` 操作计算精确壁钟时间差（秒），超过 60 秒时以"X分Y秒"格式输出
- 完全本地执行，不依赖任何外部 API

### update_profile_tool

```python
update_profile_tool(
    profile_json='{"question":"...", "answer_word_count": 180, ...}',
    profile_path="Qiuyi_ielts_profile.json"
)
```

核心逻辑：
1. 尝试读取已有 JSON 文件（容错：文件缺失或损坏时自动创建初始结构）
2. 追加新会话记录并分配 `session_id`
3. 自动重算 `average_score`（排除 0 分记录）、去重汇总 `cumulative_weaknesses`
4. 写入文件，返回操作结果摘要

---

## 题库说明

题库包含 10 道精心设计的雅思 Part 2 Cue Card 题目，每道题包含标准格式的 3-4 个提示点（bullet points）：

| 类别 | 题数 | 示例题目 |
|------|------|----------|
| 🧑 人物类 (People) | 1 | Describe a person who has influenced your life greatly |
| 🏙️ 地点类 (Places) | 2 | Describe a place you have visited that you would like to go back to |
| 📅 事件/经历类 (Events) | 3 | Describe an unforgettable photography experience |
| 📦 物品类 (Objects) | 2 | Describe an important piece of technology you use every day |
| 🎯 活动/习惯类 (Activities) | 2 | Describe a hobby you have that is unusual or interesting |

---

## 常见问题

### Q: 运行时提示 `DASHSCOPE_API_KEY` 未设置？

**A**: 按照[快速开始](#快速开始)第 3 步设置环境变量。不要把 API Key 写入代码或提交到 Git；同时确保在阿里云百炼平台已开通 qwen-turbo 模型服务。

### Q: 程序在等待输入时卡住了？

**A**: 这是正常现象——系统通过 LangGraph 的 `interrupt()` 机制挂起图执行，等待你的口语回答。输入英文回答后按回车即可继续。

### Q: 长期记忆文件在哪里？

**A**: 运行一次后，`Qiuyi_ielts_profile.json` 会自动生成在当前目录下。它是一个结构化的 JSON 文件，可用任何文本编辑器打开查看。

### Q: 可以更换模型吗？

**A**: 可以。修改代码中的 `model` 配置即可——系统使用 OpenAI 兼容接口，支持任何兼容 `ChatOpenAI` 的模型服务端点。例如：

```python
model = ChatOpenAI(
    model="qwen-plus",   # 换成更强模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
    temperature=0.7,
)
```

---

**🤖 技术栈**：LangGraph · LangChain · qwen-turbo · DashScope API · Python

**📧 作者**：Qiuyi Wen
