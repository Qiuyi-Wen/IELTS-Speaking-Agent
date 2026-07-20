# 🎓 交互式雅思口语模拟陪练与智能评估系统

**Interactive IELTS Speaking Practice & Assessment System**

基于 **LangGraph + LangChain** 构建的多智能体协作系统，通过 4 个分工明确的 AI Agent 为用户提供从出题、多维分析到综合评分的全流程雅思口语 Part 2 模拟训练。

---

## 📋 目录

- [系统概述](#系统概述)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [使用说明](#使用说明)
- [项目结构](#项目结构)
- [Agent 工作流](#agent-工作流)
- [记忆机制](#记忆机制)
- [工具说明](#工具说明)
- [题库说明](#题库说明)
- [常见问题](#常见问题)

---

## 系统概述

本系统模拟了真实雅思口语 Part 2 的考试与评估全流程：

1. **Examiner**（考官）从内置题库随机抽取一道 Cue Card 题目
2. 用户在计时器监督下输入英文口语回答
3. **Grammar_Judge**（语法考官）逐条分析语法错误并给出维度评分
4. **Vocab_Judge**（词汇考官）评估词汇使用并提供高级替换词推荐
5. **Head_Coach**（主教练）通过 **Reflection 反思机制**综合评估，输出总分与分阶段复习建议
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
│  │  持久化)  │    │ (反思+   │    │ (词汇    │       │
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
| ⏱️ **Timer** | 计时器 | 记录真实壁钟耗时，为流利度推断提供数据支撑 |
| 🔍 **Grammar_Judge** | 语法考官 | 8 维度语法分析，逐条标注"原文→建议"，含引文保真机制 |
| 📚 **Vocab_Judge** | 词汇考官 | 7 维度词汇分析，提供替换建议表 + 高级词汇推荐 |
| 🎓 **Head_Coach** | 主教练 | **Reflection 反思** → 5 项自检 → 四维评分 + 分阶段复习建议 |
| 📄 **Output** | 输出 | 分层打印评估报告 + 长期记忆 JSON 持久化 |

---

## 核心特性

### 🧠 Reflection 反思机制

Head_Coach 在输出评分前，必须在 `<thought>...</thought>` 标签内完成 5 项深度反思：

1. 考官反馈一致性检查
2. 耗时合理性分析（理想：1-2 分钟 / 150-250 词）
3. 与长期记忆中的历史表现对比分析
4. 各维度评分权重动态调整
5. 预估总分与分项评分的一致性校准

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

- **Python**：3.9+
- **API 密钥**：阿里云百炼 DashScope API Key（需开通 qwen-turbo 模型服务）
- **操作系统**：Windows / macOS / Linux

**Python 依赖包**：

| 包名 | 用途 |
|------|------|
| `langchain` | LCEL 基础框架 |
| `langchain-openai` | ChatOpenAI 接口（兼容 DashScope） |
| `langraph` | StateGraph 图编排 + MemorySaver + interrupt |

---

## 快速开始

### 1. 克隆或下载项目

```bash
cd /path/to/
```

### 2. 安装依赖

```bash
pip install langchain langchain-openai langraph
```

### 3. 配置 API 密钥

**Windows (PowerShell)**：

```powershell
$env:OPENAI_API_KEY = '你的DashScope API密钥'
```

**macOS / Linux (Bash)**：

```bash
export OPENAI_API_KEY='你的DashScope API密钥'
```

> 💡 API 密钥获取地址：https://dashscope.console.aliyun.com/billing
>
> 如果不设置环境变量，代码将使用内置的默认密钥（可能已过期）。

### 4. 运行系统

```bash
python AILab_AgentIELTSTestPreparation.py
```

---

## 使用说明

### 单轮练习流程

1. 系统启动后自动打印架构信息和 Agent 角色说明
2. Examiner 从题库中随机抽取一道 Part 2 题目并展示
3. 系统进入等待状态，显示输入提示框
4. 用户输入英文口语回答（建议 150-250 词，1-2 分钟），按回车提交
5. 系统自动完成：计时 → 语法分析 → 词汇分析 → 反思 → 综合评分 → 报告输出
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
🎯 本轮雅思口语模拟评估报告
├── ⏱️ 答题数据（耗时 + 词数）
├── 🧠 主教练深度反思（<thought> 块内容）
├── 📊 预估总分（含水平等级标签）
├── 📝 语法维度诊断（精简版）
├── 📚 词汇维度诊断（精简版）
├── 📅 下一步复习建议（🔴短期 / 🟡中期 / 🟢长期）
├── 💪 鼓励语
└── 💾 长期记忆状态（文件路径 + 累计统计）
```

---

## 项目结构

```
AgentLab/
├── AILab_AgentIELTSTestPreparation.py   # 主程序（雅思陪练系统）
├── Qiuyi_ielts_profile.json              # 长期记忆文件（运行后自动生成）
├── README.md                             # 本文件
└── Report.md                             # 实验报告
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
| `head_coach` | `head_coach_node` | vocab 完成后 | 全部反馈 + 历史记录 | 反思 + 综合评分 |
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
    elapsed_time: float                       # 答题耗时（秒）
    estimated_score: float                    # 预估总分
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

### Q: 运行时提示 `OPENAI_API_KEY` 未设置？

**A**: 按照[快速开始](#快速开始)第 3 步设置环境变量，或在代码中直接填入 API Key。确保在阿里云百炼平台已开通 qwen-turbo 模型服务。

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

**📧 作者**：文秋懿 
