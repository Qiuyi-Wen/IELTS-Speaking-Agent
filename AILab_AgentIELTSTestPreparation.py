"""
================================================================================
AILab_AgentIELTSTestPreparation.py  (v2 — BUGFIX + UX 优化)
交互式雅思口语模拟陪练与评估系统
Interactive IELTS Speaking Practice & Assessment System
================================================================================

【评分标准对照说明】

标准1 — 4 个分工明确的 Agent：
  Examiner → Grammar_Judge → Vocab_Judge → Head_Coach（各含独立 System Prompt）

标准2 — Reflection 反思机制：
  Head_Coach 输出评分前必须通过 <thought> 标签完成 5 项内部反思

标准3 — 双重记忆：
  短期：AgentState.messages (operator.add reducer) | 长期：Qiuyi_ielts_profile.json

标准4 — 2 种工具：
  @tool timer_tool | @tool update_profile_tool

标准5 — Human-in-the-loop：
  LangGraph interrupt() → 外部 input() → Command(resume=...) 恢复

【v2 修复清单】
  BUGFIX-1: 将输入提示 UI 从 human_input_node 移出到 run_practice_round()
             消除 LangGraph resume 时的重复打印
  BUGFIX-2: Grammar_Judge 新增"⚠️ 引文保真铁律"→ 强制 LLM 逐字复制原文
  UX-1:     output_node 重构为分层报告模板（答题数据 → 反思 → 分数 → 诊断 → 建议）

【v3 更新】
  EXAMINER: 废弃 LLM 生成题目 → 改用内置题库 (IELTS_PART2_QUESTION_BANK)
            随机抽取 + 去重，确保多轮题目不重复
            题库包含 10 道经典题目，涵盖人物/物品/事件/地点/活动 5 大类别

【v3.1 BUGFIX — 题库计数器跨轮次修复】
  根因：available_questions 被放在 run_practice_round() 内部，
        每轮调用时都重新初始化为 []，导致计数器卡死且无法真正去重。
  修复：将该列表提升到 main() 的 while 循环外部，
        run_practice_round() 通过参数接收并通过返回值传回更新后的列表，
        examiner_node 使用 list.remove(chosen_idx) 真正剔除已用题目。
  新增：题库耗尽时优雅提示，允许用户选择重置或退出。
================================================================================
"""

import os
import sys
import json
import time
import re
import random
from typing import TypedDict, Annotated, Optional, Any
from datetime import datetime
import operator

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.tools import tool


# ============================================================================
# 第1部分：模型配置
# ============================================================================

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    #print("⚠️  警告：未设置 OPENAI_API_KEY 环境变量！")
    #print("   请执行：$env:OPENAI_API_KEY='你的通义千问API密钥'  (PowerShell)")
    #print("   或：  export OPENAI_API_KEY='你的通义千问API密钥'   (Bash)")
    API_KEY = "your_api_key_here"

model = ChatOpenAI(
    model="qwen-turbo",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=API_KEY,
    temperature=0.7,
)

print(f"✅ 模型初始化完成：qwen-turbo @ dashscope.aliyuncs.com")


# ============================================================================
# 第1.5部分：雅思口语 Part 2 内置题库
# 【v3 新增】每次 Examiner 触发时随机抽取，去重避免多轮重复
# 涵盖人物、地点、事件、物品、活动等标准类别
# ============================================================================

IELTS_PART2_QUESTION_BANK = [
    # --- 人物类 (People) ---
    {
        "topic": "Describe a person who has influenced your life greatly.",
        "prompts": [
            "who this person is",
            "how you know this person",
            "what influence this person has had on you",
            "and explain why this person has influenced you so greatly"
        ],
        "category": "人物类 (People)"
    },
    # --- 物品/乐器类 (Objects) ---
    {
        "topic": "Describe a musical instrument you enjoy playing.",
        "prompts": [
            "what the instrument is (e.g. acoustic guitar)",
            "when and why you started learning it",
            "how you feel when you play it",
            "and explain what kind of music you enjoy playing on it"
        ],
        "category": "物品/活动类 (Objects & Activities)"
    },
    # --- 事件/经历类 (Events) ---
    {
        "topic": "Describe an unforgettable photography experience or the photo you are most proud of.",
        "prompts": [
            "when and where you took this photo",
            "what the photo shows",
            "why this photo is special to you",
            "and explain what this experience taught you about photography"
        ],
        "category": "事件/经历类 (Events)"
    },
    # --- 物品/娱乐类 (Objects) ---
    {
        "topic": "Describe your favorite video game (cooperative or story-driven).",
        "prompts": [
            "what the game is called",
            "when you first played it",
            "what the gameplay or story is about",
            "and explain why you enjoy it so much, especially the cooperative or narrative elements"
        ],
        "category": "物品/娱乐类 (Objects)"
    },
    # --- 事件/经历类 (Events) ---
    {
        "topic": "Describe a time when you helped someone learn something (e.g. tutoring in math).",
        "prompts": [
            "who you helped and what you helped them with",
            "when and where this took place",
            "how you helped them learn",
            "and explain how you felt about this experience and what you learned from it"
        ],
        "category": "事件/经历类 (Events)"
    },
    # --- 地点类 (Places) ---
    {
        "topic": "Describe a place you have visited that you would like to go back to.",
        "prompts": [
            "where this place is",
            "when you went there",
            "what you did there",
            "and explain why you would like to return to this place"
        ],
        "category": "地点类 (Places)"
    },
    # --- 物品类 (Objects) ---
    {
        "topic": "Describe an important piece of technology you use every day.",
        "prompts": [
            "what the technology is",
            "how you use it",
            "how it has changed your daily life",
            "and explain why it is so important to you"
        ],
        "category": "物品类 (Objects)"
    },
    # --- 事件/经历类 (Events) ---
    {
        "topic": "Describe a time when you had to wait for something important.",
        "prompts": [
            "what you were waiting for",
            "when and where this happened",
            "how you felt while waiting",
            "and explain what happened when the waiting was finally over"
        ],
        "category": "事件/经历类 (Events)"
    },
    # --- 活动/习惯类 (Activities) ---
    {
        "topic": "Describe a hobby you have that is unusual or interesting.",
        "prompts": [
            "what the hobby is",
            "how you got started with it",
            "how often you do it",
            "and explain why you think this hobby is unusual or interesting"
        ],
        "category": "活动/习惯类 (Activities)"
    },
    # --- 地点类 (Places) ---
    {
        "topic": "Describe a quiet place where you like to go to think or relax.",
        "prompts": [
            "where this place is",
            "how you discovered it",
            "what you do there",
            "and explain why this place helps you relax or think clearly"
        ],
        "category": "地点类 (Places)"
    },
]

print(f"✅ 题库加载完成：共 {len(IELTS_PART2_QUESTION_BANK)} 道 Part 2 题目")


# ============================================================================
# 第2部分：工具定义（评分标准4 — 2种工具）
# ============================================================================

@tool
def timer_tool(action: str, start_time: float = 0.0, end_time: float = 0.0) -> str:
    """计时器工具：记录从考官提问到用户回答完毕的真实耗时。"""
    if action == "start":
        return f"计时开始：{datetime.now().strftime('%H:%M:%S')}"
    elif action == "stop":
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            return f"计时结束。答题耗时：{minutes}分{seconds}秒（{elapsed:.1f}秒）"
        else:
            return f"计时结束。答题耗时：{seconds}秒（{elapsed:.1f}秒）"
    elif action == "elapsed":
        return f"当前耗时：{end_time - start_time:.1f}秒"
    else:
        return f"未知操作：{action}"


@tool
def update_profile_tool(profile_json: str, profile_path: str = "Qiuyi_ielts_profile.json") -> str:
    """文件写入工具：将主教练最新评价总结写入长期记忆 JSON 文件。"""
    try:
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing = {
                "student_name": "Qiuyi",
                "created_at": datetime.now().isoformat(),
                "practice_sessions": [],
                "total_sessions": 0,
                "average_score": 0.0,
                "cumulative_weaknesses": [],
                "last_updated": "",
            }

        new_session = json.loads(profile_json)
        new_session["session_id"] = len(existing["practice_sessions"]) + 1
        new_session["timestamp"] = datetime.now().isoformat()

        existing["practice_sessions"].append(new_session)
        existing["total_sessions"] = len(existing["practice_sessions"])
        existing["last_updated"] = datetime.now().isoformat()

        all_scores = [s.get("estimated_score", 0) for s in existing["practice_sessions"]]
        all_scores = [s for s in all_scores if s > 0]
        if all_scores:
            existing["average_score"] = round(sum(all_scores) / len(all_scores), 1)

        all_weaknesses = []
        for s in existing["practice_sessions"]:
            for w in s.get("weaknesses", []):
                if w not in all_weaknesses:
                    all_weaknesses.append(w)
        existing["cumulative_weaknesses"] = all_weaknesses

        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)

        total = existing["total_sessions"]
        avg = existing["average_score"]
        return (
            f"✅ 长期记忆更新成功！\n"
            f"   - 文件：{profile_path}\n"
            f"   - 累计练习次数：{total}\n"
            f"   - 历史平均分：{avg}\n"
            f"   - 累计薄弱项：{', '.join(all_weaknesses) if all_weaknesses else '暂无'}"
        )
    except json.JSONDecodeError as e:
        return f"❌ JSON解析错误：{str(e)}"
    except Exception as e:
        return f"❌ 更新长期记忆失败：{str(e)}"


# ============================================================================
# 第3部分：长期记忆读取（评分标准3）
# ============================================================================

def load_long_term_memory(profile_path: str = "Qiuyi_ielts_profile.json") -> dict:
    """读取长期记忆文件，获取学生全部历史练习记录。"""
    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        session_count = len(profile.get("practice_sessions", []))
        avg = profile.get("average_score", 0)
        weaknesses = profile.get("cumulative_weaknesses", [])
        print(f"\n📖 [长期记忆] 已加载：{session_count}次练习 | 平均分{avg} | 薄弱项：{weaknesses}")
        return profile
    except (FileNotFoundError, json.JSONDecodeError):
        print("\n📖 [长期记忆] 未找到历史记录，这将是第一次练习。")
        return {
            "student_name": "Qiuyi",
            "created_at": datetime.now().isoformat(),
            "practice_sessions": [],
            "total_sessions": 0,
            "average_score": 0.0,
            "cumulative_weaknesses": [],
        }


# ============================================================================
# 第4部分：状态定义（评分标准3 — 短期记忆）
# ============================================================================

class AgentState(TypedDict):
    """LangGraph AgentState — messages 用 operator.add 做 reducer，自动累加。"""
    messages: Annotated[list, operator.add]
    current_question: str
    user_answer: str
    grammar_feedback: str
    vocab_feedback: str
    head_coach_feedback: str
    elapsed_time: float
    timer_result: str
    estimated_score: float
    phase: str
    question_start_time: float
    available_questions: list   # 【v3.1】跨轮次的剩余题库索引池，examiner用remove()真剔除


# ============================================================================
# 第5部分：辅助函数（报告排版用）
# ============================================================================

def extract_concise_feedback(full_text: str, max_chars: int = 450) -> str:
    """
    从考官的长篇反馈中提取精简版摘要。
    优先保留评分行、❌/✅ 标记行、表格数据（最多5行）。
    如果提取内容过短，退回使用前 max_chars 字符。
    """
    if not full_text:
        return "(无反馈)"

    lines = full_text.split("\n")
    result_lines = []
    table_rows = 0

    for line in lines:
        stripped = line.strip()
        # 评分行：总是保留
        if re.search(r'(综合预估分|预估.*分|X\.\d/9)', stripped):
            result_lines.append(line)
        # 错误/亮点标记行
        elif re.match(r'^\d+\.\s*(❌|✅|\*\*)', stripped):
            result_lines.append(line)
        # 错误类型标记：1. [时态] — 原文：... → 建议：...
        elif re.match(r'^\d+\.\s*\[.*?\]', stripped):
            result_lines.append(line)
        # 表格标题和分隔行
        elif '| 序号 |' in stripped or '|------|' in stripped:
            result_lines.append(line)
        # 表格数据行（最多 5 行）
        elif stripped.startswith('|') and '|' in stripped[1:]:
            if table_rows < 5:
                result_lines.append(line)
                table_rows += 1

    result = "\n".join(result_lines)

    if len(result) < 100:
        result = full_text[:max_chars]
        if len(full_text) > max_chars:
            result += "\n  ... (完整诊断见各考官原始报告)"

    return result


def extract_thought_block(text: str) -> str:
    """从 Head_Coach 反馈中提取 <thought>...</thought> 块内容。"""
    m = re.search(r'<thought>\s*(.*?)\s*</thought>', text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_review_section(text: str) -> str:
    """
    从 Head_Coach 反馈中提取复习建议部分。
    从"分阶段复习建议"或第一个 🔴 标记开始，到鼓励语之前结束。
    """
    # 尝试匹配 "分阶段复习建议" 之后的全部内容
    m = re.search(r'(?:###?\s*📋\s*分阶段复习建议[：:]*)(.*)', text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        # 截断到鼓励语
        stop = re.search(r'###?\s*💪', content)
        if stop:
            content = content[:stop.start()].strip()
        return content

    # 备选：从 🔴 开始匹配
    m = re.search(r'(\*\*🔴.*)', text, re.DOTALL)
    if m:
        content = m.group(1).strip()
        stop = re.search(r'###?\s*💪', content)
        if stop:
            content = content[:stop.start()].strip()
        return content

    return ""


def extract_encouragement(text: str) -> str:
    """提取鼓励语。"""
    m = re.search(r'(?:###?\s*💪\s*鼓励语[：:]?\s*)(.*?)$', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


# ============================================================================
# 第6部分：各 Agent 的 System Prompt 和节点函数
# ============================================================================

# ---------------------------------------------------------------------------
# Agent 1: Examiner（考官）—— 出题
# ---------------------------------------------------------------------------

EXAMINER_SYSTEM_PROMPT = """你是一位经验丰富的雅思口语考官，持有剑桥大学认证的雅思考官资格。

## 你的任务
提出一个地道的雅思口语 Part 2 题目（Cue Card 形式）。

## 出题要求
1. 题目应模仿真实雅思考试的难度和风格
2. 每个题目必须包含 3-4 个提示点（bullet points），引导学生展开回答
3. 话题应覆盖以下类别，建议每次轮换：
   - 人物类 (People)：描述一位名人/家人/朋友
   - 地点类 (Places)：描述一个你去过的地方
   - 事件/经历类 (Events)：描述一件难忘的事
   - 物品类 (Objects)：描述一件你拥有的重要物品
   - 习惯/活动类 (Activities)：描述你的一个爱好

## 输出格式
请只输出题目本身（英文），不要输出任何其他文字。格式如下：

---
Describe [topic]

You should say:
- [point 1]
- [point 2]
- [point 3]
- And explain [point 4]
---

现在请出题。"""


def examiner_node(state: AgentState) -> dict:
    """
    Agent 1: Examiner — 从内置题库随机抽取雅思口语 Part 2 题目。

    【v3.1 修复】题库状态管理改为"剩余池 + 真正剔除"：
    1. state["available_questions"] 是一个跨轮次维护的"剩余索引列表"
       （该列表由 main() 在 while 循环外部创建，每轮通过 initial_state 传入）
    2. 使用 list.remove(chosen_idx) 将抽到的题目从剩余池中**真正移除**
    3. 池空了则自动重置（全部题目轮完一遍后刷新）
    4. 终端打印的计数 =（题库总量 - 剩余数量）/ 题库总量
    """
    print("\n" + "=" * 65)
    print("  🎯 Agent 1: Examiner（考官）— 正在准备雅思口语 Part 2 题目")
    print("=" * 65)

    total = len(IELTS_PART2_QUESTION_BANK)

    # ---- 获取跨轮次的剩余题库索引池 ----
    available = state.get("available_questions", [])
    if available is None:
        available = []

    # 如果剩余池为空（首次运行或全部轮完），初始化/重置
    if not available:
        available = list(range(total))
        if state.get("available_questions") is not None and len(state.get("available_questions", [])) == 0:
            print(f"  🔄 题库已用尽（{total}题全部练习完毕），自动重置题库。")

    # ---- 从剩余池中随机抽取并真正剔除 ----
    chosen_idx = random.choice(available)
    available.remove(chosen_idx)   # 【关键】真正从列表中移除，而非仅记录

    card = IELTS_PART2_QUESTION_BANK[chosen_idx]

    # ---- 格式化输出题目 ----
    question_lines = [card["topic"]]
    question_lines.append("\nYou should say:")
    for prompt in card["prompts"]:
        question_lines.append(f"- {prompt}")
    question = "\n".join(question_lines)

    start_time = time.time()
    used_count = total - len(available)
    remaining = len(available)

    print(f"\n📋 考官出题（第{used_count}题 / 题库共{total}题，剩余{remaining}题）：")
    print(f"   🏷️  类别：{card['category']}")
    print(f"\n{question}")

    return {
        "current_question": question,
        "question_start_time": start_time,
        "available_questions": available,   # 返回更新后的剩余池
        "phase": "question_asked",
        "messages": [{
            "role": "examiner",
            "content": question,
            "timestamp": datetime.now().isoformat()
        }],
    }


# ---------------------------------------------------------------------------
# Human-in-the-Loop 中断节点（评分标准5）
# 【BUGFIX-1】节点内部不打印输入 UI —— 所有 UI 提示统一在 run_practice_round() 外层打印
# 原因：LangGraph resume 时会重新进入该节点函数，若在此处打印会导致重复 UI
# ---------------------------------------------------------------------------

def human_input_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop 中断节点。

    【BUGFIX-1】
    本节点仅调用 interrupt() + 确认回执。
    不打印任何交互 UI（提示框在外层 run_practice_round() 中打印一次）。
    这确保了 UI 在整个流程中只出现一次，无论 graph 如何 resume。
    """
    # interrupt() 挂起，await 外部 Command(resume=<answer>) 恢复
    user_answer = interrupt("请在此输入你的雅思口语回答（英文）：")

    # --- 以下仅在 resume 恢复后执行一次 ---
    word_count = len(user_answer.split()) if user_answer else 0
    print(f"  ✅ 收到回答：{word_count} 词")

    return {
        "user_answer": user_answer,
        "phase": "answer_received",
        "messages": [{
            "role": "student",
            "content": user_answer,
            "timestamp": datetime.now().isoformat()
        }],
    }


# ---------------------------------------------------------------------------
# Timer 计时节点
# ---------------------------------------------------------------------------

def timer_node(state: AgentState) -> dict:
    """Timer 节点 — 计算真实耗时并调用 timer_tool。"""
    end_time = time.time()
    start_time = state["question_start_time"]
    elapsed = end_time - start_time

    timer_output = timer_tool.invoke({
        "action": "stop",
        "start_time": start_time,
        "end_time": end_time,
    })

    print(f"  ⏱️  {timer_output}")

    return {
        "elapsed_time": elapsed,
        "timer_result": timer_output,
        "phase": "timer_done",
    }


# ---------------------------------------------------------------------------
# Agent 2: Grammar_Judge（语法考官）
# 【BUGFIX-2】新增"⚠️ 引文保真铁律"→ 强制 LLM 逐字复制原文字符串
# ---------------------------------------------------------------------------

GRAMMAR_JUDGE_SYSTEM_PROMPT = """你是一位严格的雅思语法考官（Grammar Judge），拥有10年以上的英语教学经验。

## 你的任务
仔细分析学生的口语回答，找出所有语法问题，并给出雅思语法维度的预估分数。

## ⚠️ 引文保真铁律 — CRITICAL（违反将导致评估无效）

在报告中引用学生的原始语句时，你必须：
1. **100% 逐字复制**学生的原始输入，**绝对不允许**在引号内进行任何修改或修正。
2. 即使学生的原文包含：
   - 拼写错误（如 "defeinitely"、"recieve"、"occured"、"goverment"）
   - 语法错误（如 "he go"、"she don't"、"I have went"）
   - 标点错误、大小写错误、空格问题
   你也**必须**在"原文"引号中原样保留这些错误。
3. 纠正后的正确形式应放在 "→ 建议" 后面，而不是"原文"中。
4. **错误示范（不可接受）**：
   原文："definitely" → 建议："definitely"
   ← 这等于你擅自修正了用户的拼写！正确的做法是：
   原文："defeinitely" → 建议："definitely"
5. 如果你不确定原文的精确拼写，请回到用户回答中逐字对照确认，确认后再引用。
6. 再次强调：**原文引号中的每一个字符都必须与用户输入完全一致，一个字母都不能改。**

## 分析维度
1. **时态一致性** 2. **主谓一致** 3. **冠词使用** 4. **介词搭配**
5. **句子结构** 6. **从句使用** 7. **语态和语气** 8. **代词指代**

## 输出格式（必须严格遵循）
请使用中文给出反馈：

## 📝 语法分析报告

### ❌ 发现的语法错误：
1. [错误类型] — 原文："..." → 建议："..." （说明）

### ✅ 语法亮点：
1. ...

### 📊 语法维度评估
- 语法准确性：X.X/9
- 语法多样性：X.X/9
- 语法维度综合预估分：X.X/9
- 评语：（1-2句话）

## 注意
- 区分"严重错误"和"轻微错误"
- 引用原文时**必须100%原样保留所有拼写错误**
- 每次至少找出2处改进空间"""


def grammar_judge_node(state: AgentState) -> dict:
    """Agent 2: Grammar_Judge — 语法分析与评分。"""
    print("\n" + "=" * 65)
    print("  🔍 Agent 2: Grammar_Judge（语法考官）— 正在分析语法...")
    print("=" * 65)

    question = state["current_question"]
    answer = state["user_answer"]

    analysis_prompt = f"""请分析以下雅思口语回答中的语法问题。

**口语题目：**
{question}

**学生回答（请严格逐字引用原文，包括所有拼写错误，引用时原样保留错误拼写）：**
{answer}

请严格按照 System Prompt 中要求的格式输出完整的语法分析报告。
⚠️ 最重要的提醒：引用"原文"时保持100%逐字精确，不得擅自修正拼写。"""

    response = model.invoke([
        SystemMessage(content=GRAMMAR_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt)
    ])

    feedback = response.content.strip()
    print(f"  ✅ 语法分析完成 ({len(feedback)} 字符)")

    return {
        "grammar_feedback": feedback,
        "phase": "grammar_done",
        "messages": [{
            "role": "grammar_judge",
            "content": feedback,
            "timestamp": datetime.now().isoformat()
        }],
    }


# ---------------------------------------------------------------------------
# Agent 3: Vocab_Judge（词汇考官）
# ---------------------------------------------------------------------------

VOCAB_JUDGE_SYSTEM_PROMPT = """你是一位专业的雅思词汇考官（Vocabulary Judge），精通英语词汇学和二语习得理论。

## 你的任务
分析学生的词汇使用情况，指出可以提升的地方，并提供更高级/更地道的替换表达。

## 分析维度
1. **词汇多样性 (Lexical Range)** 2. **词汇准确性 (Accuracy)**
3. **搭配地道性 (Collocation)** 4. **同义替换能力 (Paraphrase)**
5. **低频词汇使用 (Less Common Vocabulary)**
6. **习语和短语动词 (Idioms & Phrasal Verbs)**
7. **话题词汇 (Topic-specific Vocabulary)**

## 输出格式（必须严格遵循）
请使用中文给出反馈：

## 📚 词汇分析报告

### ✅ 词汇使用亮点：
1. "..." — （好在哪）

### 🔄 词汇提升建议表：
| 序号 | 原文表达 | 建议替换 | 词汇等级 | 说明 |
|------|----------|----------|----------|------|
| 1    | ...      | ...      | 基础/中级/高级 | ...  |

### 💡 话题相关高级词汇推荐：
1. **词汇**: 词性 — 释义 — 例句

### 📊 词汇维度评估
- 词汇多样性：X.X/9
- 词汇准确性：X.X/9
- 词汇维度综合预估分：X.X/9
- 评语：（1-2句话）

## 注意
- 既肯定亮点也指出提升空间，替换建议循序渐进
- 至少提供5个具体替换建议"""


def vocab_judge_node(state: AgentState) -> dict:
    """Agent 3: Vocab_Judge — 词汇分析与高级替换推荐。"""
    print("\n" + "=" * 65)
    print("  📚 Agent 3: Vocab_Judge（词汇考官）— 正在分析词汇使用...")
    print("=" * 65)

    question = state["current_question"]
    answer = state["user_answer"]

    analysis_prompt = f"""请分析以下雅思口语回答中的词汇使用情况。

**口语题目：**
{question}

**学生回答：**
{answer}

请严格按照 System Prompt 中要求的格式输出完整的词汇分析报告。
特别注意"词汇提升建议表"和"话题相关高级词汇推荐"这两个部分。"""

    response = model.invoke([
        SystemMessage(content=VOCAB_JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt)
    ])

    feedback = response.content.strip()
    print(f"  ✅ 词汇分析完成 ({len(feedback)} 字符)")

    return {
        "vocab_feedback": feedback,
        "phase": "vocab_done",
        "messages": [{
            "role": "vocab_judge",
            "content": feedback,
            "timestamp": datetime.now().isoformat()
        }],
    }


# ---------------------------------------------------------------------------
# Agent 4: Head_Coach（主教练）
# 【评分标准2】Reflection 反思机制
# ---------------------------------------------------------------------------

HEAD_COACH_SYSTEM_PROMPT = """你是一位资深雅思培训主教练（Head Coach），拥有15年雅思教学经验，
曾帮助超过500名学生实现目标分数（6.5-8.0分）。

## 你的任务
汇总语法考官和词汇考官的反馈，进行综合评估，给出预估雅思口语总分和复习建议。

## 【核心机制】Reflection（反思）
在输出最终评分前，你**必须**先完成一个内部反思过程，放入 <thought> 标签中。
反思时需要逐一回答以下问题：

1. **考官反馈一致性检查**：语法考官和词汇考官的反馈是否客观？评分是否有明显偏差？
2. **耗时合理性分析**：学生的答题耗时是否在理想范围内（建议1-2分钟/150-250词）？
3. **历史对比分析**：将学生本次表现与历史记录（长期记忆）对比，
   识别是否有进步或反复出现的薄弱项。
4. **评分权重调整**：根据回答特点，是否需要调整各维度（语法/词汇/流利度/发音）的权重？
5. **综合评分校准**：你的预估总分是否与两位考官的维度评分一致？如有差异请说明理由。

## 输出格式（必须严格遵循）

<thought>
（逐一回答5个反思问题。这是你的内部思考，需要体现深度反思过程。）
</thought>

## 📊 综合评估报告

### 🎯 各维度预估分：
| 维度 | 分数 | 简要说明 |
|------|------|----------|
| 语法 (Grammatical Range & Accuracy) | X.X/9 | 一句话说明 |
| 词汇 (Lexical Resource) | X.X/9 | 一句话说明 |
| 流利度与连贯性 (Fluency & Coherence) | X.X/9 | 基于耗时和内容推断 |
| 发音 (Pronunciation) | X.X/9 | 基于文本复杂度推断 |

### 🎯 预估雅思口语总分：X.X 分

### 📈 历史对比（如果有多于1次记录）：
（对比本次与历史表现的差异，说明进步或退步趋势）

### 📋 分阶段复习建议：

**🔴 短期（1周内）紧急改进：**
1. [具体可操作的建议]

**🟡 中期（1个月）系统提升：**
1. [具体可操作的建议]

**🟢 长期（3个月）战略发展：**
1. [具体可操作的建议]

### 💪 鼓励语：
（一句真诚的鼓励，引导学生保持信心）

## 注意事项
- Reflection 反思过程必须真实深入，不能流于表面
- 预估分数要有理有据，复习建议必须具体可操作"""


def head_coach_node(state: AgentState) -> dict:
    """
    Agent 4: Head_Coach — Reflection 反思 + 综合评分 + 复习建议。

    评分前：
    1. 读取长期记忆，对比历史表现
    2. System Prompt 强制要求 <thought> 反思块
    3. 综合语法/词汇/耗时/历史四维信息给出总分
    """
    print("\n" + "=" * 65)
    print("  🎓 Agent 4: Head_Coach（主教练）— 正在综合评估...")
    print("  🔄 执行 Reflection 反思机制...")
    print("=" * 65)

    # ---- 读取长期记忆 ----
    print("\n  📖 读取长期记忆文件...")
    profile = load_long_term_memory()

    sessions = profile.get("practice_sessions", [])
    if sessions:
        lines = [f"共 {len(sessions)} 次历史练习记录"]
        lines.append(f"历史平均分：{profile.get('average_score', 0)} 分")
        lines.append(f"累计薄弱项：{', '.join(profile.get('cumulative_weaknesses', []))}")
        for s in sessions[-3:]:
            lines.append(
                f"  第{s.get('session_id','?')}次：预估分{s.get('estimated_score','N/A')}，"
                f"薄弱项：{s.get('weaknesses',[])}"
            )
        history_summary = "\n".join(lines)
    else:
        history_summary = "无历史记录（这是首次练习，无法进行历史对比）"

    question = state["current_question"]
    answer = state["user_answer"]
    word_count = len(answer.split()) if answer else 0

    coach_prompt = f"""请对本次雅思口语练习进行综合评估。

## 口语题目
{question}

## 学生回答
{answer}
（共 {word_count} 词）

## 答题耗时
{state['timer_result']}
（雅思Part 2建议：1-2分钟 / 150-250词）

## 语法考官反馈
{state['grammar_feedback']}

## 词汇考官反馈
{state['vocab_feedback']}

## 学生历史记录（长期记忆）
{history_summary}

请严格按照 System Prompt 中的格式：
1. 先在 <thought> 标签中完成深度反思（逐一回答5个反思问题）
2. 然后输出完整的综合评估报告
3. 预估分数需有理有据，复习建议需具体可操作"""

    response = model.invoke([
        SystemMessage(content=HEAD_COACH_SYSTEM_PROMPT),
        HumanMessage(content=coach_prompt)
    ])

    feedback = response.content.strip()

    # 提取预估分数
    estimated_score = 0.0
    for pattern in [
        r'预估雅思口语总分[：:]\s*(\d+\.?\d*)',
        r'总分[：:]\s*(\d+\.?\d*)',
        r'overall[:\s]+(\d+\.?\d*)',
    ]:
        m = re.search(pattern, feedback, re.IGNORECASE)
        if m:
            try:
                estimated_score = float(m.group(1))
            except ValueError:
                pass
            break

    print(f"  ✅ 主教练评估完成 ({len(feedback)} 字符)")
    if estimated_score > 0:
        print(f"  🎯 预估总分：{estimated_score}")

    return {
        "head_coach_feedback": feedback,
        "estimated_score": estimated_score,
        "phase": "coach_done",
        "messages": [{
            "role": "head_coach",
            "content": feedback,
            "timestamp": datetime.now().isoformat()
        }],
    }


# ---------------------------------------------------------------------------
# Output 输出节点（UX 重构版 — 分层报告模板）
# ---------------------------------------------------------------------------

def output_node(state: AgentState) -> dict:
    """
    Output 节点 — 按清晰的分层模板打印最终报告，并更新长期记忆。

    【UX 优化】
    报告结构（严格按此顺序）：
    ====================================
    🎯 本轮雅思口语模拟评估报告
    【⏱️ 答题数据】
    【🧠 主教练深度反思】<thought> 块内容
    【📊 预估总分】
    ─── 各维度详细诊断 ───
    [语法反馈] (精简版)
    [词汇反馈] (精简版)
    ──────────────────────
    【📅 下一步复习建议】
    【💾 长期记忆状态】
    ====================================
    """
    coach_fb = state["head_coach_feedback"]
    grammar_fb = state["grammar_feedback"]
    vocab_fb = state["vocab_feedback"]
    elapsed = state["elapsed_time"]
    user_answer = state["user_answer"]
    word_count = len(user_answer.split()) if user_answer else 0
    score = state["estimated_score"]

    # ---- 提取各内容块 ----
    thought = extract_thought_block(coach_fb)
    review = extract_review_section(coach_fb)
    encouragement = extract_encouragement(coach_fb)
    grammar_concise = extract_concise_feedback(grammar_fb)
    vocab_concise = extract_concise_feedback(vocab_fb)

    # ---- 打印分层报告 ----
    print("\n")
    print("=" * 65)
    print("    🎯 本轮雅思口语模拟评估报告")
    print("=" * 65)

    # ⏱️ 答题数据
    if elapsed >= 60:
        elapsed_display = f"{int(elapsed // 60)}分{int(elapsed % 60)}秒"
    else:
        elapsed_display = f"{elapsed:.1f}秒"
    print(f"\n  【⏱️ 答题数据】")
    print(f"     耗时：{elapsed_display}  |  词数：{word_count} 词")

    # 🧠 主教练深度反思
    if thought:
        thought_display = thought[:800]
        if len(thought) > 800:
            thought_display += "\n  ... (完整反思见上方模型原始输出)"
        print(f"\n  【🧠 主教练深度反思】")
        for line in thought_display.split("\n"):
            line = line.strip()
            if line:
                print(f"     {line}")
    else:
        print(f"\n  【🧠 主教练深度反思】")
        print(f"     (未检测到 <thought> 块，请查看上方完整输出了解反思内容)")

    # 📊 预估总分
    if score > 0:
        if score >= 8.0:
            level = "🔝 专家水平 (Expert)"
        elif score >= 7.0:
            level = "✅ 良好水平 (Good User)"
        elif score >= 6.0:
            level = "👍 合格水平 (Competent User)"
        elif score >= 5.0:
            level = "📘 基础水平 (Modest User)"
        else:
            level = "📚 待提升 (Limited User)"
        print(f"\n  【📊 预估总分】 {score} 分  —  {level}")
    else:
        print(f"\n  【📊 预估总分】 见上方完整报告")

    # ─── 各维度详细诊断 ───
    print(f"\n  {'─' * 55}")
    print(f"  【📝 语法维度诊断】(精简版 — 完整内容见上方输出)")
    print(f"  {'─' * 55}")
    for line in grammar_concise.split("\n"):
        print(f"     {line}")

    print(f"\n  {'─' * 55}")
    print(f"  【📚 词汇维度诊断】(精简版 — 完整内容见上方输出)")
    print(f"  {'─' * 55}")
    for line in vocab_concise.split("\n"):
        print(f"     {line}")

    print(f"\n  {'─' * 55}")

    # 📅 复习建议
    if review:
        print(f"\n  【📅 下一步复习建议】")
        for line in review.split("\n"):
            line = line.strip()
            if line:
                print(f"     {line}")
    else:
        print(f"\n  【📅 下一步复习建议】")
        print(f"     (请参见上方主教练完整报告)")

    # 💪 鼓励语
    if encouragement:
        print(f"\n  【💪】 {encouragement}")

    print(f"\n{'=' * 65}")

    # ---- 薄弱项检测 & 长期记忆更新 ----
    coach_text = state["head_coach_feedback"]
    grammar_text = state["grammar_feedback"]

    weakness_keywords = {
        "时态": ["时态", "tense"],
        "主谓一致": ["主谓一致", "subject-verb"],
        "冠词": ["冠词", "article"],
        "介词": ["介词", "preposition"],
        "从句": ["从句", "clause"],
        "句子结构": ["句子结构", "sentence structure", "残缺句", "run-on"],
        "词汇多样性": ["词汇多样", "lexical range", "词汇量"],
        "词汇准确性": ["词汇准确", "accuracy"],
        "搭配": ["搭配", "collocation"],
        "流利度": ["流利", "fluency"],
        "发音": ["发音", "pronunciation"],
        "同义替换": ["同义替换", "paraphrase", "替换"],
    }

    detected_weaknesses = []
    all_text = (coach_text + " " + grammar_text).lower()
    for cn_name, keywords in weakness_keywords.items():
        for kw in keywords:
            if kw.lower() in all_text:
                detected_weaknesses.append(cn_name)
                break

    profile_update = {
        "question": state["current_question"][:150],
        "answer_word_count": word_count,
        "elapsed_seconds": round(elapsed, 1),
        "weaknesses": detected_weaknesses,
        "estimated_score": score,
    }

    result = update_profile_tool.invoke({
        "profile_json": json.dumps(profile_update, ensure_ascii=False),
        "profile_path": "Qiuyi_ielts_profile.json",
    })

    print(f"\n  【💾 长期记忆状态】")
    print(f"     {result}")
    print(f"\n{'=' * 65}")

    return {"phase": "completed"}


# ============================================================================
# 第7部分：图构建
# ============================================================================

def build_ielts_graph() -> StateGraph:
    """
    构建 LangGraph 工作流图。

    examiner → human_input(interrupt) → timer → grammar_judge → vocab_judge → head_coach → output → END
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("examiner", examiner_node)
    workflow.add_node("human_input", human_input_node)
    workflow.add_node("timer", timer_node)
    workflow.add_node("grammar_judge", grammar_judge_node)
    workflow.add_node("vocab_judge", vocab_judge_node)
    workflow.add_node("head_coach", head_coach_node)
    workflow.add_node("output", output_node)

    workflow.add_edge("examiner", "human_input")
    workflow.add_edge("human_input", "timer")
    workflow.add_edge("timer", "grammar_judge")
    workflow.add_edge("grammar_judge", "vocab_judge")
    workflow.add_edge("vocab_judge", "head_coach")
    workflow.add_edge("head_coach", "output")
    workflow.add_edge("output", END)

    workflow.set_entry_point("examiner")

    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)

    # 尝试渲染并保存 LangGraph 架构图
    try:
        with open("Agent_Architecture.png", "wb") as f:
            f.write(graph.get_graph().draw_mermaid_png())
        print("✅ 架构图已成功渲染并保存为 Agent_Architecture.png")
    except Exception as e:
        print(f"⚠️ 架构图渲染失败: {e}")

    return graph


# ============================================================================
# 第8部分：主函数
# 【BUGFIX-1】输入提示 UI 统一在 graph 流外层打印，确保只出现一次
# ============================================================================

def run_practice_round(graph, config: dict, round_num: int, available_questions: list):
    """
    运行一轮完整的雅思口语练习。

    参数：
        graph:              编译后的 LangGraph 图
        config:             含 thread_id 的配置
        round_num:          当前轮次编号
        available_questions: 跨轮次维护的剩余题库索引列表（由 main() 持有）

    返回：
        (should_continue: bool, available_questions: list)
        - should_continue: True 继续 / False 退出
        - available_questions: 更新后的剩余题库列表

    Human-in-the-loop 流程：
    阶段1 — graph.stream(initial_state) → examiner 出题 → human_input 遇 interrupt() 暂停
    阶段2 — 【外层】打印输入提示 UI（只此一次），input() 获取用户回答
    阶段3 — graph.stream(Command(resume=answer)) → 从 interrupt 恢复 → 计时 → 分析 → 评分 → 输出
    """
    print(f"\n{'='*65}")
    print(f"  🎯 第 {round_num} 轮雅思口语练习")
    print(f"{'='*65}")

    # ---- 检查题库是否已耗尽 ----
    if not available_questions:
        print("\n  📭 题库已耗尽！所有题目均已练习完毕。")
        print("  感谢你的坚持练习！")
        return False, available_questions

    initial_state = {
        "messages": [],
        "current_question": "",
        "user_answer": "",
        "grammar_feedback": "",
        "vocab_feedback": "",
        "head_coach_feedback": "",
        "elapsed_time": 0.0,
        "timer_result": "",
        "estimated_score": 0.0,
        "phase": "init",
        "question_start_time": 0.0,
        "available_questions": available_questions,  # 【v3.1】跨轮次剩余题库池
    }

    # ================================================================
    # 阶段1：运行图到 interrupt 暂停点
    # ================================================================
    print("\n  ⏳ 启动考试流程（考官出题 → 等待回答）...\n")

    last_event = None
    try:
        for event in graph.stream(initial_state, config, stream_mode="values"):
            last_event = event  # 捕获最新的状态快照
    except Exception as e:
        print(f"\n  ⚠️  执行异常：{e}")
        return True, available_questions  # 保留原列表，允许重试

    # ---- 从流事件中提取更新后的题库剩余池 ----
    if last_event and "available_questions" in last_event:
        available_questions = last_event["available_questions"]

    # ================================================================
    # 阶段2：【外层】打印输入 UI —— 只执行一次，resume 不会回到这里
    # ================================================================
    print("\n" + "-" * 65)
    print("  ⏸️  请在下方输入你的英文回答")
    print("  💡 建议回答时长：1-2分钟 | 建议词数：150-250词")
    print("  💡 输入完毕后按回车键提交")
    print("  💡 输入 'quit' 或 'exit' 可退出本轮")
    print("-" * 65)

    user_answer = input("\n> ").strip()

    if not user_answer:
        print("\n  ⚠️  回答为空，请重新开始。")
        return True, available_questions

    if user_answer.lower() in ("quit", "exit", "退出"):
        print("\n  👋 用户选择退出。")
        return False, available_questions

    # ================================================================
    # 阶段3：Command(resume=...) 恢复图，继续 timer → judges → coach → output
    # ================================================================
    print(f"\n  ✅ 收到回答（{len(user_answer.split())} 词）")
    print("\n  ⏳ 正在恢复分析流程...")
    print("  📊 流程：Timer → Grammar_Judge → Vocab_Judge → Head_Coach → Output\n")

    try:
        for event in graph.stream(
            Command(resume=user_answer),
            config,
            stream_mode="values",
        ):
            phase = event.get("phase", "")
            if phase == "completed":
                pass  # output 节点已打印完整分层报告
    except Exception as e:
        print(f"\n  ⚠️  恢复执行异常：{e}")
        return True, available_questions

    return True, available_questions


def main():
    """主入口：构建图 → 循环练习 → 长期记忆持续累积。"""
    print("\n")
    print("🎓" * 33)
    print("      交互式雅思口语模拟陪练与评估系统  v3")
    print("      Interactive IELTS Speaking Practice & Assessment")
    print("🎓" * 33)
    print()
    print("  基于 LangGraph + LangChain 构建的多智能体系统")
    print()
    print("  📋 系统架构：")
    print("     Agent 1: Examiner       — 出题（雅思Part 2 Cue Card）")
    print("     Agent 2: Grammar_Judge  — 语法错误分析 & 评分")
    print("     Agent 3: Vocab_Judge    — 词汇分析 & 高级替换词推荐")
    print("     Agent 4: Head_Coach     — 综合评估 (Reflection) & 复习建议")
    print()
    print("  🔧 核心机制：")
    print("     • Reflection 反思机制    — <thought> 内部思考块")
    print("     • 短期记忆 (AgentState)   — messages + operator.add")
    print("     • 长期记忆 (JSON File)   — Qiuyi_ielts_profile.json")
    print("     • timer_tool             — 真实耗时记录")
    print("     • update_profile_tool    — 文件持久化")
    print("     • Human-in-the-loop      — interrupt + Command(resume)")
    print()
    print("  🐛 v2 修复：")
    print("     BUGFIX-1: human_input 不再打印 UI → 消除 resume 重复 print")
    print("     BUGFIX-2: Grammar_Judge 新增「引文保真铁律」→ 杜绝原文自动纠正")
    print("     UX:       output_node 重构为分层报告模板")
    print()
    print("  🆕 v3 更新：")
    print("     Examiner 采用内置题库 (10题) → 随机抽取 + 去重 → 多轮不重复")
    print()

    #if not os.environ.get("OPENAI_API_KEY"):
    #    print("⚠️  注意：OPENAI_API_KEY 环境变量未设置。")
    #    print("   请执行：$env:OPENAI_API_KEY='你的dashscope API密钥'  (PowerShell)")
    #    print("   或：    export OPENAI_API_KEY='你的dashscope API密钥'  (Bash)")
    #    print()

    print("🔧 正在构建 LangGraph 工作流图...")
    graph = build_ielts_graph()
    print("✅ 图构建完成！（7个节点，7条边）\n")

    config = {"configurable": {"thread_id": "ielts_practice_session"}}

    # ---- 【v3.1 关键修复】在 while 循环外部初始化剩余题库池 ----
    # 该列表的生命周期跨越所有轮次，确保计数器不会每轮重置
    available_questions = list(range(len(IELTS_PART2_QUESTION_BANK)))
    total_questions = len(available_questions)

    round_num = 1
    while True:
        if not available_questions:
            print("\n" + "=" * 65)
            print(f"  📭 题库已耗尽！全部 {total_questions} 道题目均已练习完毕。")
            print("  系统将优雅退出，感谢你的坚持练习！")
            print("=" * 65)
            break

        should_continue, available_questions = run_practice_round(
            graph, config, round_num, available_questions
        )
        if not should_continue:
            break

        round_num += 1

        # 若题库刚被用完，提示用户
        if not available_questions:
            print(f"\n  🎉 恭喜！你已完成全部 {total_questions} 道题库题目！")
            print("  是否再来一轮？（题库将自动重置）")
            print("  [输入 y/Y 重置并继续] [输入 n/N 或任意键退出]")
            choice = input("> ").strip().lower()
            if choice in ("y", "yes", "是", "继续", "重置"):
                available_questions = list(range(total_questions))
                print(f"\n  🔄 题库已重置：{len(available_questions)} 道题目可用。")
                continue
            else:
                break

        print("\n" + "-" * 65)
        print("  是否再进行一轮练习？")
        print(f"  [输入 y/Y 继续（剩余{len(available_questions)}题）] [输入 n/N 或任意键退出]")
        print("-" * 65)
        choice = input("> ").strip().lower()
        if choice not in ("y", "yes", "是", "继续"):
            break

        print("\n" + "🔄" * 33)
        print(f"  开始新一轮练习...（题库剩余 {len(available_questions)} 题）")
        print("🔄" * 33)

    print("\n" + "=" * 65)
    print("  👋 感谢使用雅思口语陪练系统！")
    print("  📝 你的长期记忆已保存至：Qiuyi_ielts_profile.json")
    print("  🎓 祝雅思考试顺利，取得理想分数！")
    print("=" * 65)
    print()


if __name__ == "__main__":
    main()
