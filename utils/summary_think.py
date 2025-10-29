"""
简化版思维链生成器
把280行垃圾代码简化成50行
"""
from typing import Dict, Any
from models import ProgressEvent


def process_thinking_event(event: ProgressEvent, mode: str = "deepthink") -> str:
    """处理进度事件，生成思维链文本"""
    event_type = event.type
    data = event.data
    
    # 简单的事件映射
    event_messages = {
        "init": "Analyzing the problem...\n\n",
        "thinking": f"Iteration {data.get('iteration', 0) + 1}\n",
        "verification": "Verified\n" if data.get('passed') else "Refining approach\n",
        "summarizing": "\nGenerating final answer...\n",
        "planning": f"Planning approach\n",
        "agent-start": f"Agent {data.get('agentId', '')} starting: {data.get('approach', '')}\n",
        "agent-complete": f"Agent {data.get('agentId', '')} completed\n",
        "synthesis": "Synthesizing results...\n",
    }
    
    # 成功/失败特殊处理
    if event_type == "success":
        iterations = data.get('iterations', 0)
        return f"\nAnalysis complete ({iterations} iterations)\n\n---\n\n"
    elif event_type == "failure":
        return f"\nAnalysis failed: {data.get('reason', 'Unknown')}\n"
    
    return event_messages.get(event_type, "")


class ThinkingSummaryGenerator:
    """保持接口兼容的简化版生成器"""
    def __init__(self, mode: str = "deepthink"):
        self.mode = mode
    
    def process_event(self, event: ProgressEvent) -> str:
        return process_thinking_event(event, self.mode)


class UltraThinkSummaryGenerator(ThinkingSummaryGenerator):
    """UltraThink版本"""
    def __init__(self):
        super().__init__("ultrathink")


def generate_simple_thinking_tag(mode: str = "deepthink") -> str:
    """生成简单的思考标签"""
    return f"""Analyzing with {mode}...

Analysis complete

---

"""