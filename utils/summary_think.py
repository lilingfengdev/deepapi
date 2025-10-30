"""
思维链总结生成器
支持简单文本映射和 LLM 智能总结两种模式
"""
from typing import Dict, Any, Optional, List
import asyncio
import time
import logging
from models import ProgressEvent
from utils.openai_client import OpenAIClient

logger = logging.getLogger(__name__)


def process_thinking_event(event: ProgressEvent, mode: str = "deepthink") -> str:
    """处理进度事件，生成思维链文本（简单映射版本）"""
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


class LLMThinkingSummaryGenerator:
    """使用 LLM 生成 o1 风格的实时思维链总结"""
    
    def __init__(
        self, 
        client: OpenAIClient,
        model: str,
        mode: str = "deepthink",
        fallback_to_simple: bool = True  # 失败时是否回退到简单模式
    ):
        self.client = client
        self.model = model
        self.mode = mode
        self.fallback_to_simple = fallback_to_simple
        
        # 不需要历史，每个事件独立处理
        self.current_iteration = 0
        
        # 简单模式生成器作为后备
        self.simple_generator = ThinkingSummaryGenerator(mode) if mode == "deepthink" else UltraThinkSummaryGenerator()
        
        # 用于存储待处理的事件队列
        self.event_queue = asyncio.Queue()
        self.summary_queue = asyncio.Queue()
    
    async def process_event_async(self, event: ProgressEvent) -> str:
        """异步处理事件，生成 o1 风格的思维链总结"""
        try:
            # 构建事件描述
            event_description = self._format_event_for_llm(event)
            if not event_description:
                return ""
            
            # 直接调用异步方法，不保存历史
            summary = await self._generate_summary_async(event, event_description)
            return summary
            
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            # 回退到简单模式
            if self.fallback_to_simple:
                return self.simple_generator.process_event(event)
            return ""
    
    def process_event(self, event: ProgressEvent) -> str:
        """同步接口 - 为了保持向后兼容"""
        # 如果在异步环境中，只能返回简单结果
        try:
            loop = asyncio.get_running_loop()
            # 在异步环境中，回退到简单模式
            logger.debug("In async context, falling back to simple mode")
            return self.simple_generator.process_event(event)
        except RuntimeError:
            # 不在异步环境，可以创建新循环
            return self._generate_summary_sync(event)
    
    def _generate_summary_sync(self, event: ProgressEvent) -> str:
        """同步生成总结（仅在非异步环境中使用）"""
        try:
            # 构建事件描述
            event_description = self._format_event_for_llm(event)
            if not event_description:
                return ""
            
            # 创建新的事件循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                summary = loop.run_until_complete(
                    self._generate_summary_async(event, event_description)
                )
                return summary
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            # 回退到简单模式
            if self.fallback_to_simple:
                return self.simple_generator.process_event(event)
            return ""
    
    async def _generate_summary_async(self, event: ProgressEvent, event_description: str) -> str:
        """异步生成 o1 风格的思考总结"""
        # 根据事件类型选择合适的提示词
        event_type = event.type
        data = event.data
        
        # 根据事件类型生成不同风格的总结
        system_prompt = """You are the AI's inner voice. Generate first-person thoughts that include SPECIFIC DETAILS about what you're doing.
Use "I", "my", "me" - speak AS the AI. Include concrete details from the actual content.
Examples:
- "I need to implement a recursive function to calculate the Fibonacci sequence."
- "I'm realizing the user wants self-awareness discussion, not just a greeting."
- "My verification shows the solution correctly handles edge cases for n <= 1."
Keep it under one line. ONLY output the thought itself."""
        
        # 根据事件类型调整 prompt - 第一人称 + 要求具体内容
        if event_type == "init":
            prompt = f"You are analyzing this problem: {event_description}\nExpress your specific first thought about what you need to do:"
        elif event_type == "thinking" and data.get("phase") == "initial-exploration":
            prompt = f"You are exploring this approach: {event_description}\nWhat specific method or idea are you considering:"
        elif event_type == "thinking" and data.get("phase") == "correction":
            prompt = f"You found this issue: {event_description}\nWhat specific fix are you implementing:"
        elif event_type == "verification":
            if data.get("passed"):
                prompt = f"Your verification shows: {event_description}\nWhat specific aspect passed verification:"
            else:
                prompt = f"Your verification found: {event_description}\nWhat specific problem did you discover:"
        elif event_type == "planning":
            prompt = f"You are planning: {event_description}\nWhat specific steps are you outlining:"
        elif event_type == "thinking" and data.get("solution"):
            prompt = f"You generated this solution: {event_description}\nWhat specific approach did you take:"
        else:
            prompt = f"Current content: {event_description}\nWhat specific aspect are you focusing on:"
        
        try:
            summary = await self.client.generate_text(
                model=self.model,
                system=system_prompt,
                prompt=prompt,
                temperature=0.7,  # 稍低一点，更稳定
                max_tokens=200   # 控制在一行内
            )
            
            # 根据事件类型添加适当的格式
            formatted_summary = self._format_summary(event_type, summary.strip(), data)
            return formatted_summary
            
        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}")
            raise
    
    def _format_summary(self, event_type: str, summary: str, data: Dict[str, Any]) -> str:
        """格式化总结文本 - 简洁的单行格式"""
        # 保持简洁，所有输出都是单行
        return summary.strip() + "\n"
    
    def _format_event_for_llm(self, event: ProgressEvent) -> str:
        """格式化单个事件，包含 AI 生成的内容"""
        event_type = event.type
        data = event.data
        
        if event_type == "init":
            problem = data.get('problem', '')
            if problem:
                # 不截断，把完整内容传给总结模型
                return problem  # 直接返回问题内容，让总结模型处理
            return ""
            
        elif event_type == "thinking":
            iteration = data.get('iteration', 0) + 1
            self.current_iteration = iteration
            solution = data.get('solution', '')
            phase = data.get('phase', '')
            
            if phase:
                # 有特定阶段信息
                return f"Thinking ({phase}): iteration {iteration}"
            elif solution:
                # 不截断，把完整解决方案传给总结模型
                return solution
            return ""
            
        elif event_type == "verification":
            passed = data.get('passed', False)
            bug_report = data.get('bug_report', '')
            good_verify = data.get('good_verify', '')
            
            if passed:
                return good_verify  # 让总结模型处理
            else:
                return bug_report   # 让总结模型处理
                
        elif event_type == "success":
            iterations = data.get('iterations', self.current_iteration)
            solution = data.get('solution', '')
            return ""  # 让总结模型决定怎么表述完成
            
        elif event_type == "failure":
            reason = data.get('reason', 'Unknown')
            return reason  # 让总结模型决定怎么表述失败
            
        elif event_type == "planning":
            plan = data.get('plan', '')
            if plan:
                return plan  # 让总结模型处理
            return ""
            
        elif event_type == "agent-start":
            agent_id = data.get('agentId', '')
            approach = data.get('approach', '')
            specific_prompt = data.get('specific_prompt', '')
            return f"{approach}. {specific_prompt}"  # 简化内容
            
        elif event_type == "agent-complete":
            agent_id = data.get('agentId', '')
            solution = data.get('solution', '')
            if solution:
                return solution  # 让总结模型处理
            return ""
            
        elif event_type == "synthesis":
            return ""  # 综合阶段没有具体内容
            
        elif event_type == "summarizing":
            return ""  # 总结阶段没有具体内容
            
        elif event_type == "progress":
            message = data.get('message', '')
            if message:
                return message
            return ""
            
        else:
            # 未知事件类型，尝试提取有意义的信息
            if 'message' in data:
                return data['message']
            elif 'solution' in data:
                return data['solution']  # 返回原始内容
            return ""
    
    def reset(self):
        """重置计数器"""
        self.current_iteration = 0


def generate_simple_thinking_tag(mode: str = "deepthink") -> str:
    """生成简单的思考标签"""
    return f"""Analyzing with {mode}...

Analysis complete

---

"""