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
        
        # 保存思考历史，用于上下文
        self.thinking_history: List[str] = []
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
            
            # 直接调用异步方法
            summary = await self._generate_summary_async(event, event_description)
            
            # 保存到历史中
            if summary:
                self.thinking_history.append(summary)
                # 只保留最近的 20 条思考
                if len(self.thinking_history) > 20:
                    self.thinking_history = self.thinking_history[-20:]
            
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
        # 获取最近的思考历史作为上下文
        recent_thoughts = "\n".join(self.thinking_history[-3:]) if self.thinking_history else ""
        
        # 根据事件类型选择合适的提示词
        event_type = event.type
        data = event.data
        
        # 根据事件类型生成不同风格的总结
        system_prompt = """You are an AI's internal reasoning process. Generate thoughts in the style of OpenAI's o1 model:
- Natural, conversational inner monologue
- Short paragraphs with clear thinking
- Use "Hmm", "Actually", "Wait", "Oh", etc. for natural transitions
- Show the thinking process, including corrections and realizations"""
        
        # 根据事件类型调整 prompt
        if event_type == "init":
            prompt = f"I'm starting to analyze a new problem. {event_description}\n\nGenerate the first thought:"
        elif event_type == "thinking" and data.get("phase") == "initial-exploration":
            prompt = f"Beginning initial exploration. {event_description}\n\nGenerate a thought about starting the analysis:"
        elif event_type == "thinking" and data.get("phase") == "correction":
            prompt = f"Found an issue that needs correction. {event_description}\n\nGenerate a thought about fixing the problem:"
        elif event_type == "verification":
            if data.get("passed"):
                prompt = f"Verification result: {event_description}\n\nGenerate a thought about the successful verification:"
            else:
                prompt = f"Verification failed: {event_description}\n\nGenerate a thought about what went wrong:"
        elif event_type == "planning":
            prompt = f"Creating a plan: {event_description}\n\nGenerate a thought about the planning process:"
        else:
            prompt = f"Current situation: {event_description}\n\nGenerate a natural thought:"
        
        # 添加历史上下文
        if recent_thoughts:
            prompt = f"Recent thoughts:\n{recent_thoughts}\n\n{prompt}"
        
        try:
            summary = await self.client.generate_text(
                model=self.model,
                system=system_prompt,
                prompt=prompt,
                temperature=0.7,  # 稍低一点，更稳定
                max_tokens=150  # 增加 token 数量
            )
            
            # 根据事件类型添加适当的格式
            formatted_summary = self._format_summary(event_type, summary.strip(), data)
            return formatted_summary
            
        except Exception as e:
            logger.error(f"Failed to generate LLM summary: {e}")
            raise
    
    def _format_summary(self, event_type: str, summary: str, data: Dict[str, Any]) -> str:
        """格式化总结文本，添加适当的标题和空行"""
        # 特殊事件的标题
        if event_type == "init":
            return f"\n# Analyzing the problem\n\n{summary}\n\n"
        elif event_type == "thinking" and data.get("phase") == "initial-exploration":
            return f"\n## Initial exploration\n\n{summary}\n\n"
        elif event_type == "thinking" and data.get("phase") == "correction":
            iteration = data.get("iteration", 0)
            return f"\n## Correction (iteration {iteration})\n\n{summary}\n\n"
        elif event_type == "verification":
            iteration = data.get("iteration", 0)
            if data.get("passed"):
                return f"\n## Verification passed (iteration {iteration})\n\n{summary}\n\n"
            else:
                return f"\n## Verification failed (iteration {iteration})\n\n{summary}\n\n"
        elif event_type == "planning":
            return f"\n# Planning approach\n\n{summary}\n\n"
        elif event_type == "summarizing":
            return f"\n# Generating final answer\n\n{summary}\n\n"
        elif event_type == "success":
            return f"\n# Analysis complete\n\n{summary}\n\n"
        elif event_type == "thinking":
            # 普通 thinking 事件（没有特定 phase）
            iteration = data.get('iteration')
            if iteration is not None:
                return f"\n## Iteration {iteration}\n\n{summary}\n\n"
            else:
                return f"{summary}\n\n"
        else:
            # 其他事件只添加换行
            return f"{summary}\n\n"
    
    def _format_event_for_llm(self, event: ProgressEvent) -> str:
        """格式化单个事件，包含 AI 生成的内容"""
        event_type = event.type
        data = event.data
        
        if event_type == "init":
            problem = data.get('problem', '')
            if problem:
                # 截取问题的前 200 个字符
                problem_preview = problem[:200] + "..." if len(problem) > 200 else problem
                return f"Looking at the problem: {problem_preview}"
            return "Starting to analyze the problem"
            
        elif event_type == "thinking":
            iteration = data.get('iteration', 0) + 1
            self.current_iteration = iteration
            solution = data.get('solution', '')
            phase = data.get('phase', '')
            
            if phase:
                # 有特定阶段信息
                return f"Thinking ({phase}): iteration {iteration}"
            elif solution:
                # 提取解决方案的关键部分
                solution_preview = solution[:200] + "..." if len(solution) > 200 else solution
                return f"Working on iteration {iteration}. Current approach: {solution_preview}"
            return f"Working on iteration {iteration}"
            
        elif event_type == "verification":
            passed = data.get('passed', False)
            bug_report = data.get('bug_report', '')
            good_verify = data.get('good_verify', '')
            
            if passed:
                return f"Verification passed. The solution looks correct: {good_verify[:100]}..."
            else:
                return f"Found an issue: {bug_report[:150]}... Need to refine the approach."
                
        elif event_type == "success":
            iterations = data.get('iterations', self.current_iteration)
            solution = data.get('solution', '')
            return f"Completed analysis after {iterations} iterations."
            
        elif event_type == "failure":
            reason = data.get('reason', 'Unknown')
            return f"Hit a roadblock: {reason}. May need a different approach."
            
        elif event_type == "planning":
            plan = data.get('plan', '')
            if plan:
                return f"Planning the approach: {plan[:200]}..."
            return "Developing a strategy to tackle this problem"
            
        elif event_type == "agent-start":
            agent_id = data.get('agentId', '')
            approach = data.get('approach', '')
            specific_prompt = data.get('specific_prompt', '')
            return f"Agent {agent_id} exploring: {approach}. Focus: {specific_prompt[:100]}..."
            
        elif event_type == "agent-complete":
            agent_id = data.get('agentId', '')
            solution = data.get('solution', '')
            if solution:
                return f"Agent {agent_id} finished. Found: {solution[:150]}..."
            return f"Agent {agent_id} completed its analysis"
            
        elif event_type == "synthesis":
            return "Now combining insights from all the different approaches..."
            
        elif event_type == "summarizing":
            return "Preparing the final answer based on all the analysis..."
            
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
                return f"{event_type}: {data['solution'][:100]}..."
            return ""
    
    def reset(self):
        """重置思考历史"""
        self.thinking_history.clear()
        self.current_iteration = 0


def generate_simple_thinking_tag(mode: str = "deepthink") -> str:
    """生成简单的思考标签"""
    return f"""Analyzing with {mode}...

Analysis complete

---

"""