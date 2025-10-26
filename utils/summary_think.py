"""
Summary Think 功能
根据实际推理进度动态生成思维链
"""
from typing import Dict, Any
from models import ProgressEvent


class ThinkingSummaryGenerator:
    """
    思维链生成器 - 根据引擎的实际进度事件动态生成思维链文本
    """
    
    def __init__(self, mode: str = "deepthink"):
        self.mode = mode
        self.started = False
        self.current_iteration = -1
        self.current_phase = None
        self.agent_states: Dict[str, Dict[str, Any]] = {}
    
    def process_event(self, event: ProgressEvent) -> str:
        """
        处理进度事件,生成对应的思维链文本 (o1 专业风格)
        
        Args:
            event: 进度事件
        
        Returns:
            思维链文本块 (如果有)
        """
        output = ""
        event_type = event.type
        data = event.data
        
        # 初始化
        if event_type == "init":
            if not self.started:
                self.started = True
                output += "Analyzing the problem...\n\n"
        
        # 思考阶段
        elif event_type == "thinking":
            iteration = data.get("iteration", 0)
            phase = data.get("phase", "")
            
            # 新的迭代轮次
            if iteration != self.current_iteration:
                self.current_iteration = iteration
                if iteration > 0:
                    output += f"\nIteration {iteration + 1}\n"
            
            # 新的阶段
            if phase != self.current_phase:
                self.current_phase = phase
                phase_name = self._format_phase_name(phase)
                output += f"{phase_name}\n"
        
        # 生成解决方案
        elif event_type == "solution":
            # o1 风格：不显示每个解决方案生成
            pass
        
        # 验证
        elif event_type == "verification":
            passed = data.get("passed", False)
            if passed:
                output += "Verified\n"
            else:
                output += "Refining approach\n"
        
        # 修正
        elif event_type == "correction":
            # o1 风格：合并到验证流程
            pass
        
        # 总结
        elif event_type == "summarizing":
            output += "\nGenerating final answer...\n"
        
        # 成功
        elif event_type == "success":
            iterations = data.get("iterations", 0)
            stats = data.get("statistics", {})
            output += f"\nAnalysis complete ({iterations} iterations)\n"
            
            # 显示统计信息
            if stats:
                api_calls = stats.get("api_calls", 0)
                total_tokens = stats.get("total_tokens", 0)
                output += f"API calls: {api_calls}"
                if total_tokens > 0:
                    output += f" | Tokens: {total_tokens:,}"
                output += "\n"
            
            output += "\n---\n\n"
        
        # 失败
        elif event_type == "failure":
            reason = data.get("reason", "Unknown")
            stats = data.get("statistics", {})
            output += f"\nNote: {reason}\n"
            
            # 显示统计信息
            if stats:
                api_calls = stats.get("api_calls", 0)
                total_tokens = stats.get("total_tokens", 0)
                output += f"API calls: {api_calls}"
                if total_tokens > 0:
                    output += f" | Tokens: {total_tokens:,}"
                output += "\n"
            
            output += "\n---\n\n"
        
        # 进度消息
        elif event_type == "progress":
            # o1 风格：隐藏详细进度消息
            pass
        
        # 计划阶段
        elif event_type == "planning":
            output += "Planning approach\n"
        
        return output
    
    def _format_phase_name(self, phase: str) -> str:
        """
        格式化阶段名称 (o1 专业风格)
        
        Args:
            phase: 阶段标识
        
        Returns:
            格式化的阶段名称
        """
        phase_map = {
            "initial-exploration": "Exploring approaches",
            "self-improvement": "Refining solution",
            "initializing": "Initializing",
            "correcting": "Applying corrections",
            "summarizing": "Summarizing",
            "verifying": "Verifying solution",
        }
        
        default_name = phase.replace("_", " ").replace("-", " ").title()
        return phase_map.get(phase, default_name)


class UltraThinkSummaryGenerator(ThinkingSummaryGenerator):
    """
    UltraThink 模式的思维链生成器
    额外处理 Agent 状态更新
    """
    
    def __init__(self):
        super().__init__(mode="ultrathink")
        self.agent_count = 0
        self.completed_agents = 0
    
    def process_event(self, event: ProgressEvent) -> str:
        """处理 UltraThink 事件 (o1 专业风格)"""
        output = ""
        event_type = event.type
        data = event.data
        
        # 重写初始化消息
        if event_type == "init":
            if not self.started:
                self.started = True
                output += "Analyzing with multiple approaches...\n\n"
            return output
        
        # 处理 Agent 更新
        if event_type == "agent-update":
            agent_id = data.get("agentId", "")
            status = data.get("status", "")
            approach = data.get("approach", "")
            
            # 新 Agent 启动
            if agent_id and agent_id not in self.agent_states:
                self.agent_count += 1
                self.agent_states[agent_id] = {"approach": approach}
                if approach:
                    output += f"Approach {self.agent_count}: {approach}\n"
            
            # 状态变化
            if agent_id and status:
                old_status = self.agent_states[agent_id].get("status")
                if status != old_status:
                    self.agent_states[agent_id]["status"] = status
                    
                    if status == "completed":
                        self.completed_agents += 1
                        # 只显示完成状态
                        output += f"  Completed ({self.completed_agents}/{self.agent_count})\n"
                    
                    elif status == "failed":
                        # 只显示失败，不显示详细错误
                        output += f"  Approach encountered difficulties\n"
        
        # 综合阶段
        elif event_type == "progress":
            message = data.get("message", "")
            if "Synthesizing" in message:
                if "deep thinking" in message.lower():
                    output += "\nSynthesizing results with deep analysis...\n"
                else:
                    output += "\nSynthesizing results from all approaches...\n"
            elif "Deep thinking on synthesis" in message:
                output += "  Analyzing synthesis approach\n"
            elif "Synthesis verified" in message:
                output += "  Synthesis verified\n"
            elif "Refining synthesis" in message:
                output += "  Refining synthesis\n"
        
        # 生成摘要
        elif event_type == "summarizing":
            output += "Generating final comprehensive answer...\n"
        
        # 成功完成
        elif event_type == "success":
            stats = data.get("statistics", {})
            output += f"\nAnalysis complete\n"
            
            # 显示统计信息
            if stats:
                api_calls = stats.get("api_calls", 0)
                total_tokens = stats.get("total_tokens", 0)
                output += f"API calls: {api_calls}"
                if total_tokens > 0:
                    output += f" | Tokens: {total_tokens:,}"
                output += "\n"
            
            output += "\n---\n\n"
        
        else:
            # 其他事件用父类处理
            output = super().process_event(event)
        
        return output


def generate_simple_thinking_tag(mode: str = "deepthink") -> str:
    """
    生成简单的思考标签(非流式场景) - o1 专业风格
    
    Args:
        mode: 模式 (deepthink 或 ultrathink)
    
    Returns:
        简单的思考提示
    """
    if mode == "ultrathink":
        return """Analyzing with multiple approaches...

Exploring different solution paths
Verifying feasibility
Synthesizing results from all approaches
Generating final comprehensive answer

Analysis complete

---

"""
    else:
        return """Analyzing the problem...

Understanding requirements
Exploring approaches
Refining solution
Verifying correctness
Generating final answer

Analysis complete

---

"""

