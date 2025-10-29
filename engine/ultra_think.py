"""
Ultra Think 引擎
多 Agent 并行探索引擎
"""
import asyncio
from typing import Optional, Callable, List, Dict, Any
import json

from models import (
    UltraThinkResult,
    AgentResult,
    Source,
    ProgressEvent,
    MessageContent,
    extract_text_from_content
)
from utils.openai_client import OpenAIClient
from engine.prompts import (
    ULTRA_THINK_PLAN_PROMPT,
    GENERATE_AGENT_PROMPTS_PROMPT,
    SYNTHESIZE_RESULTS_PROMPT,
    build_final_summary_prompt,
)
from engine.deep_think import DeepThinkEngine


class UltraThinkEngine:
    """Ultra Think 引擎 - 多 Agent 并行探索"""
    
    def __init__(
        self,
        client: OpenAIClient,
        model: str,
        problem_statement: MessageContent,  # 支持多模态内容
        conversation_history: List[Dict[str, Any]] = None,  # 完整的消息历史
        other_prompts: List[str] = None,  # 已弃用，保留向后兼容
        knowledge_context: str = None,
        max_iterations: int = 30,
        required_successful_verifications: int = 3,
        max_errors_before_give_up: int = 10,
        num_agents: Optional[int] = None,
        parallel_run_agent: int = 3,
        model_stages: Dict[str, str] = None,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        on_agent_update: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        enable_parallel_check: bool = False,
        llm_params: Optional[Dict[str, Any]] = None,
    ):
        self.client = client
        self.model = model
        self.problem_statement = problem_statement  # 可能是字符串或多模态内容
        self.problem_statement_text = extract_text_from_content(problem_statement)  # 提取纯文本版本
        self.conversation_history = conversation_history or []  # 结构化的消息历史
        self.other_prompts = other_prompts or []  # 向后兼容
        self.knowledge_context = knowledge_context
        self.max_iterations = max_iterations
        self.required_verifications = required_successful_verifications
        self.max_errors = max_errors_before_give_up
        self.num_agents = num_agents
        self.parallel_run_agent = parallel_run_agent
        self.model_stages = model_stages or {}
        self.on_progress = on_progress
        self.on_agent_update = on_agent_update
        self.enable_parallel_check = enable_parallel_check
        self.sources: List[Source] = []
        self.llm_params = llm_params or {}
    
    def _get_model_for_stage(self, stage: str) -> str:
        """获取特定阶段的模型"""
        return self.model_stages.get(stage, self.model)
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """发送进度事件"""
        if self.on_progress:
            self.on_progress(ProgressEvent(type=event_type, data=data))
    
    async def _generate_plan(self, problem_statement: MessageContent) -> str:
        """生成思考计划"""
        self._emit("progress", {"message": "Generating thinking plan..."})
        
        planning_model = self._get_model_for_stage("planning")
        
        # 构建消息列表：历史消息 + 当前问题
        messages = []
        if self.conversation_history:
            messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": problem_statement})
        
        # 使用完整的消息历史生成计划
        plan = await self.client.generate_text(
            model=planning_model,
            messages=messages,
            **self.llm_params
        )
        
        return plan
    
    async def _generate_agent_configs(self, plan: str) -> List[Dict[str, str]]:
        """生成 Agent 配置"""
        self._emit("progress", {"message": "Generating agent configurations..."})
        
        agent_config_model = self._get_model_for_stage("agent_config")
        
        try:
            # 尝试使用结构化输出
            result = await self.client.generate_object(
                model=agent_config_model,
                prompt=GENERATE_AGENT_PROMPTS_PROMPT.replace("{plan}", plan),
                response_format={"type": "json_object"},
                **self.llm_params
            )
            
            # 处理可能的响应格式
            if isinstance(result, list):
                configs = result
            elif "configs" in result:
                configs = result["configs"]
            else:
                configs = [result]
            
            return configs
        
        except Exception as e:
            # 回退到文本解析
            text = await self.client.generate_text(
                model=agent_config_model,
                prompt=GENERATE_AGENT_PROMPTS_PROMPT.replace("{plan}", plan),
                **self.llm_params
            )
            
            # 清理和解析JSON
            json_text = text.strip()
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            parsed = json.loads(json_text)
            return parsed if isinstance(parsed, list) else parsed.get("configs", parsed)
    
    async def _run_agent(
        self,
        config: Dict[str, str],
        problem_statement: MessageContent,
    ) -> AgentResult:
        """运行单个 Agent"""
        agent_id = config["agentId"]
        approach = config["approach"]
        specific_prompt = config["specificPrompt"]
        
        result = AgentResult(
            agent_id=agent_id,
            approach=approach,
            specific_prompt=specific_prompt,
            status="thinking",
            progress=0,
        )
        
        # 通知 agent 开始
        if self.on_agent_update:
            self.on_agent_update(agent_id, {"status": "thinking", "progress": 10})
        
        try:
            # Agent 思考阶段使用专门的模型
            agent_thinking_model = self._get_model_for_stage("agent_thinking")
            
            # 创建进度处理器
            def agent_progress_handler(event: ProgressEvent):
                if event.type == "thinking":
                    progress = min(20 + event.data.get("iteration", 0) * 2, 80)
                    result.progress = progress
                    result.status = "thinking"
                    if self.on_agent_update:
                        self.on_agent_update(agent_id, {"progress": progress, "status": "thinking"})
                elif event.type == "verification":
                    result.status = "verifying"
                    if self.on_agent_update:
                        self.on_agent_update(agent_id, {"status": "verifying"})
                elif event.type == "success":
                    result.status = "completed"
                    result.progress = 100
                    if self.on_agent_update:
                        self.on_agent_update(agent_id, {"status": "completed", "progress": 100})
                elif event.type == "failure":
                    result.status = "failed"
                    result.error = event.data.get("reason", "Unknown error")
                    if self.on_agent_update:
                        self.on_agent_update(agent_id, {"status": "failed", "error": result.error})
            
            # 运行 Deep Think 引擎
            # 为 agent 创建增强的消息历史，包含特定提示词
            agent_history = []
            if self.conversation_history:
                agent_history.extend(self.conversation_history)
            
            # 如果有特定提示词，添加为系统级指导（通过在问题前添加上下文）
            # 注意：这里我们将 agent 特定提示词作为上下文，不改变核心问题
            
            engine = DeepThinkEngine(
                client=self.client,
                model=agent_thinking_model,
                problem_statement=problem_statement,
                conversation_history=agent_history,
                other_prompts=[specific_prompt],  # agent 特定提示词作为额外上下文
                knowledge_context=self.knowledge_context,
                max_iterations=self.max_iterations,
                required_successful_verifications=self.required_verifications,
                max_errors_before_give_up=self.max_errors,
                model_stages=self.model_stages,
                on_progress=agent_progress_handler,
                enable_parallel_check=self.enable_parallel_check,
                llm_params=self.llm_params,
            )
            
            deep_think_result = await engine.run()
            
            result.solution = deep_think_result.final_solution
            result.verifications = deep_think_result.verifications
            result.status = "completed"
            result.progress = 100
            
            # 收集搜索来源
            if deep_think_result.sources:
                self.sources.extend(deep_think_result.sources)
            
            if self.on_agent_update:
                self.on_agent_update(agent_id, {
                    "status": "completed",
                    "progress": 100,
                    "solution": deep_think_result.final_solution,
                    "verifications": deep_think_result.verifications,
                })
        
        except Exception as err:
            result.status = "failed"
            result.error = str(err)
            if self.on_agent_update:
                self.on_agent_update(agent_id, {
                    "status": "failed",
                    "error": result.error,
                })
        
        return result
    
    async def run(self) -> UltraThinkResult:
        """运行 Ultra Think 引擎"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 发送事件时使用文本版本
            self._emit("init", {"problem": self.problem_statement_text})
            
            # 生成计划 (UltraThink 内置功能) - 传递多模态内容
            plan = await self._generate_plan(self.problem_statement)
            
            # 生成 agent 配置
            configs = await self._generate_agent_configs(plan)
            
            # 使用所有 LLM 建议的 agents,或者限制数量
            selected_configs = configs[:self.num_agents] if self.num_agents else configs
            num_agents = len(selected_configs)
            
            # 更新 agent 配置到 UI
            if self.on_agent_update:
                for config in selected_configs:
                    self.on_agent_update(config["agentId"], {
                        "approach": config["approach"],
                        "specific_prompt": config["specificPrompt"],
                    })
            
            # 并行运行 agents (使用信号量控制并发数)
            self._emit("progress", {
                "message": f"Running {num_agents} agents in parallel..."
            })
            
            semaphore = asyncio.Semaphore(self.parallel_run_agent)
            
            async def run_with_semaphore(config):
                async with semaphore:
                    return await self._run_agent(config, self.problem_statement)
            
            agent_results = await asyncio.gather(*[
                run_with_semaphore(config) for config in selected_configs
            ])
            
            # 综合结果 - 使用 DeepThink 引擎进行深度思考
            self._emit("progress", {"message": "Synthesizing results with deep thinking..."})
            
            agent_results_text = "\n\n---\n\n".join([
                f"""
### Agent {idx + 1}: {result.approach}

**Status:** {result.status}
{f"**Error:** {result.error}" if result.error else ""}

**Solution:**
{result.solution or "No solution generated"}
"""
                for idx, result in enumerate(agent_results)
            ])
            
            # 构建综合问题提示 - 使用文本版本
            synthesis_problem = f"""Based on the following problem and multiple solution approaches, synthesize a comprehensive final solution:

**Original Problem:**
{self.problem_statement_text}

**Solution Approaches:**
{agent_results_text}

Please analyze all approaches, identify the best insights from each, resolve any contradictions, and provide a unified, comprehensive solution."""
            
            # 创建综合阶段的进度处理器
            synthesis_state = {"thinking_started": False, "last_verification": None}
            
            def synthesis_progress_handler(event: ProgressEvent):
                if event.type == "thinking":
                    # 只在第一次 thinking 事件时发送消息
                    if not synthesis_state["thinking_started"]:
                        synthesis_state["thinking_started"] = True
                        self._emit("progress", {"message": "Deep thinking on synthesis..."})
                elif event.type == "verification":
                    passed = event.data.get("passed", False)
                    # 只在验证状态变化时发送消息
                    if synthesis_state["last_verification"] != passed:
                        synthesis_state["last_verification"] = passed
                        if passed:
                            self._emit("progress", {"message": "Synthesis verified"})
                        else:
                            self._emit("progress", {"message": "Refining synthesis..."})
            
            # 使用 DeepThink 引擎进行综合
            synthesis_model = self._get_model_for_stage("synthesis")
            synthesis_engine = DeepThinkEngine(
                client=self.client,
                model=synthesis_model,
                problem_statement=synthesis_problem,
                conversation_history=self.conversation_history,  # 传递结构化的消息历史
                knowledge_context=self.knowledge_context,
                max_iterations=self.max_iterations,
                required_successful_verifications=self.required_verifications,
                max_errors_before_give_up=self.max_errors,
                model_stages=self.model_stages,
                on_progress=synthesis_progress_handler,
                enable_parallel_check=self.enable_parallel_check,
                llm_params=self.llm_params,
            )
            
            synthesis_result = await synthesis_engine.run()
            synthesis = synthesis_result.summary  # 使用 DeepThink 的摘要作为综合结果
            
            # 收集 synthesis 阶段的搜索来源
            if synthesis_result.sources:
                self.sources.extend(synthesis_result.sources)
            
            # 生成最终摘要
            self._emit("summarizing", {"message": "Creating final summary for user..."})
            
            summary_model = self._get_model_for_stage("summary")
            # 使用文本版本构建摘要提示词
            summary_prompt = build_final_summary_prompt(
                self.problem_statement_text,
                synthesis
            )
            
            final_summary = await self.client.generate_text(
                model=summary_model,
                prompt=summary_prompt,
                **self.llm_params
            )
            
            # 获取统计信息
            stats = self.client.get_statistics()
            
            self._emit("success", {
                "solution": final_summary, 
                "iterations": 1,
                "statistics": stats
            })
            
            return UltraThinkResult(
                mode="ultra-think",
                plan=plan,
                agent_results=agent_results,
                synthesis=synthesis,
                final_solution=synthesis,
                summary=final_summary,
                total_agents=num_agents,
                completed_agents=len([r for r in agent_results if r.status == "completed"]),
                sources=self.sources if self.sources else None,
                knowledge_enhanced=len(self.sources) > 0,
            )
        
        except asyncio.CancelledError:
            # 引擎被取消，记录日志并退出
            logger.info("UltraThink engine cancelled by client disconnect")
            raise

