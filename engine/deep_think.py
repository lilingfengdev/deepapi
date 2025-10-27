"""
Deep Think 引擎
单 Agent 深度推理引擎
"""
import time
from typing import Optional, Callable, List, Dict, Any, Union
import asyncio

from models import (
    DeepThinkResult,
    DeepThinkIteration,
    Verification,
    Source,
    ProgressEvent,
    MessageContent,
    extract_text_from_content
)
from utils.openai_client import OpenAIClient
from engine.prompts import (
    DEEP_THINK_INITIAL_PROMPT,
    SELF_IMPROVEMENT_PROMPT,
    VERIFICATION_SYSTEM_PROMPT,
    CORRECTION_PROMPT,
    EXTRACT_DETAILED_SOLUTION_MARKER,
    build_verification_prompt,
    build_initial_thinking_prompt,
    build_final_summary_prompt,
    build_thinking_plan_prompt,
)


class DeepThinkEngine:
    """Deep Think 引擎 - 单 Agent 深度推理"""
    
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
        model_stages: Dict[str, str] = None,
        on_progress: Optional[Callable[[ProgressEvent], None]] = None,
        enable_planning: bool = False,
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
        self.model_stages = model_stages or {}
        self.on_progress = on_progress
        self.sources: List[Source] = []
        self.enable_planning = enable_planning
        self.enable_parallel_check = enable_parallel_check
        self.llm_params = llm_params or {}
        self._task = None  # 用于存储当前任务，以便取消
    
    def _get_model_for_stage(self, stage: str) -> str:
        """获取特定阶段的模型"""
        return self.model_stages.get(stage, self.model)
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """发送进度事件"""
        if self.on_progress:
            self.on_progress(ProgressEvent(type=event_type, data=data))
    
    def _extract_detailed_solution(
        self,
        solution: str,
        marker: str = EXTRACT_DETAILED_SOLUTION_MARKER,
        after: bool = True
    ) -> str:
        """提取详细解决方案"""
        idx = solution.find(marker)
        if idx == -1:
            return "" if after else solution
        if after:
            return solution[idx + len(marker):].strip()
        else:
            return solution[:idx].strip()
    
    async def _generate_thinking_plan(self, problem_statement: MessageContent) -> str:
        """计划阶段 - 生成思考计划"""
        self._emit("progress", {"message": "Generating thinking plan..."})
        
        # 提取文本用于构建提示词
        problem_text = extract_text_from_content(problem_statement)
        prompt = build_thinking_plan_prompt(problem_text)
        
        # 直接使用 prompt 参数传递多模态内容
        plan = await self.client.generate_text(
            model=self.model,
            prompt=problem_statement,  # 保留多模态内容
            **self.llm_params
        )
        
        self._emit("planning", {"plan": plan})
        
        return plan
    
    async def _verify_solution(
        self,
        problem_statement: MessageContent,
        solution: str
    ) -> Dict[str, str]:
        """验证解决方案"""
        detailed_solution = self._extract_detailed_solution(solution)
        # 提取文本用于构建提示词
        problem_text = extract_text_from_content(problem_statement)
        verification_prompt = build_verification_prompt(
            problem_text,
            detailed_solution,
            self.conversation_history  # 传入对话历史
        )
        
        self._emit("progress", {"message": "Verifying solution..."})
        
        # 使用验证阶段的模型
        verification_model = self._get_model_for_stage("verification")
        
        # 获取验证结果
        verification_output = await self.client.generate_text(
            model=verification_model,
            system=VERIFICATION_SYSTEM_PROMPT,
            prompt=verification_prompt,
            **self.llm_params
        )
        
        # 检查验证是否通过
        check_prompt = (
            f'Response in "yes" or "no". Is the following statement saying the '
            f'solution is correct, or does not contain critical error or a major '
            f'justification gap?\n\n{verification_output}'
        )
        
        good_verify = await self.client.generate_text(
            model=verification_model,
            prompt=check_prompt,
            **self.llm_params
        )
        
        bug_report = ""
        if "yes" not in good_verify.lower():
            bug_report = self._extract_detailed_solution(
                verification_output,
                "Detailed Review",
                False
            )
        
        return {"bug_report": bug_report, "good_verify": good_verify}
    
    async def _verify_solution_parallel(
        self,
        problem_statement: MessageContent,
        solution: str
    ) -> Dict[str, str]:
        """并行验证解决方案 - 同时启动required_verifications个验证LLM调用，全部通过才算成功"""
        detailed_solution = self._extract_detailed_solution(solution)
        # 提取文本用于构建提示词
        problem_text = extract_text_from_content(problem_statement)
        verification_prompt = build_verification_prompt(
            problem_text,
            detailed_solution,
            self.conversation_history  # 传入对话历史
        )
        
        num_checks = self.required_verifications
        self._emit("progress", {"message": f"Parallel verifying solution ({num_checks} concurrent checks)..."})
        
        # 使用验证阶段的模型
        verification_model = self._get_model_for_stage("verification")
        
        # 同时启动required_verifications个验证LLM调用
        verification_tasks = [
            self.client.generate_text(
                model=verification_model,
                system=VERIFICATION_SYSTEM_PROMPT,
                prompt=verification_prompt,
                **self.llm_params
            )
            for _ in range(num_checks)
        ]
        
        # 等待所有验证完成
        verification_outputs = await asyncio.gather(*verification_tasks)
        
        # 检查每个验证结果
        check_tasks = []
        for verification_output in verification_outputs:
            check_prompt = (
                f'Response in "yes" or "no". Is the following statement saying the '
                f'solution is correct, or does not contain critical error or a major '
                f'justification gap?\n\n{verification_output}'
            )
            check_tasks.append(
                self.client.generate_text(
                    model=verification_model,
                    prompt=check_prompt,
                    **self.llm_params
                )
            )
        
        # 等待所有检查完成
        good_verifies = await asyncio.gather(*check_tasks)
        
        # 统计通过的验证数量
        passed_count = sum(1 for gv in good_verifies if "yes" in gv.lower())
        
        # 需要全部通过才算验证成功
        passed = passed_count == num_checks
        
        # 收集 bug 报告（如果有的话）
        bug_reports = []
        if not passed:
            for i, (verification_output, good_verify) in enumerate(zip(verification_outputs, good_verifies)):
                if "yes" not in good_verify.lower():
                    bug_report = self._extract_detailed_solution(
                        verification_output,
                        "Detailed Review",
                        False
                    )
                    if bug_report:
                        bug_reports.append(f"[Check {i+1}] {bug_report}")
        
        combined_bug_report = "\n\n".join(bug_reports) if bug_reports else ""
        
        # 返回综合结果
        good_verify_summary = f"yes (passed {passed_count}/{num_checks} checks)" if passed else f"no (passed {passed_count}/{num_checks} checks)"
        
        return {
            "bug_report": combined_bug_report,
            "good_verify": good_verify_summary,
            "parallel_results": {
                "total_checks": num_checks,
                "passed_checks": passed_count,
                "individual_results": good_verifies
            }
        }
    
    async def _initial_exploration(
        self,
        problem_statement: MessageContent,
        other_prompts: List[str]
    ) -> Dict[str, Any]:
        """初始探索阶段"""
        self._emit("thinking", {"iteration": 0, "phase": "initial-exploration"})
        
        # 使用初始阶段的模型
        initial_model = self._get_model_for_stage("initial")
        
        # 构建系统提示词，包含知识库
        system_prompt = DEEP_THINK_INITIAL_PROMPT
        
        if self.knowledge_context:
            system_prompt += (
                "\n\n### Available Knowledge Base ###\n\n" +
                self.knowledge_context +
                "\n\n### End of Knowledge Base ###\n"
            )
        
        # 添加额外的提示词（用于向后兼容，例如 agent 特定提示词）
        if self.other_prompts:
            system_prompt += "\n\n### Additional Context ###\n\n"
            system_prompt += "\n\n".join(self.other_prompts)
        
        # 构建消息列表：历史消息 + 当前问题
        messages = []
        
        # 添加历史对话消息
        if self.conversation_history:
            messages.extend(self.conversation_history)
        
        # 添加当前用户消息
        messages.append({"role": "user", "content": problem_statement})
        
        # 第一次思考 - 使用完整的消息历史
        first_solution = await self.client.generate_text(
            model=initial_model,
            system=system_prompt,
            messages=messages,
            **self.llm_params
        )
        
        self._emit("solution", {"solution": first_solution, "iteration": 0})
        
        # 自我改进
        self._emit("thinking", {"iteration": 0, "phase": "self-improvement"})
        
        improvement_model = self._get_model_for_stage("improvement")
        
        # 构建系统提示词，包含知识库
        system_prompt = DEEP_THINK_INITIAL_PROMPT
        if self.knowledge_context:
            system_prompt += (
                "\n\n### Available Knowledge Base ###\n\n" +
                self.knowledge_context +
                "\n\n### End of Knowledge Base ###\n"
            )
        
        # 添加额外的提示词（用于向后兼容）
        if self.other_prompts:
            system_prompt += "\n\n### Additional Context ###\n\n"
            system_prompt += "\n\n".join(self.other_prompts)
        
        # 构建消息列表：历史 + 当前问题 + 第一次回答 + 改进请求
        improvement_messages = []
        if self.conversation_history:
            improvement_messages.extend(self.conversation_history)
        improvement_messages.extend([
            {"role": "user", "content": problem_statement},
            {"role": "assistant", "content": first_solution},
            {"role": "user", "content": SELF_IMPROVEMENT_PROMPT},
        ])
        
        improved_solution = await self.client.generate_text(
            model=improvement_model,
            system=system_prompt,
            messages=improvement_messages,
            **self.llm_params
        )
        
        self._emit("solution", {"solution": improved_solution, "iteration": 0})
        
        # 验证 - 根据配置选择串行或并行
        if self.enable_parallel_check:
            verification = await self._verify_solution_parallel(
                problem_statement,
                improved_solution
            )
        else:
            verification = await self._verify_solution(
                problem_statement,
                improved_solution
            )
        
        self._emit("verification", {
            "passed": "yes" in verification["good_verify"].lower(),
            "iteration": 0,
        })
        
        return {"solution": improved_solution, "verification": verification}
    
    async def run(self) -> DeepThinkResult:
        """运行 Deep Think 引擎"""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # 发送事件时使用文本版本
            self._emit("init", {"problem": self.problem_statement_text})
            
            plan = None
            
            # Planning 阶段 (如果启用)
            if self.enable_planning:
                # 传递多模态内容给计划生成
                plan = await self._generate_thinking_plan(self.problem_statement)
                # 把计划加入 other_prompts
                if plan:
                    self.other_prompts.append(f"\n### Thinking Plan ###\n{plan}\n")
            
            # 初始探索 - 传递多模态内容
            initial = await self._initial_exploration(
                self.problem_statement,
                self.other_prompts
            )
            
            solution = initial["solution"]
            verification = initial["verification"]
            
            iterations: List[DeepThinkIteration] = []
            verifications: List[Verification] = []
            
            error_count = 0
            correct_count = 1 if "yes" in verification["good_verify"].lower() else 0
            
            # 主循环
            for i in range(self.max_iterations):
                # 检查是否被取消
                await asyncio.sleep(0)  # 给取消信号一个检查点
                
                passed = "yes" in verification["good_verify"].lower()
                
                verifications.append(Verification(
                    timestamp=int(time.time()),
                    passed=passed,
                    bug_report=verification["bug_report"],
                    good_verify=verification["good_verify"],
                ))
                
                iterations.append(DeepThinkIteration(
                    iteration=i,
                    solution=solution,
                    verification=verifications[-1],
                    status="completed" if passed else "correcting",
                ))
                
                if not passed:
                    correct_count = 0
                    error_count += 1
                    
                    if error_count >= self.max_errors:
                        self._emit("failure", {"reason": "Too many errors"})
                        break
                    
                    # 修正
                    self._emit("correction", {"iteration": i})
                    
                    correction_model = self._get_model_for_stage("correction")
                    
                    # 构建系统提示词，包含知识库
                    system_prompt = DEEP_THINK_INITIAL_PROMPT
                    if self.knowledge_context:
                        system_prompt += (
                            "\n\n### Available Knowledge Base ###\n\n" +
                            self.knowledge_context +
                            "\n\n### End of Knowledge Base ###\n"
                        )
                    
                    # 添加额外的提示词（用于向后兼容）
                    if self.other_prompts:
                        system_prompt += "\n\n### Additional Context ###\n\n"
                        system_prompt += "\n\n".join(self.other_prompts)
                    
                    # 构建消息列表：历史 + 当前问题 + 之前的解答 + 修正请求
                    correction_messages = []
                    if self.conversation_history:
                        correction_messages.extend(self.conversation_history)
                    correction_messages.extend([
                        {"role": "user", "content": self.problem_statement},
                        {"role": "assistant", "content": solution},
                        {"role": "user", "content": CORRECTION_PROMPT + "\n\n" + verification["bug_report"]},
                    ])
                    
                    solution = await self.client.generate_text(
                        model=correction_model,
                        system=system_prompt,
                        messages=correction_messages,
                        **self.llm_params
                    )
                    
                    self._emit("solution", {"solution": solution, "iteration": i + 1})
                else:
                    correct_count += 1
                    error_count = 0
                
                if correct_count >= self.required_verifications:
                    # 生成最终摘要
                    self._emit("summarizing", {"message": "Generating final summary..."})
                    
                    summary_model = self._get_model_for_stage("summary")
                    # 提取文本用于构建摘要提示词
                    summary_prompt = build_final_summary_prompt(
                        self.problem_statement_text,
                        solution
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
                        "iterations": i + 1,
                        "statistics": stats
                    })
                    
                    return DeepThinkResult(
                        mode="deep-think",
                        plan=plan,
                        initial_thought=initial["solution"],
                        improvements=[],
                        iterations=iterations,
                        verifications=verifications,
                        final_solution=solution,
                        summary=final_summary,
                        total_iterations=i + 1,
                        successful_verifications=correct_count,
                        sources=self.sources if self.sources else None,
                        knowledge_enhanced=len(self.sources) > 0,
                    )
                
                # 再次验证 - 根据配置选择串行或并行
                if self.enable_parallel_check:
                    verification = await self._verify_solution_parallel(self.problem_statement, solution)
                else:
                    verification = await self._verify_solution(self.problem_statement, solution)
                self._emit("verification", {
                    "passed": "yes" in verification["good_verify"].lower(),
                    "iteration": i + 1,
                })
            
            # 失败 - 仍然生成摘要
            self._emit("summarizing", {"message": "Generating final summary..."})
            
            summary_model = self._get_model_for_stage("summary")
            # 提取文本用于构建摘要提示词
            summary_prompt = build_final_summary_prompt(
                self.problem_statement_text,
                solution
            )
            
            final_summary = await self.client.generate_text(
                model=summary_model,
                prompt=summary_prompt,
                **self.llm_params
            )
            
            # 获取统计信息
            stats = self.client.get_statistics()
            
            self._emit("failure", {
                "reason": "Max iterations reached",
                "statistics": stats
            })
            
            return DeepThinkResult(
                mode="deep-think",
                plan=plan,
                initial_thought=initial["solution"],
                improvements=[],
                iterations=iterations,
                verifications=verifications,
                final_solution=solution,
                summary=final_summary,
                total_iterations=self.max_iterations,
                successful_verifications=correct_count,
                sources=self.sources if self.sources else None,
                knowledge_enhanced=len(self.sources) > 0,
            )
        
        except asyncio.CancelledError:
            # 引擎被取消，记录日志并退出
            logger.info("DeepThink engine cancelled by client disconnect")
            raise

