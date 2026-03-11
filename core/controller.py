import time
import logging
from typing import Dict, Any, Optional, List

class SearchController:
    """
    实验控制器：管理预算、监测停止规则并驱动搜索策略切换。
    使 Agent 从“无限试错”转变为“预算受限的科研决策器”。
    """
    def __init__(
        self,
        max_runs: int = 50,
        max_wall_clock_seconds: float = 3600.0,
        no_improvement_limit: int = 10,
        failure_limit: int = 3,
        improvement_threshold: float = 1e-4,
        logger: Optional[logging.Logger] = None
    ):
        self.max_runs = max_runs
        self.max_wall_clock_seconds = max_wall_clock_seconds
        self.no_improvement_limit = no_improvement_limit
        self.failure_limit = failure_limit
        self.improvement_threshold = improvement_threshold
        self.logger = logger or logging.getLogger("SearchController")
        
        # Callbacks for strategy management (to be fully implemented in Phase 2)
        self.on_strategy_switch = None
        self.on_space_reduction = None

        # 状态追踪
        self.start_time = time.time()
        self.total_runs = 0
        self.consecutive_no_improvement = 0
        self.consecutive_failures = 0
        self.best_energy = float('inf')
        self.best_num_params = float('inf')
        self.is_stopped = False
        self.stop_reason = ""

    def check_budget(self) -> bool:
        """检查是否超出预算"""
        elapsed = time.time() - self.start_time
        if self.total_runs >= self.max_runs:
            self.stop("Reached maximum number of runs budget.")
            return False
        if elapsed >= self.max_wall_clock_seconds:
            self.stop(f"Reached maximum wall-clock time budget ({elapsed:.1f}s).")
            return False
        return True

    def report_result(self, results: Dict[str, Any], is_failure: bool = False):
        """记录一轮实验结果并更新状态"""
        self.total_runs += 1
        
        if is_failure:
            self.consecutive_failures += 1
            self.logger.warning(f"Run {self.total_runs} failed. Consecutive failures: {self.consecutive_failures}")
            if self.consecutive_failures >= self.failure_limit:
                self.handle_persistent_failure()
            return

        self.consecutive_failures = 0
        energy = results.get("val_energy", float('inf'))
        num_params = results.get("num_params", float('inf'))

        # Pareto 占优检查 (奥卡姆剃刀)
        improved = False
        if energy < self.best_energy - self.improvement_threshold:
            improved = True
        elif abs(energy - self.best_energy) < self.improvement_threshold and num_params < self.best_num_params:
            improved = True

        if improved:
            self.logger.info(f"Improvement found! Energy: {energy:.6f}, Params: {num_params}")
            self.best_energy = energy
            self.best_num_params = num_params
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1
            self.logger.info(f"No significant improvement. Consecutive: {self.consecutive_no_improvement}")
            
            if self.consecutive_no_improvement >= self.no_improvement_limit:
                self.handle_no_improvement()

    def handle_persistent_failure(self):
        """处理持续失败的情况"""
        self.logger.error(
            f"!!! PERSISTENT FAILURE !!! Detected {self.consecutive_failures} consecutive failures. "
            f"Total runs: {self.total_runs}. Elapsed: {self.elapsed_time:.1f}s. "
            f"Triggering search space reduction."
        )
        if self.on_space_reduction:
            self.on_space_reduction(self)
        self.consecutive_failures = 0 # 重置以允许新策略尝试

    def handle_no_improvement(self):
        """处理长时间无进展的情况"""
        self.logger.warning(
            f"--- NO IMPROVEMENT --- No progress for {self.no_improvement_limit} rounds. "
            f"Best Energy so far: {self.best_energy:.6f}. Total runs: {self.total_runs}. "
            f"Triggering strategy switch."
        )
        if self.on_strategy_switch:
            self.on_strategy_switch(self)
        self.consecutive_no_improvement = 0 # 重置以允许新策略观察

    def stop(self, reason: str):
        self.is_stopped = True
        self.stop_reason = reason
        self.logger.info(f"STOPPING: {reason}")

    def should_continue(self) -> bool:
        if self.is_stopped:
            return False
        return self.check_budget()

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

class SearchOrchestrator:
    """
    策略编排器：管理多个 SearchStrategy 的执行。
    支持按需切换策略（如 GA -> Grid）。
    """
    def __init__(
        self,
        strategies: List[Any],
        controller: Optional[SearchController] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.strategies = strategies
        self.logger = logger or logging.getLogger("SearchOrchestrator")
        self.controller = controller or SearchController(logger=self.logger)
        
        # 绑定控制器的回调
        self.controller.on_strategy_switch = self._handle_strategy_switch
        self.current_strategy_index = 0
        self.skip_current_strategy = False

    def _handle_strategy_switch(self, controller: SearchController):
        """当控制器触发策略切换信号时调用"""
        self.logger.info("Orchestrator received strategy switch signal.")
        self.skip_current_strategy = True

    def run(self) -> List[Dict[str, Any]]:
        """按顺序运行所有策略"""
        all_results = []
        
        for i, strategy in enumerate(self.strategies):
            self.current_strategy_index = i
            self.skip_current_strategy = False
            
            if not self.controller.should_continue():
                self.logger.info("Orchestrator stopped: Global budget reached.")
                break
                
            self.logger.info(f"\n>>>> Starting Strategy {i+1}/{len(self.strategies)}: {strategy.name} <<<<")
            
            # 确保策略使用同一个控制器以共享预算
            strategy.controller = self.controller
            
            # 为了支持 skip 逻辑，我们需要一个包装或在策略内部检查
            # 这里简单处理：如果触发了 skip，策略内部的 should_continue 会返回 False (逻辑需对齐)
            
            # 临时注入针对该策略的 skip 检查
            original_should_continue = self.controller.should_continue
            def orchestrator_should_continue():
                if self.skip_current_strategy:
                    self.logger.warning(f"Skipping strategy {strategy.name} due to orchestrator signal.")
                    return False
                return original_should_continue()
            
            self.controller.should_continue = orchestrator_should_continue
            
            try:
                result = strategy.run()
                all_results.append(result)
            finally:
                # 恢复控制器的原始方法
                self.controller.should_continue = original_should_continue
                
            self.logger.info(f">>>> Strategy {strategy.name} finished. <<<<")
            
        return all_results
