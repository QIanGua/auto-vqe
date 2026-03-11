import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.controller import SearchController

class SearchStrategy(ABC):
    """
    量子电路搜索策略基类。
    所有具体的搜索算法（GA, Grid, ADAPT 等）都应继承此类。
    """
    def __init__(
        self,
        env: Any,
        controller: Optional[SearchController] = None,
        logger: Optional[logging.Logger] = None,
        name: str = "BaseStrategy"
    ):
        self.env = env
        self.logger = logger or logging.getLogger(name)
        self.controller = controller or SearchController(logger=self.logger)
        self.name = name

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        运行搜索任务，返回包含最优结果的字典。
        返回格式应包含: {"best_config": ..., "best_results": ..., "ansatz_spec": ...}
        """
        pass
