import pytest
import time
from unittest.mock import MagicMock
from core.orchestration.controller import SearchController, SearchOrchestrator

def test_controller_max_runs():
    controller = SearchController(max_runs=2)
    assert controller.should_continue() is True
    
    controller.report_result({"val_energy": -1.0, "num_params": 10})
    assert controller.should_continue() is True
    
    controller.report_result({"val_energy": -1.1, "num_params": 10})
    assert controller.should_continue() is False
    assert "Reached maximum number of runs budget" in controller.stop_reason

def test_controller_improvement_tracking():
    controller = SearchController(no_improvement_limit=2, improvement_threshold=0.1)
    
    # 1. First result
    controller.report_result({"val_energy": -1.0, "num_params": 10})
    assert controller.consecutive_no_improvement == 0
    
    # 2. No significant improvement
    controller.report_result({"val_energy": -1.05, "num_params": 10}) # Diff < 0.1
    assert controller.consecutive_no_improvement == 1
    
    # 3. Trigger strategy switch signal (simulated)
    mock_callback = MagicMock()
    controller.on_strategy_switch = mock_callback
    controller.report_result({"val_energy": -1.08, "num_params": 10})
    assert controller.consecutive_no_improvement == 0 # Reset after callback
    mock_callback.assert_called_once()

def test_controller_failure_limit():
    controller = SearchController(failure_limit=2)
    mock_callback = MagicMock()
    controller.on_space_reduction = mock_callback
    
    controller.report_result({}, is_failure=True)
    assert controller.consecutive_failures == 1
    
    controller.report_result({}, is_failure=True)
    mock_callback.assert_called_once()
    assert controller.consecutive_failures == 0 # Reset after callback

def test_orchestrator_switching():
    mock_strategy1 = MagicMock()
    mock_strategy1.name = "Strategy1"
    mock_strategy2 = MagicMock()
    mock_strategy2.name = "Strategy2"
    
    # Simulate strategy1 signaling finished after one step
    controller = SearchController(max_runs=10)
    orchestrator = SearchOrchestrator(generators=[mock_strategy1, mock_strategy2], controller=controller)
    
    orchestrator.run()
    
    mock_strategy1.run.assert_called_once()
    mock_strategy2.run.assert_called_once()

def test_orchestrator_stop_propagation():
    mock_strategy1 = MagicMock()
    mock_strategy1.name = "Strategy1"
    mock_strategy2 = MagicMock()
    mock_strategy2.name = "Strategy2"
    
    controller = SearchController(max_runs=1)
    orchestrator = SearchOrchestrator(generators=[mock_strategy1, mock_strategy2], controller=controller)
    
    # After 1st strategy, budget is reached
    mock_strategy1.run.side_effect = lambda: controller.report_result({"val_energy": -1.0, "num_params": 10})
    
    orchestrator.run()
    
    mock_strategy1.run.assert_called_once()
    mock_strategy2.run.assert_not_called()
