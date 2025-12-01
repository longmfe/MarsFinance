"""
策略开发流水线
提供端到端的策略开发流程
"""

from ..core.signal_generator import VolumePriceSignalGenerator
from ..core.risk_manager import RiskManager
from ..core.position_sizer import PositionSizer

class StrategyPipeline:
    """策略开发流水线"""
    
    def __init__(self, config: Dict[str, Any]):
        self.signal_generator = VolumePriceSignalGenerator(config)
        self.risk_manager = RiskManager(config)
        self.position_sizer = PositionSizer(config)
        
    def run_strategy_development(self, data: pd.DataFrame) -> Dict[str, Any]:
        """运行策略开发流程"""
        results = {}
        
        # 1. 生成信号
        results['signals'] = self.signal_generator.generate_signals(data)
        
        # 2. 风险管理
        results['risk_assessment'] = self.risk_manager.assess_risk(data, results['signals'])
        
        # 3. 仓位管理
        results['position_sizes'] = self.position_sizer.calculate_positions(
            data, results['signals'], results['risk_assessment']
        )
        
        return results
