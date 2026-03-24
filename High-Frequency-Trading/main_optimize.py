import pandas as pd
import itertools
from backtestengine import BacktestEngine
from TickMomentumStrategy import TickMomentumStrategy
from optimizer import HFTAlgorithmOptimizer

def evaluate_strategy(params: dict, data_path: str = "rb2505_tick_cleaned.csv") -> float:
    """仿真层的对外黑盒接口 (Objective Function)"""
    # 提取当前测试的参数
    window = params.get('momentum_window', 10)
    threshold = params.get('obi_threshold', 0.5)
    profit_ticks = params.get('expected_profit_ticks', 3)
    
    # 初始化底层引擎与策略
    engine = BacktestEngine(data_path=data_path, commission_rate=0.0001, base_slippage=1)
    strategy = TickMomentumStrategy(
        engine=engine, 
        momentum_window=window, 
        obi_threshold=threshold, 
        expected_profit_ticks=profit_ticks
    )
    
    # 注入时序数据
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: 找不到数据文件 {data_path}，请检查路径。")
        return -9999.0

    for index, row in df.iterrows():
        tick = row.to_dict()
        engine.current_tick = tick
        engine.tick_history.append(tick)
        
        # 假设 engine 有 volatility_window 属性，若没有请根据实际 engine 代码调整
        if hasattr(engine, 'volatility_window') and len(engine.tick_history) > engine.volatility_window:
            engine.tick_history.pop(0)
            
        strategy.on_tick(tick)
        
    # ==========================================
    # Reward Shaping (结合了你新增的最大回撤属性)
    # ==========================================
    pnl = strategy.total_pnl
    trade_count = strategy.trade_count
    max_drawdown = strategy.max_drawdown 
    
    # 防御 1：拒绝低频偶然性
    if trade_count < 3: #if trade_count < 10
        return -1000.0
        
    # 防御 2：计算风险调整后收益
    penalty = trade_count * 0.5  # 隐形成本惩罚
    
    # 核心公式：(净利润 - 惩罚) / (最大回撤 + 极小值防止除以0)
    reward = (pnl - penalty) / (max_drawdown + 1e-5)
    
    return reward

def generate_param_space() -> list:
    windows = [5, 10, 20, 30]
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    profit_ticks = [2, 3, 4, 5]
    return [
        {'momentum_window': w, 'obi_threshold': t, 'expected_profit_ticks': p}
        for w, t, p in itertools.product(windows, thresholds, profit_ticks)
    ]

if __name__ == "__main__":
    # 多进程安全保护块
    print("1. 正在初始化参数空间...")
    param_space = generate_param_space()
    print(f"共生成 {len(param_space)} 种参数组合待探索。")
    
    print("\n2. 初始化 UCB 算法优化器...")
    optimizer = HFTAlgorithmOptimizer(
        backtest_func=evaluate_strategy, 
        param_space=param_space, 
        exploration_constant=1.5 
    )
    
    # 核心改动：调用并行方法
    best_params, best_score = optimizer.run_optimization_parallel(
        total_iterations=150, 
        max_workers=None  # 自动检测 CPU 核心数，例如 8核机器会同时跑 7 个回测
    )
    
    print("\n" + "="*40)
    print("并行优化完成！")
    print(f"最优参数组合: {best_params}")
    print(f"最高风险调整得分 (Reward): {best_score:.4f}")
    print("="*40)