import os
import random
import pandas as pd
import itertools

# 导入你的前置模块 (请确保这些文件在同级目录下)
from backtestengine import BacktestEngine
from TickMomentumStrategy import TickMomentumStrategy
from optimizer import HFTAlgorithmOptimizer

# ==========================================
# 进程级全局缓存：避免多进程重复读取大型 CSV 导致的 I/O 崩溃
# ==========================================
_GLOBAL_TICKS = []

def _load_data_once(data_path: str):
    global _GLOBAL_TICKS
    if not _GLOBAL_TICKS:
        # 调试打印：确认哪个进程在尝试加载
        # print(f"DEBUG: Process {os.getpid()} is loading {data_path}") 
        if not os.path.exists(data_path):
            # 这里的报错会被 UCB 捕获并显示在终端
            raise FileNotFoundError(f"【关键错误】子进程找不到数据文件: {os.path.abspath(data_path)}")
            
        try:
            df = pd.read_csv(data_path)
            if df.empty:
                raise ValueError(f"【关键错误】文件 {data_path} 是空的")
            _GLOBAL_TICKS = df.to_dict('records')
            # print(f"DEBUG: Process {os.getpid()} loaded {len(_GLOBAL_TICKS)} ticks.")
        except Exception as e:
            raise Exception(f"加载数据时发生未知错误: {str(e)}")

# ==========================================
# 黑盒评估函数 (Stochastic Evaluation)
# 注意：此函数必须在顶层作用域，以便多进程能够正确 Pickle 序列化
# ==========================================
def evaluate_strategy_stochastic(params: dict) -> float:
    data_path = "rb2505_tick_cleaned.csv"
    _load_data_once(data_path)
    
    sample_size = params.get('sample_size', 9000) # 保持 2 小时级别的窗口
    total_ticks = len(_GLOBAL_TICKS)
    
    if total_ticks <= sample_size:
        return -9999.0
        
    start_idx = random.randint(0, total_ticks - sample_size)
    sampled_ticks = _GLOBAL_TICKS[start_idx : start_idx + sample_size]
    
    window = params.get('momentum_window', 60)
    threshold = params.get('obi_threshold', 0.4)
    profit_ticks = params.get('expected_profit_ticks', 10)
    
    # 实例化引擎 (对齐你的新版接口)
    engine = BacktestEngine(
        data_path=None, 
        commission_open=0.0001,    
        commission_today=0.0001,   
        commission_yesterday=0.0001, 
        price_tick=1.0             
    )
    
    # 实例化策略
    strategy = TickMomentumStrategy(
        engine=engine, 
        momentum_window=window, 
        obi_threshold=threshold, 
        expected_profit_ticks=profit_ticks,
        is_ideal=False # UCB 必须在最残酷的真实环境中进行优胜劣汰
    )
    
    # 在随机切片上运行
    for tick in sampled_ticks:
        engine.current_tick = tick
        engine.tick_history.append(tick)
        
        if hasattr(engine, 'volatility_window') and len(engine.tick_history) > engine.volatility_window:
            engine.tick_history.pop(0)
            
        strategy.on_tick(tick)
        
    # === 核心新增：切片结束，强制平仓结算盈亏 ===
    strategy.force_close_at_end()
        
    # Reward 计算
    pnl = strategy.total_pnl
    trade_count = strategy.trade_count
    max_drawdown = getattr(strategy, 'max_drawdown', 0.0) 
    
    # 如果 2 小时内交易不到 2 次，给予轻微时间成本惩罚
    if trade_count < 2:
        return -0.5
        
    # 风险调整收益
    if pnl > 0:
        reward = pnl / (max_drawdown + 1e-5)
    else:
        reward = pnl - max_drawdown
        
    return reward

def generate_param_space() -> list:
    """全面升级的参数寻优网格"""
    # 贴合新版策略，放大动量窗口，寻找更稳固的趋势
    windows = [30, 60, 90]
    # OBI 阈值保持中等敏感度
    thresholds = [0.3,0.5, 0.6, 0.7]
    # 预期利润跳数必须远超摩擦成本 (一买一卖摩擦约在3-4跳)
    profit_ticks = [6, 8, 10, 12, 15]
    
    return [
        {'momentum_window': w, 'obi_threshold': t, 'expected_profit_ticks': p, 'sample_size': 9000}
        for w, t, p in itertools.product(windows, thresholds, profit_ticks)
    ]

# ==========================================
# 主程序入口 (必须写在 __main__ 中以兼容 Windows 的多进程)
# ==========================================
if __name__ == "__main__":
    print("1. 正在初始化高维参数空间...")
    param_space = generate_param_space()
    print(f"共生成 {len(param_space)} 种参数组合待探索。")
    
    print("\n2. 启动基于随机抽样的 UCB 算法优化器...")
    optimizer = HFTAlgorithmOptimizer(
        backtest_func=evaluate_strategy_stochastic, 
        param_space=param_space, 
        exploration_constant=2.0  # 增加了探索常数，鼓励算法在早期多去测试不同的参数
    )
    
    # 引入随机抽样后，同一个参数需要多次被选中才能收敛到真实的期望均值
    # 因此 total_iterations 应当远大于 param_space 的长度
    best_params, best_score = optimizer.run_optimization_parallel(
        total_iterations=1000, 
        max_workers=None 
    )
    
    print("\n" + "="*50)
    print("🎯 并行参数寻优圆满完成！")
    print(f"全局最优生存参数组合: {best_params}")
    print(f"该参数在随机环境下的最高期望得分: {best_score:.4f}")
    print("="*50)