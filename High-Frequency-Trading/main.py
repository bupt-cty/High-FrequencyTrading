import matplotlib.pyplot as plt
import pandas as pd
# 假设你已经将之前的类分别保存在 engine.py 和 strategy.py 中
from backtestengine import BacktestEngine
from TickMomentumStrategy import TickMomentumStrategy

def run_backtest_scenario(data_path, scenario_name, base_slippage, is_ideal_matching=False):
    print(f"正在运行场景: {scenario_name}...")
    
    # 1. 初始化引擎
    engine = BacktestEngine(data_path=data_path, commission_rate=0.0001, base_slippage=base_slippage)
    
    # 2. 核心修复：真正实现“理想环境”的无摩擦撮合
    if is_ideal_matching:
        engine.calculate_dynamic_slippage = lambda: 0.0  # 强制滑点为0
        
        # 动态替换撮合逻辑：强行用 LastPrice 成交，忽略盘口容量和价差
        def ideal_match(direction, volume):
            price = engine.current_tick['last_price']
            # 在理想回测中，通常连手续费也忽略不计，或者只算单边极低手续费
            cost = price * volume * engine.commission_rate 
            return True, {'price': price, 'volume': volume, 'cost': cost}
            
        engine.match_order = ideal_match # 覆盖原有的严格撮合方法
    
    # 3. 绑定策略 (如果真实环境下依然不交易，可以尝试调低 obi_threshold 比如到 0.3)
    strategy = TickMomentumStrategy(engine, momentum_window=10, obi_threshold=0.4, expected_profit_ticks=3)
    
    # 4. 运行主循环
    import pandas as pd
    df = pd.read_csv(data_path)
    
    pnl_record = [] # 核心修复：在策略外部记录资金曲线，绝不漏掉任何一个Tick
    
    for index, row in df.iterrows():
        tick = row.to_dict()
        engine.current_tick = tick
        engine.tick_history.append(tick)
        
        if len(engine.tick_history) > engine.volatility_window:
            engine.tick_history.pop(0)
            
        # 触发策略逻辑
        strategy.on_tick(tick)
        
        # 无论策略内部是否 return，我们都在外部强制记录当前总盈亏
        pnl_record.append(strategy.total_pnl) 
        
    print(f"[{scenario_name}] 回测完成! 总交易次数: {strategy.trade_count}, 最终盈亏: {strategy.total_pnl:.2f}")
    return pnl_record

def main():
    data_path = r"E:\code\High-Frequency-Trading\rb2505_tick_sample.csv" # 替换为你的真实路径
    
    # 1. 运行对照组：理想环境（零滑点，假设完全流动性）
    ideal_pnl = run_backtest_scenario(
        data_path, 
        scenario_name="理想环境 (0滑点)", 
        base_slippage=0, 
        is_ideal_matching=True
    )
    
    # 2. 运行实验组：真实环境（包含你的动态滑点和盘口撮合）
    realistic_pnl = run_backtest_scenario(
        data_path, 
        scenario_name="真实环境 (动态滑点+盘口撮合)", 
        base_slippage=1, 
        is_ideal_matching=False
    )
    
    # 3. 数据可视化 (生成论文配图)
    plt.figure(figsize=(12, 6))
    
    # 绘制两条资金曲线
    plt.plot(ideal_pnl, label='Ideal Environment (Zero Slippage)', color='blue', alpha=0.7)
    plt.plot(realistic_pnl, label='Realistic Environment (Dynamic Slippage)', color='red', alpha=0.9)
    
    # 图表美化与学术化格式
    plt.title('HFT Strategy Equity Curve Comparison: Ideal vs Realistic Environment', fontsize=14, fontweight='bold')
    plt.xlabel('Tick Steps', fontsize=12)
    plt.ylabel('Cumulative PnL (Tick Value)', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1) # 盈亏平衡线
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 自动调整布局并显示/保存
    plt.tight_layout()
    plt.savefig('equity_curve_comparison.png', dpi=300) # 保存高清图供论文使用
    plt.show()

if __name__ == "__main__":
    main()