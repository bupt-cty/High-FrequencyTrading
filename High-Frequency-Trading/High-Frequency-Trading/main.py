import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from backtestengine import BacktestEngine
from TickMomentumStrategy import TickMomentumStrategy

def run_backtest_scenario(data_path, scenario_name, is_ideal_matching=False):
    """
    运行单个回测场景，返回策略对象以供提取评估指标
    """
    print(f"\n🚀 正在运行场景: {scenario_name}...")
    
    # 1. 初始化高保真引擎 (配置螺纹钢 rb 的真实费率与最小变动价位)
    # 假设：开仓万分之1，平今免费(0)，平昨万分之1，最小变动价位1元/吨
    engine = BacktestEngine(
        data_path=data_path, 
        commission_open=0.0001, 
        commission_today=0.0000, 
        commission_yesterday=0.0001, 
        price_tick=1.0
    )
    
    # 2. 理想环境下的“降维打击” (核心控制变量法)
    if is_ideal_matching:
        # 强行覆盖真实盘口穿透撮合，模拟传统回测的“无摩擦最新价成交”
        def ideal_match(direction, volume):
            price = engine.current_tick['last_price']
            # 理想环境下依然象征性收取单边开仓手续费，但不计算滑点和点差
            cost = price * volume * engine.commission_open 
            return True, {'price': price, 'volume': volume, 'cost': cost}
            
        engine.match_order = ideal_match 
    
    # 3.. 绑定策略时，把环境标识传进去，并稍微调低预期跳数以激活交易
    strategy = TickMomentumStrategy(
        engine=engine, 
        momentum_window=60,      # 观察过去 60 个 Tick 的动量
        obi_threshold=0.3,       # 稍微降低盘口失衡的触发门槛
        expected_profit_ticks=8, # 期望赚取 8 跳的趋势利润
        is_ideal=is_ideal_matching
    )
    
    # 4. 运行主循环 (事件驱动的脉搏)
    df = pd.read_csv(data_path)
    
    for index, row in df.iterrows():
        tick = row.to_dict()
        engine.current_tick = tick
        # 直接将 Tick 推送给策略大脑
        strategy.on_tick(tick)
    strategy.force_close_at_end()   
    # 5. 打印毕设所需的核心绩效指标
    print(f"[{scenario_name}] 运行完毕!")
    print(f" ➔ 总交易次数: {strategy.trade_count}")
    print(f" ➔ 最终总盈亏: {strategy.total_pnl:.2f}")
    print(f" ➔ 最大回撤:   {strategy.max_drawdown:.2f}")
    
    return strategy

def main():
    data_path = "rb2505_tick_cleaned.csv" # 请确保路径正确
    
    # --- A/B 测试执行 ---
    # 对照组：理想环境 (按最新价成交，无买卖价差，无穿透滑点)
    ideal_strategy = run_backtest_scenario(
        data_path, 
        scenario_name="理想环境 (最新价无摩擦撮合)", 
        is_ideal_matching=True
    )
    
    # 实验组：真实环境 (盘口对价撮合，带订单穿透滑点与高频手续费)
    realistic_strategy = run_backtest_scenario(
        data_path, 
        scenario_name="真实环境 (盘口穿透与精准费率)", 
        is_ideal_matching=False
    )
    
    # --- 论文级数据可视化 ---
    plt.figure(figsize=(12, 6))
    
    plt.plot(ideal_strategy.pnl_history, label='Ideal Environment (LastPrice, No Slippage)', color='#1f77b4', alpha=0.8, linewidth=1.5)
    plt.plot(realistic_strategy.pnl_history, label='Realistic Environment (L1 OrderBook Match)', color='#d62728', alpha=0.9, linewidth=1.5)
    
    plt.title('Micro-Momentum Strategy: Ideal vs Realistic Execution', fontsize=14, fontweight='bold')
    plt.xlabel('Tick Timeline', fontsize=12)
    plt.ylabel('Cumulative PnL', fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1) 
    
    # 添加文字标注：直接在图表上显示最大回撤，增加学术感
    plt.text(0.02, 0.05, f"Realistic Max Drawdown: {realistic_strategy.max_drawdown:.2f}", 
             transform=plt.gca().transAxes, fontsize=11, color='#d62728',
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
             
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('final_equity_curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    main()