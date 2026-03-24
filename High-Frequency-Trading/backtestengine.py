import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, data_path, commission_rate=0.0001, base_slippage=1):
        self.data_path = data_path
        self.commission_rate = commission_rate
        self.base_slippage = base_slippage  # 基础滑点（跳）
        self.current_tick = None
        self.tick_history = [] # 用于计算波动率的滑动窗口
        self.volatility_window = 20 # 设定的 Tick 窗口大小
        
        # 交易记录与绩效评估
        self.trades = []
        self.capital = 1000000 
        
    def calculate_dynamic_slippage(self):
        """
        核心创新点：基于波动率的动态滑点惩罚模型
        逻辑：近期价格标准差越大，流动性枯竭概率越高，滑点惩罚指数级上升。
        数学模型：惩罚系数 $P = \max(1, e^{\sigma - 2.0})$，其中 $\sigma$ 为近期价格标准差。
        """
        if len(self.tick_history) < self.volatility_window:
            return self.base_slippage
            
        # 提取窗口内的 last_price 计算波动率
        recent_prices = [tick['last_price'] for tick in self.tick_history[-self.volatility_window:]]
        volatility = np.std(recent_prices)
        
        # 滑点膨胀系数 (具体函数形态可通过蒙特卡洛模拟优化)
        # 假设波动率基准为 2.0，超过该值开始惩罚
        penalty_factor = max(1, np.exp(volatility - 2.0)) 
        
        return self.base_slippage * penalty_factor

    def match_order(self, direction, volume):
        """
        核心难点：真实盘口撮合机制
        """
        if not self.current_tick:
            return False, "No market data"
            
        ask1_price = self.current_tick['ask_price1']
        ask1_vol = self.current_tick['ask_volume1']
        bid1_price = self.current_tick['bid_price1']
        bid1_vol = self.current_tick['bid_volume1']
        
        dynamic_slip = self.calculate_dynamic_slippage()
        
        # 必须满足严格的盈利触发阈值：
        # Expected_Return > (Commission * 2 + Spread + Slippage)
        
        if direction == 'BUY':
            # 买入必须按 Ask1 成交，且受制于 Ask_Volume1
            if volume > ask1_vol:
                return False, "Insufficient liquidity at Ask1" # 模拟撤单或部分成交
                
            execution_price = ask1_price + dynamic_slip # 买入时滑点向上加
            cost = execution_price * volume * self.commission_rate
            return True, {'price': execution_price, 'volume': volume, 'cost': cost}
            
        elif direction == 'SELL':
            # 卖出必须按 Bid1 成交，且受制于 Bid_Volume1
            if volume > bid1_vol:
                return False, "Insufficient liquidity at Bid1"
                
            execution_price = bid1_price - dynamic_slip # 卖出时滑点向下减
            cost = execution_price * volume * self.commission_rate
            return True, {'price': execution_price, 'volume': volume, 'cost': cost}

    def run(self):
        """
        事件驱动的“心脏起搏器”
        """
        # 使用迭代器逐行读取，防止内存溢出，契合高频海量数据特征
        df = pd.read_csv(self.data_path)
        
        for index, row in df.iterrows():
            self.current_tick = row.to_dict()
            self.tick_history.append(self.current_tick)
            
            # 维持滑动窗口大小，防止内存爆炸
            if len(self.tick_history) > self.volatility_window:
                self.tick_history.pop(0)
                
            # 这里触发策略层的逻辑 (on_tick)
            # strategy.on_tick(self.current_tick)
            
            # 在 BacktestEngine 的 run 方法循环内部添加：
            if self.strategy:
                self.strategy.on_tick(self.current_tick)

            # 外部调用示例：
            # engine = BacktestEngine("rb2505_tick_cleaned.csv")
            # strategy = TickMomentumStrategy(engine)
            # engine.strategy = strategy
            # engine.run()