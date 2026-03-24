import numpy as np

class TickMomentumStrategy:
    def __init__(self, engine, momentum_window=10, obi_threshold=0.6, expected_profit_ticks=3):
        """
        初始化策略参数
        :param engine: 绑定的回测引擎实例
        :param momentum_window: 计算动量的时间窗口 (Tick数量)
        :param obi_threshold: 订单簿不平衡度触发阈值 (-1 到 1 之间)
        :param expected_profit_ticks: 预期盈利跳数 (用于校验是否能覆盖成本)
        """
        self.engine = engine
        self.momentum_window = momentum_window
        self.obi_threshold = obi_threshold
        self.expected_profit_ticks = expected_profit_ticks
        self.pnl_history = []#每次tick变化后的资金流水
        
        # 状态记录
        self.price_history = []
        self.current_position = 0  # 1 为多头，-1 为空头，0 为空仓
        self.entry_price = 0.0
        
        # 统计数据
        self.trade_count = 0
        self.total_pnl = 0.0

    def calculate_obi(self, tick):
        """计算订单簿不平衡度"""
        bid_vol = tick['bid_volume1']
        ask_vol = tick['ask_volume1']
        if bid_vol + ask_vol == 0:
            return 0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def on_tick(self, tick):
        """
        引擎每推送一个 Tick，都会调用此方法 (策略的大脑)
        """
        self.price_history.append(tick['last_price'])
        if len(self.price_history) > self.momentum_window:
            self.price_history.pop(0)
            
        # 1. 数据积累期，暂不交易
        if len(self.price_history) < self.momentum_window:
            return

        # 2. 计算微观指标
        current_price = tick['last_price']
        momentum = current_price - self.price_history[0] # 简单Tick级动量，动量 = 当前价 - N个Tick前的价格
        obi = self.calculate_obi(tick)
        
        # 3. 成本与生存边界前置校验
        spread = tick['ask_price1'] - tick['bid_price1']
        dynamic_slip = self.engine.calculate_dynamic_slippage()
        
        # 假设1跳的价值为1 (可根据具体品种如 rb 螺纹钢修改)
        estimated_cost = (current_price * self.engine.commission_rate * 2) + spread + (dynamic_slip * 2)
        expected_return = self.expected_profit_ticks * 1.0 # 假设预期赚取3跳
        
        # 如果预期收益连摩擦成本都覆盖不了，直接放弃寻找信号
        if expected_return <= estimated_cost:
            return 

        # 4. 信号生成与执行逻辑
        if self.current_position == 0:
            # 开多仓逻辑：动量向上 且 买盘强劲
            if momentum > 0 and obi > self.obi_threshold:
                success, result = self.engine.match_order('BUY', 1)
                if success:
                    self.current_position = 1
                    self.entry_price = result['price']
                    self.trade_count += 1
                    
            # 开空仓逻辑：动量向下 且 卖盘强劲
            elif momentum < 0 and obi < -self.obi_threshold:
                success, result = self.engine.match_order('SELL', 1)
                if success:
                    self.current_position = -1
                    self.entry_price = result['price']
                    self.trade_count += 1

        # 5. 平仓逻辑 (极其简化版：达到预期跳数止盈，或反向动量止损)
        elif self.current_position == 1:
            if current_price >= self.entry_price + self.expected_profit_ticks or momentum < 0:
                success, result = self.engine.match_order('SELL', 1)
                if success:
                    pnl = result['price'] - self.entry_price - result['cost'] # 粗略计算单笔盈亏
                    self.total_pnl += pnl
                    self.current_position = 0
                    
        elif self.current_position == -1:
            if current_price <= self.entry_price - self.expected_profit_ticks or momentum > 0:
                success, result = self.engine.match_order('BUY', 1)
                if success:
                    pnl = self.entry_price - result['price'] - result['cost']
                    self.total_pnl += pnl
                    self.current_position = 0
        self.pnl_history.append(self.total_pnl)#显示tick变化后的资金流水
        
        
    @property
    def max_drawdown(self):
        """利用资金流水计算最大回撤"""
        if not self.pnl_history:
            return 0.0
        # 使用 numpy 向量化计算，提升高频回测效率
        pnl_array = np.array(self.pnl_history)
        running_max = np.maximum.accumulate(pnl_array)
        drawdowns = running_max - pnl_array
        return np.max(drawdowns)