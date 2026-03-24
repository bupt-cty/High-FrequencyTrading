import numpy as np

class TickMomentumStrategy:
    def __init__(self, engine, momentum_window=90, obi_threshold=0.7, expected_profit_ticks=8, is_ideal=False, is_mean_reversion=True):
        """
        初始化策略
        :param engine: 回测引擎实例
        :param momentum_window: 计算动量的时间窗口 (Tick数量)
        :param obi_threshold: 订单簿不平衡度触发阈值 (0 到 1)
        :param expected_profit_ticks: 预期盈利跳数
        :param is_ideal: 是否为理想无摩擦环境 (用于论文对照实验)
        :param is_mean_reversion: 是否开启微观均值反转模式 (高频实盘核心)
        """
        self.engine = engine
        self.momentum_window = momentum_window
        self.obi_threshold = obi_threshold
        self.expected_profit_ticks = expected_profit_ticks
        self.is_ideal = is_ideal
        self.is_mean_reversion = is_mean_reversion
        
        # 风险控制参数
        self.stop_loss_ticks = 5           # 固定止损跳数
        self.max_holding_ticks = 300       # 最大持仓时间 (防止僵尸单)
        self.ticks_held = 0 
        
        # 状态记录与资金流水
        self.pnl_history = []
        self.price_history = []
        self.current_position = 0          # 1 为多头，-1 为空头，0 为空仓
        self.entry_price = 0.0
        self.trade_count = 0
        self.total_pnl = 0.0

    @property
    def max_drawdown(self):
        """计算全局最大回撤"""
        if not self.pnl_history: 
            return 0.0
        pnl_array = np.array(self.pnl_history)
        running_max = np.maximum.accumulate(pnl_array)
        drawdowns = running_max - pnl_array
        return np.max(drawdowns)

    def calculate_obi(self, tick):
        """计算微观订单簿不平衡度 (Order Book Imbalance)"""
        bid_vol = tick['bid_volume1']
        ask_vol = tick['ask_volume1']
        if bid_vol + ask_vol == 0: 
            return 0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def on_tick(self, tick):
        """核心大脑：逐笔 Tick 响应逻辑"""
        # 1. 维护时间序列窗口
        self.price_history.append(tick['last_price'])
        if len(self.price_history) > self.momentum_window:
            self.price_history.pop(0)
            
        if len(self.price_history) < self.momentum_window:
            return

        current_price = tick['last_price']
        momentum = current_price - self.price_history[0] 
        obi = self.calculate_obi(tick)
        
        # ==========================================
        # 2. 持仓状态下的平仓逻辑 (止盈 / 止损 / 超时)
        # ==========================================
        if self.current_position != 0:
            self.ticks_held += 1
            close_signal = False
            
            # 多头持仓的平仓条件
            if self.current_position == 1: 
                if current_price >= self.entry_price + (self.expected_profit_ticks * self.engine.price_tick):
                    close_signal = True # 止盈
                elif current_price <= self.entry_price - (self.stop_loss_ticks * self.engine.price_tick):
                    close_signal = True # 止损
                    
            # 空头持仓的平仓条件
            elif self.current_position == -1: 
                if current_price <= self.entry_price - (self.expected_profit_ticks * self.engine.price_tick):
                    close_signal = True # 止盈
                elif current_price >= self.entry_price + (self.stop_loss_ticks * self.engine.price_tick):
                    close_signal = True # 止损
                    
            # 时间止损 (流动性衰竭或行情陷入死水)
            if self.ticks_held >= self.max_holding_ticks:
                close_signal = True
                
            # 执行平仓指令
            if close_signal:
                action = 'SELL' if self.current_position == 1 else 'BUY'
                success, result = self.engine.match_order(action, 1)
                if success:
                    if self.current_position == 1:
                        gross_profit = result['price'] - self.entry_price
                    else:
                        gross_profit = self.entry_price - result['price']
                        
                    self.total_pnl += (gross_profit - result['cost'])
                    self.current_position = 0
                    self.ticks_held = 0
            
            self.pnl_history.append(self.total_pnl)
            return # 处于持仓状态时，不再执行后续开仓逻辑
        
        # ==========================================
        # 3. 空仓状态下的生存边界验证
        # ==========================================
        est_commission = current_price * (self.engine.commission_open + self.engine.commission_today)
        
        if self.is_ideal:
            estimated_cost = est_commission # 理想环境下忽略滑点与点差
        else:
            spread = tick['ask_price1'] - tick['bid_price1']
            estimated_cost = est_commission + spread # 真实摩擦 = 手续费 + 点差
            
        expected_return = self.expected_profit_ticks * self.engine.price_tick 
        
        # 核心防御：如果预期利润连摩擦成本都无法覆盖，直接放弃交易
        if expected_return <= estimated_cost:
            self.pnl_history.append(self.total_pnl)
            return 

        # ==========================================
        # 4. 信号生成与开仓执行 (A/B 实验核心区)
        # ==========================================
        buy_signal = False
        sell_signal = False
        
        if self.is_mean_reversion:
            # 【实验 A: 微观均值反转】—— 专治高频震荡与假突破
            # 当散户疯狂买入推高价格时，做市商介入做空
            if momentum >= 1 * self.engine.price_tick and obi > self.obi_threshold:
                sell_signal = True 
            # 当散户恐慌砸盘打低价格时，做市商介入做多
            elif momentum <= -1 * self.engine.price_tick and obi < -self.obi_threshold:
                buy_signal = True 
        else:
            # 【实验 B: 顺势动量】—— 容易被高频噪音反复收割
            if momentum >= 1 * self.engine.price_tick and obi > self.obi_threshold:
                buy_signal = True
            elif momentum <= -1 * self.engine.price_tick and obi < -self.obi_threshold:
                sell_signal = True

        # 执行开仓指令
        if buy_signal:
            success, result = self.engine.match_order('BUY', 1)
            if success:
                self.current_position = 1
                self.entry_price = result['price']
                self.total_pnl -= result['cost'] # 扣除开仓费
                self.trade_count += 1
                self.ticks_held = 0
                
        elif sell_signal:
            success, result = self.engine.match_order('SELL', 1)
            if success:
                self.current_position = -1
                self.entry_price = result['price']
                self.total_pnl -= result['cost'] # 扣除开仓费
                self.trade_count += 1
                self.ticks_held = 0

        self.pnl_history.append(self.total_pnl)
        
    def force_close_at_end(self):
        """强制结账：回测切片结束时，平掉所有未了结的敞口"""
        if self.current_position != 0:
            action = 'SELL' if self.current_position == 1 else 'BUY'
            success, result = self.engine.match_order(action, 1)
            if success:
                if self.current_position == 1:
                    gross_profit = result['price'] - self.entry_price
                else:
                    gross_profit = self.entry_price - result['price']
                self.total_pnl += (gross_profit - result['cost'])
                self.current_position = 0
            self.pnl_history.append(self.total_pnl)