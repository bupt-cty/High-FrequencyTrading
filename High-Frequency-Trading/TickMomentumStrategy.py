import numpy as np

class TickMomentumStrategy:
    def __init__(self, engine, momentum_window=60, obi_threshold=0.4, expected_profit_ticks=10, is_ideal=False):
        self.engine = engine
        self.momentum_window = momentum_window
        self.obi_threshold = obi_threshold
        self.expected_profit_ticks = expected_profit_ticks
        self.is_ideal = is_ideal # 核心新增：让策略知道自己身处什么环境
        
        self.stop_loss_ticks = 5 
        self.max_holding_ticks = 300 # 核心新增：最大持仓Tick数（防止僵尸单）
        self.ticks_held = 0 
        
        self.pnl_history = []
        self.price_history = []
        self.current_position = 0  
        self.entry_price = 0.0
        self.trade_count = 0
        self.total_pnl = 0.0

    @property
    def max_drawdown(self):
        if not self.pnl_history: return 0.0
        pnl_array = np.array(self.pnl_history)
        running_max = np.maximum.accumulate(pnl_array)
        drawdowns = running_max - pnl_array
        return np.max(drawdowns)

    def calculate_obi(self, tick):
        bid_vol = tick['bid_volume1']
        ask_vol = tick['ask_volume1']
        if bid_vol + ask_vol == 0: return 0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def on_tick(self, tick):
        self.price_history.append(tick['last_price'])
        if len(self.price_history) > self.momentum_window:
            self.price_history.pop(0)
            
        if len(self.price_history) < self.momentum_window:
            return

        current_price = tick['last_price']
        momentum = current_price - self.price_history[0] 
        obi = self.calculate_obi(tick)
        
        # ==========================================
        # 1. 持仓状态下的平仓逻辑 (优先处理)
        # ==========================================
        if self.current_position != 0:
            self.ticks_held += 1
            close_signal = False
            
            # 价格止盈止损 与 动量反转
            if self.current_position == 1: # 多头持仓
                if current_price >= self.entry_price + (self.expected_profit_ticks * self.engine.price_tick):
                    close_signal = True # 止盈
                elif current_price <= self.entry_price - (self.stop_loss_ticks * self.engine.price_tick):
                    close_signal = True # 止损
                elif momentum < -self.engine.price_tick*3: #扩大到3倍
                    close_signal = True # 动量实质性反转（向下），落袋为安/提前止损
                    
            elif self.current_position == -1: # 空头持仓
                if current_price <= self.entry_price - (self.expected_profit_ticks * self.engine.price_tick):
                    close_signal = True # 止盈
                elif current_price >= self.entry_price + (self.stop_loss_ticks * self.engine.price_tick):
                    close_signal = True # 止损
                elif momentum > self.engine.price_tick*3: #扩大到三倍
                    close_signal = True # 动量实质性反转（向上），落袋为安/提前止损
                    
            # 超时强平 (时间止损，防止僵尸单)
            if self.ticks_held >= self.max_holding_ticks:
                close_signal = True
                
            if close_signal:
                action = 'SELL' if self.current_position == 1 else 'BUY'
                success, result = self.engine.match_order(action, 1)
                if success:
                    # 严谨计算盈亏
                    if self.current_position == 1:
                        gross_profit = result['price'] - self.entry_price
                    else:
                        gross_profit = self.entry_price - result['price']
                        
                    self.total_pnl += (gross_profit - result['cost'])
                    self.current_position = 0
                    self.ticks_held = 0
            
            self.pnl_history.append(self.total_pnl)
            return # 持仓时不再执行后续开仓逻辑
        
        
        # ==========================================
        # 2. 空仓状态下的生存边界验证与开仓
        # ==========================================
        est_commission = current_price * (self.engine.commission_open + self.engine.commission_today)
        
        # 核心逻辑：如果是理想环境，策略大脑不应该畏惧 spread 摩擦
        if self.is_ideal:
            estimated_cost = est_commission
        else:
            spread = tick['ask_price1'] - tick['bid_price1']
            estimated_cost = est_commission + spread
            
        expected_return = self.expected_profit_ticks * self.engine.price_tick 
        
        if expected_return <= estimated_cost:
            self.pnl_history.append(self.total_pnl)
            return 

        if momentum > 0 and obi > self.obi_threshold:
            success, result = self.engine.match_order('BUY', 1)
            if success:
                self.current_position = 1
                self.entry_price = result['price']
                self.total_pnl -= result['cost']
                self.trade_count += 1
        elif momentum < 0 and obi < -self.obi_threshold:
            success, result = self.engine.match_order('SELL', 1)
            if success:
                self.current_position = -1
                self.entry_price = result['price']
                self.total_pnl -= result['cost']
                self.trade_count += 1

        self.pnl_history.append(self.total_pnl)
        
    def force_close_at_end(self):
        """核心新增：回测结束时，强制平掉所有手上未了结的单子"""
        if self.current_position != 0:
            action = 'SELL' if self.current_position == 1 else 'BUY'
            success, result = self.engine.match_order(action, 1)
            if success:
                gross_profit = (result['price'] - self.entry_price) if self.current_position == 1 else (self.entry_price - result['price'])
                self.total_pnl += (gross_profit - result['cost'])
                self.current_position = 0
            self.pnl_history.append(self.total_pnl)