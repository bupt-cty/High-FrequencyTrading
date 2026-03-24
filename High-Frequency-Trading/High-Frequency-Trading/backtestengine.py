import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, data_path, commission_open=0.0001, commission_today=0.0001, commission_yesterday=0.0001, price_tick=1.0):
        self.data_path = data_path
        
        # 费率精细化拆分 (契合 CTP 底层逻辑)
        self.commission_open = commission_open       # 开仓费率
        self.commission_today = commission_today     # 平今费率
        self.commission_yesterday = commission_yesterday # 平昨费率
        self.price_tick = price_tick                 # 品种最小变动价位 (螺纹钢通常为1)
        
        self.current_tick = None
        self.tick_history = []
        self.volatility_window = 20
        
        # 仓位明细账本：记录每一笔开仓的 [日期, 价格, 数量, 方向]
        # 例如: [{'date': '2026-03-25', 'price': 4000, 'volume': 1, 'direction': 'BUY'}]
        self.position_ledger = [] 
        
    def execute_with_market_impact(self, order_volume, l1_price, l1_volume, is_buy):
        """
        替代原本凭空捏造的指数滑点，基于订单穿透模拟滑点
        """
        remaining_vol = order_volume
        total_cost = 0.0
        current_level_price = l1_price
        current_level_vol = l1_volume
        
        while remaining_vol > 0:
            executed_vol = min(remaining_vol, current_level_vol)
            total_cost += executed_vol * current_level_price
            remaining_vol -= executed_vol
            
            # 如果一档被吃光了，根据最小变动价位向下一档推演 (虚拟吃单)
            if remaining_vol > 0:
                if is_buy:
                    current_level_price += self.price_tick # 买入越买越贵
                else:
                    current_level_price -= self.price_tick # 卖出越卖越便宜
                # 假设深层盘口的挂单量与一档相同 (一种简化的市场微观结构假设)
                current_level_vol = l1_volume 
                
        avg_execution_price = total_cost / order_volume
        return avg_execution_price

    def match_order(self, direction, volume):
        if not self.current_tick:
            return False, "No market data"
            
        current_date = str(self.current_tick['datetime'])[:10] # 提取 YYYY-MM-DD
        
        # 1. 识别开平仓意图 (简化逻辑：账本为空即为开仓，否则检查方向)
        is_opening = True
        if self.position_ledger:
            # 如果账本里是多单，现在发的是SELL，则是平仓
            if self.position_ledger[0]['direction'] != direction:
                is_opening = False

        # 2. 计算成交均价 (包含流动性穿透滑点)
        if direction == 'BUY':
            exec_price = self.execute_with_market_impact(volume, self.current_tick['ask_price1'], self.current_tick['ask_volume1'], True)
        else:
            exec_price = self.execute_with_market_impact(volume, self.current_tick['bid_price1'], self.current_tick['bid_volume1'], False)

        # 3. 计算手续费与账本更新
        trade_cost = 0.0
        
        if is_opening:
            # 开仓逻辑
            trade_cost = exec_price * volume * self.commission_open
            self.position_ledger.append({
                'date': current_date,
                'price': exec_price,
                'volume': volume,
                'direction': direction
            })
        else:
            # 平仓逻辑：先平今，后平昨 (符合国内期货交易所规则)
            temp_vol = volume
            # 倒序遍历账本，优先平最新建立的仓位 (模拟平今)
            for i in range(len(self.position_ledger)-1, -1, -1):
                pos = self.position_ledger[i]
                if temp_vol <= 0: break
                
                close_vol = min(temp_vol, pos['volume'])
                
                # 核心判别：平今还是平昨？
                if pos['date'] == current_date:
                    trade_cost += exec_price * close_vol * self.commission_today
                else:
                    trade_cost += exec_price * close_vol * self.commission_yesterday
                    
                pos['volume'] -= close_vol
                temp_vol -= close_vol
                
            # 清理已经平完的账本记录
            self.position_ledger = [p for p in self.position_ledger if p['volume'] > 0]

        return True, {'price': exec_price, 'volume': volume, 'cost': trade_cost}