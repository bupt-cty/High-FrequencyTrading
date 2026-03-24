import math
import numpy as np
import concurrent.futures
import os

class HFTAlgorithmOptimizer:
    def __init__(self, backtest_func, param_space, exploration_constant=1.5):
        """
        高频交易 UCB 参数优化器 (支持随机环境与并行计算)
        :param backtest_func: 黑盒评估函数 (输入参数字典，输出该次随机抽样的 Reward)
        :param param_space: List[dict] 离散化的参数网格空间
        :param exploration_constant: 探索常数 c，平衡探索与利用
        """
        self.backtest_func = backtest_func
        self.param_space = param_space
        self.c = exploration_constant
        
        self.n_arms = len(param_space)
        self.counts = [0] * self.n_arms       # 记录每个参数组合被测试的总次数
        self.values = [0.0] * self.n_arms     # 记录每个参数组合的历史期望得分 (均值)
    
    def select_arm(self, total_steps):
        """基于 UCB 逻辑选择本次要测试的参数组合索引"""
        # 1. 强制冷启动：确保每组参数至少跑过一次
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
                
        # 2. UCB 核心计算
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            exploitation = self.values[arm]
            exploration = self.c * math.sqrt(math.log(total_steps) / float(self.counts[arm]))
            ucb_values[arm] = exploitation + exploration
            
        return ucb_values.index(max(ucb_values))
        
    def update(self, chosen_arm, reward):
        """增量更新被选中参数组合的期望值 (Reward 均值)"""
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

    def run_optimization_parallel(self, total_iterations, max_workers=None):
        """多进程乐观并行 UCB 主循环"""
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 1)
            
        print(f"🚀 启动并行优化，分配 CPU 核心数: {max_workers}")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            step = 1
            while step <= total_iterations:
                batch_size = min(max_workers, total_iterations - step + 1)
                selected_arms = []
                
                # 乐观选择：同批次内预选参数，临时增加计数以降低其在同批次被重复选中的概率
                for _ in range(batch_size):
                    arm = self.select_arm(step)
                    selected_arms.append(arm)
                    self.counts[arm] += 1 
                    step += 1
                
                # 状态回滚：等待真实计算结果
                for arm in selected_arms:
                    self.counts[arm] -= 1
                
                # 提取对应的参数字典，并分发给各进程
                params_batch = [self.param_space[arm] for arm in selected_arms]
                rewards = list(executor.map(self.backtest_func, params_batch))
                
                # 获取结果后，真实更新 UCB 状态
                for arm, reward in zip(selected_arms, rewards):
                    self.update(arm, reward)
                
                # 定期打印进度
                if (step - 1) % (max_workers * 2) == 0 or (step - 1) == total_iterations:
                    best_current_arm = self.values.index(max(self.values))
                    print(f"[Iteration {step - 1}/{total_iterations}] 当前最高期望得分: {self.values[best_current_arm]:.4f} | 最优参数: {self.param_space[best_current_arm]}")
                    
        # 优化结束，返回期望值最高的最优解
        best_arm = self.values.index(max(self.values))
        return self.param_space[best_arm], self.values[best_arm]