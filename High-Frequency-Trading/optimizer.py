import math
import numpy as np
import concurrent.futures
import os

class HFTAlgorithmOptimizer:
    def __init__(self, backtest_func, param_space, exploration_constant=1.5):
        self.backtest_func = backtest_func
        self.param_space = param_space
        self.c = exploration_constant
        
        self.n_arms = len(param_space)
        self.counts = [0] * self.n_arms       
        self.values = [0.0] * self.n_arms     
    
    def select_arm(self, total_steps):
        # 强制冷启动
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
                
        # 计算 UCB 值
        ucb_values = [0.0] * self.n_arms
        for arm in range(self.n_arms):
            exploitation = self.values[arm]
            exploration = self.c * math.sqrt(math.log(total_steps) / float(self.counts[arm]))
            ucb_values[arm] = exploitation + exploration
            
        return ucb_values.index(max(ucb_values))
        
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / float(n)) * value + (1 / float(n)) * reward

    def run_optimization_parallel(self, total_iterations, max_workers=None):
        """
        多进程并行 UCB 优化主循环
        """
        if max_workers is None:
            # 默认使用系统 CPU 核心数 - 1，留一个核心保证系统不卡顿
            max_workers = max(1, os.cpu_count() - 1)
            
        print(f"🚀 启动并行优化，分配 CPU 核心数: {max_workers}")

        # 使用进程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            step = 1
            while step <= total_iterations:
                # 决定当前批次的大小
                batch_size = min(max_workers, total_iterations - step + 1)
                
                selected_arms = []
                # 乐观选择阶段 (Optimistic Selection)
                for _ in range(batch_size):
                    arm = self.select_arm(step)
                    selected_arms.append(arm)
                    # 临时增加 count，降低其探索期望，防止同批次选出相同的参数
                    self.counts[arm] += 1 
                    step += 1
                
                # 状态回滚：等待真实结果回来后再进行标准的 update
                for arm in selected_arms:
                    self.counts[arm] -= 1
                
                # 提取对应的参数字典
                params_batch = [self.param_space[arm] for arm in selected_arms]
                
                # 将任务映射到多进程池中并发执行 (这里的 map 会阻塞直到这批任务全完成)
                rewards = list(executor.map(self.backtest_func, params_batch))
                
                # 获取结果后，真实更新 UCB 状态
                for arm, reward in zip(selected_arms, rewards):
                    self.update(arm, reward)
                
                if (step - 1) % (max_workers * 2) == 0 or (step - 1) == total_iterations:
                    best_current_arm = self.values.index(max(self.values))
                    print(f"[Iteration {step - 1}/{total_iterations}] 当前最高得分: {self.values[best_current_arm]:.4f} | 最优参数: {self.param_space[best_current_arm]}")
                    
        best_arm = self.values.index(max(self.values))
        return self.param_space[best_arm], self.values[best_arm]