#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#   自适应差分进化收敛性判断模块
#
###########################################################################################################################

import numpy as np

def check_convergence(fitness_history, window=30, threshold=1e-5, min_gen=50, current_gen=0):
    """
    检查差分进化算法是否收敛
    
    参数:
        fitness_history: list - 适应值历史列表，每个元素是一代的适应值数组
        window: int - 收敛判断窗口大小（检查最近N代）
        threshold: float - 收敛阈值（适应值相对变化率）
        min_gen: int - 最小进化代数（至少运行N代才判断收敛）
        current_gen: int - 当前进化代数
    
    返回:
        tuple: (is_converged: bool, reason: str)
            is_converged: 是否收敛
            reason: 收敛原因或状态描述
    """
    # 如果代数小于最小代数，不判断收敛
    if current_gen < min_gen:
        return False, f"当前代数 {current_gen} < 最小代数 {min_gen}"
    
    # 如果历史记录不足窗口大小，不判断收敛
    if len(fitness_history) < window:
        return False, f"历史记录不足 {len(fitness_history)} < 窗口大小 {window}"
    
    # 提取最近window代的适应值
    recent_fitness = fitness_history[-window:]
    
    # 计算每代的最优适应值（最小值，因为是最小化问题）
    best_fitness_per_gen = []
    for gen_fitness in recent_fitness:
        if isinstance(gen_fitness, np.ndarray):
            # 如果是数组，取最小值
            if gen_fitness.size > 0:
                best_fitness_per_gen.append(np.min(gen_fitness))
        else:
            # 如果是标量
            best_fitness_per_gen.append(float(gen_fitness))
    
    if len(best_fitness_per_gen) < 2:
        return False, "适应值历史不足"
    
    # 计算相对改善率
    start_fitness = best_fitness_per_gen[0]
    end_fitness = best_fitness_per_gen[-1]
    
    # 避免除零
    if abs(start_fitness) < 1e-10:
        relative_improvement = abs(end_fitness - start_fitness)
    else:
        relative_improvement = abs(end_fitness - start_fitness) / abs(start_fitness)
    
    # 判断是否收敛（相对改善小于阈值）
    if relative_improvement < threshold:
        return True, f"收敛：相对改善率 {relative_improvement:.2e} < 阈值 {threshold:.2e}"
    
    # 检查是否停滞（最近几代无改善）
    if len(best_fitness_per_gen) >= 10:
        recent_improvement = abs(best_fitness_per_gen[-10] - best_fitness_per_gen[-1])
        if recent_improvement < threshold * abs(start_fitness):
            return True, f"停滞：最近10代改善 {recent_improvement:.2e} 过小"
    
    return False, f"未收敛：相对改善率 {relative_improvement:.2e} >= 阈值 {threshold:.2e}"


def check_stagnation(fitness_history, stagnation_window=30, improvement_threshold=1e-6):
    """
    检查算法是否停滞（连续多代无显著改善）
    
    参数:
        fitness_history: list - 适应值历史列表
        stagnation_window: int - 停滞判断窗口大小
        improvement_threshold: float - 改善阈值
    
    返回:
        tuple: (is_stagnant: bool, reason: str)
    """
    if len(fitness_history) < stagnation_window:
        return False, f"历史记录不足 {len(fitness_history)} < 停滞窗口 {stagnation_window}"
    
    # 提取最近stagnation_window代的适应值
    recent_fitness = fitness_history[-stagnation_window:]
    
    # 计算每代的最优适应值
    best_fitness_per_gen = []
    for gen_fitness in recent_fitness:
        if isinstance(gen_fitness, np.ndarray):
            if gen_fitness.size > 0:
                best_fitness_per_gen.append(np.min(gen_fitness))
        else:
            best_fitness_per_gen.append(float(gen_fitness))
    
    if len(best_fitness_per_gen) < 2:
        return False, "适应值历史不足"
    
    # 计算总改善
    start_fitness = best_fitness_per_gen[0]
    end_fitness = best_fitness_per_gen[-1]
    total_improvement = abs(end_fitness - start_fitness)
    
    # 计算基准值（用于相对判断）
    baseline = max(abs(start_fitness), abs(end_fitness), 1e-10)
    
    # 判断是否停滞
    if total_improvement < improvement_threshold * baseline:
        return True, f"停滞：{stagnation_window}代内总改善 {total_improvement:.2e} < 阈值 {improvement_threshold * baseline:.2e}"
    
    return False, f"未停滞：总改善 {total_improvement:.2e} >= 阈值 {improvement_threshold * baseline:.2e}"


def should_terminate_early(fitness_history, config_adaptive):
    """
    综合判断是否应该提前终止
    
    参数:
        fitness_history: list - 适应值历史列表
        config_adaptive: dict - 自适应配置参数
    
    返回:
        tuple: (should_terminate: bool, reason: str, current_gen: int)
    """
    if not config_adaptive.get("adaptiveMode", False):
        return False, "自适应模式未启用", 0
    
    # 从fitness_history推断当前代数
    current_gen = len(fitness_history)
    
    min_gen = config_adaptive.get("minGen", 50)
    convergence_window = config_adaptive.get("convergenceWindow", 30)
    convergence_threshold = config_adaptive.get("convergenceThreshold", 1e-5)
    stagnation_count = config_adaptive.get("stagnationCount", 30)
    
    # 检查是否达到最小代数
    if current_gen < min_gen:
        return False, f"未达最小代数 {current_gen}/{min_gen}", current_gen
    
    # 检查收敛性
    is_converged, conv_reason = check_convergence(
        fitness_history, 
        window=convergence_window,
        threshold=convergence_threshold,
        min_gen=min_gen,
        current_gen=current_gen
    )
    
    if is_converged:
        return True, f"收敛：{conv_reason}", current_gen
    
    # 检查停滞
    is_stagnant, stag_reason = check_stagnation(
        fitness_history,
        stagnation_window=stagnation_count,
        improvement_threshold=convergence_threshold
    )
    
    if is_stagnant:
        return True, f"停滞：{stag_reason}", current_gen
    
    return False, f"继续运行：{conv_reason}", current_gen

