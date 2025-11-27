"""
矿井通风阻力系数反演结果可视化模块

提供以下可视化功能：
1. 损失曲线
2. 阻力系数收敛轨迹
3. y_pred vs y_target 残差图
4. 多次运行结果分布图
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List, Optional, Dict
import warnings

# 设置中文字体支持
try:
    rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    rcParams['axes.unicode_minus'] = False
except:
    pass

# 导入反演模块
from ventilation_inversion import (
    InversionResult, InversionProblem, 
    CMAESOptimizer, CMAESConfig,
    LinearVentilationModel, InversionBounds,
    HuberLoss
)


def plot_loss_curve(result: InversionResult, 
                    title: str = "损失函数收敛曲线",
                    log_scale: bool = True,
                    save_path: Optional[str] = None):
    """
    绘制损失曲线
    
    参数:
    - result: 反演结果对象
    - title: 图表标题
    - log_scale: 是否使用对数坐标
    - save_path: 保存路径（None则显示）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = np.arange(1, len(result.loss_history) + 1)
    
    ax.plot(iterations, result.loss_history, 'b-', linewidth=2, label='损失值')
    ax.axhline(y=result.loss_best, color='r', linestyle='--', 
               label=f'最终损失: {result.loss_best:.2e}')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('损失值', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_parameter_convergence(result: InversionResult,
                               k_true: Optional[np.ndarray] = None,
                               param_names: Optional[List[str]] = None,
                               title: str = "阻力系数收敛轨迹",
                               save_path: Optional[str] = None):
    """
    绘制阻力系数收敛轨迹
    
    参数:
    - result: 反演结果对象
    - k_true: 真实阻力系数（如果已知）
    - param_names: 参数名称列表
    - title: 图表标题
    - save_path: 保存路径
    """
    x_history = np.array(result.x_history)
    n_params = x_history.shape[1]
    iterations = np.arange(1, len(result.x_history) + 1)
    
    if param_names is None:
        param_names = [f'k_{i}' for i in range(n_params)]
    
    # 使用颜色映射
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i in range(n_params):
        ax.plot(iterations, x_history[:, i], color=colors[i], 
                linewidth=1.5, label=param_names[i])
        
        # 绘制真实值
        if k_true is not None:
            ax.axhline(y=k_true[i], color=colors[i], linestyle='--', 
                       alpha=0.5, linewidth=1)
    
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('阻力系数 k', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(y_pred: np.ndarray, 
                   y_target: np.ndarray,
                   measurement_names: Optional[List[str]] = None,
                   title: str = "预测值 vs 观测值残差分析",
                   save_path: Optional[str] = None):
    """
    绘制残差分析图（三合一）
    
    参数:
    - y_pred: 预测值
    - y_target: 观测目标值
    - measurement_names: 测量点名称
    - title: 图表标题
    - save_path: 保存路径
    """
    residuals = y_pred - y_target
    n_measurements = len(y_target)
    
    if measurement_names is None:
        measurement_names = [f'测点{i+1}' for i in range(n_measurements)]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 子图1: y_pred vs y_target 散点图
    ax1 = axes[0]
    ax1.scatter(y_target, y_pred, c='blue', alpha=0.7, s=80, edgecolors='black')
    
    # 绘制理想对角线
    min_val = min(y_target.min(), y_pred.min())
    max_val = max(y_target.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想拟合')
    
    ax1.set_xlabel('观测值 y_target', fontsize=11)
    ax1.set_ylabel('预测值 y_pred', fontsize=11)
    ax1.set_title('预测值 vs 观测值', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 残差柱状图
    ax2 = axes[1]
    x_pos = np.arange(n_measurements)
    colors = ['green' if r >= 0 else 'red' for r in residuals]
    ax2.bar(x_pos, residuals, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax2.set_xlabel('测量点', fontsize=11)
    ax2.set_ylabel('残差 (y_pred - y_target)', fontsize=11)
    ax2.set_title('各测点残差', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'{i+1}' for i in range(n_measurements)])
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 子图3: 残差直方图
    ax3 = axes[2]
    ax3.hist(residuals, bins=max(5, n_measurements//2), color='steelblue', 
             alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零残差')
    ax3.axvline(x=np.mean(residuals), color='orange', linestyle='-', 
                linewidth=2, label=f'均值: {np.mean(residuals):.3f}')
    
    ax3.set_xlabel('残差值', fontsize=11)
    ax3.set_ylabel('频数', fontsize=11)
    ax3.set_title('残差分布', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_run_analysis(results: List[InversionResult],
                            k_true: Optional[np.ndarray] = None,
                            param_names: Optional[List[str]] = None,
                            title: str = "多次运行结果分析",
                            save_path: Optional[str] = None):
    """
    绘制多次运行结果分布图
    
    参数:
    - results: 多次运行结果列表
    - k_true: 真实阻力系数
    - param_names: 参数名称
    - title: 图表标题
    - save_path: 保存路径
    """
    n_runs = len(results)
    n_params = len(results[0].x_best)
    
    if param_names is None:
        param_names = [f'k_{i}' for i in range(n_params)]
    
    # 收集所有结果
    x_all = np.array([r.x_best for r in results])
    losses = np.array([r.loss_best for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1: 参数箱线图
    ax1 = axes[0, 0]
    bp = ax1.boxplot(x_all, labels=param_names, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    if k_true is not None:
        ax1.scatter(range(1, n_params + 1), k_true, c='red', s=100, 
                   marker='*', zorder=5, label='真实值')
        ax1.legend()
    
    ax1.set_xlabel('参数', fontsize=11)
    ax1.set_ylabel('参数值', fontsize=11)
    ax1.set_title('各参数的分布（箱线图）', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 损失值分布
    ax2 = axes[0, 1]
    ax2.bar(range(1, n_runs + 1), losses, color='steelblue', alpha=0.7, 
            edgecolor='black')
    ax2.axhline(y=np.mean(losses), color='red', linestyle='--', 
                linewidth=2, label=f'均值: {np.mean(losses):.2e}')
    ax2.axhline(y=np.min(losses), color='green', linestyle=':', 
                linewidth=2, label=f'最优: {np.min(losses):.2e}')
    
    ax2.set_xlabel('运行次数', fontsize=11)
    ax2.set_ylabel('最终损失值', fontsize=11)
    ax2.set_title('各次运行的最终损失', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 平行坐标图
    ax3 = axes[1, 0]
    
    # 归一化参数用于可视化
    x_min = x_all.min(axis=0)
    x_max = x_all.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range == 0] = 1  # 避免除零
    x_normalized = (x_all - x_min) / x_range
    
    for i in range(n_runs):
        alpha = 0.3 + 0.7 * (1 - (losses[i] - losses.min()) / (losses.max() - losses.min() + 1e-10))
        ax3.plot(range(n_params), x_normalized[i, :], 'o-', alpha=alpha, linewidth=1.5)
    
    ax3.set_xticks(range(n_params))
    ax3.set_xticklabels(param_names)
    ax3.set_xlabel('参数', fontsize=11)
    ax3.set_ylabel('归一化参数值', fontsize=11)
    ax3.set_title('平行坐标图（颜色深浅表示损失值大小）', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 参数相关性热力图
    ax4 = axes[1, 1]
    
    # 计算各运行结果的相似度矩阵
    similarity_matrix = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(n_runs):
            # 使用欧氏距离的倒数作为相似度
            dist = np.linalg.norm(x_all[i] - x_all[j])
            similarity_matrix[i, j] = 1 / (1 + dist)
    
    im = ax4.imshow(similarity_matrix, cmap='Blues', aspect='auto')
    ax4.set_xlabel('运行编号', fontsize=11)
    ax4.set_ylabel('运行编号', fontsize=11)
    ax4.set_title('解的相似度矩阵', fontsize=12)
    ax4.set_xticks(range(n_runs))
    ax4.set_yticks(range(n_runs))
    ax4.set_xticklabels([f'{i+1}' for i in range(n_runs)])
    ax4.set_yticklabels([f'{i+1}' for i in range(n_runs)])
    
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('相似度')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def run_visualization_demo():
    """
    运行完整的可视化演示
    """
    print("="*60)
    print("矿井通风阻力系数反演 - 可视化演示")
    print("="*60)
    
    # 设置问题
    n_params = 8
    n_measurements = 8
    
    np.random.seed(42)
    k_true = np.array([0.0025, 0.0030, 0.0028, 0.0022, 
                       0.0035, 0.0020, 0.0032, 0.0027])
    
    # 创建模型
    model = LinearVentilationModel(
        n_params=n_params,
        n_measurements=n_measurements,
        noise_std=0.1
    )
    
    y_target = model.forward(k_true, add_noise=True)
    forward_fn = model.create_forward_function()
    
    bounds = InversionBounds(
        lower=np.ones(n_params) * 0.001,
        upper=np.ones(n_params) * 0.010
    )
    
    from ventilation_inversion import InversionProblem
    problem = InversionProblem(
        forward_fn=forward_fn,
        y_target=y_target,
        bounds=bounds,
        loss_fn=HuberLoss(delta=0.5)
    )
    
    # 运行优化
    print("\n运行 CMA-ES 优化...")
    config = CMAESConfig(maxiter=300, verbose=0, seed=123)
    optimizer = CMAESOptimizer(problem, config)
    
    result = optimizer.run()
    print(f"优化完成，最终损失: {result.loss_best:.6e}")
    
    # 多次运行
    print("\n执行多次运行...")
    results = optimizer.run_multi(n_runs=8, random_init=True)
    print(f"最佳损失: {min(r.loss_best for r in results):.6e}")
    
    # 生成可视化
    print("\n生成可视化图表...")
    
    # 1. 损失曲线
    plot_loss_curve(result, 
                    title="CMA-ES 优化损失曲线",
                    save_path="loss_curve.png")
    
    # 2. 参数收敛轨迹
    plot_parameter_convergence(result, 
                               k_true=k_true,
                               title="阻力系数收敛轨迹",
                               save_path="parameter_convergence.png")
    
    # 3. 残差分析
    y_pred = forward_fn(result.x_best)
    plot_residuals(y_pred, y_target,
                   title="反演结果残差分析",
                   save_path="residual_analysis.png")
    
    # 4. 多次运行分析
    plot_multi_run_analysis(results,
                            k_true=k_true,
                            title="多次运行结果分析",
                            save_path="multi_run_analysis.png")
    
    print("\n所有图表已生成并保存！")
    print("  - loss_curve.png: 损失曲线")
    print("  - parameter_convergence.png: 参数收敛轨迹")
    print("  - residual_analysis.png: 残差分析")
    print("  - multi_run_analysis.png: 多次运行分析")


if __name__ == "__main__":
    run_visualization_demo()

