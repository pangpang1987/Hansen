"""
矿井通风阻力系数反演 - 实时可视化模块

在 CMA-ES 优化过程中实时显示关键指标
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Optional, Callable
import time
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from threading import Thread, Lock
import queue
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class OptimizationState:
    """优化状态数据"""
    iteration: int = 0
    loss_history: List[float] = field(default_factory=list)
    best_loss_history: List[float] = field(default_factory=list)
    sigma_history: List[float] = field(default_factory=list)
    eval_count: int = 0
    elapsed_time: float = 0.0
    
    # 当前最优解的详细信息
    current_best_x: Optional[np.ndarray] = None
    current_y_pred: Optional[np.ndarray] = None
    current_y_target: Optional[np.ndarray] = None
    current_residuals: Optional[np.ndarray] = None
    
    # 参数变化跟踪（只跟踪部分关键参数）
    param_history: List[np.ndarray] = field(default_factory=list)
    
    # 每轮种群的欧式距离统计（预测与观测的方差）
    euclidean_best_history: List[float] = field(default_factory=list)    # 最佳个体
    euclidean_worst_history: List[float] = field(default_factory=list)   # 最差个体
    euclidean_mean_history: List[float] = field(default_factory=list)    # 平均值
    euclidean_std_history: List[float] = field(default_factory=list)     # 标准差
    
    # 锁
    lock: Lock = field(default_factory=Lock)


class LiveVisualizer:
    """
    实时可视化器
    
    在优化过程中实时更新四个子图：
    1. 损失曲线
    2. Sigma（步长）变化
    3. 残差分布
    4. 预测值 vs 目标值
    """
    
    def __init__(self, 
                 y_target: np.ndarray,
                 update_interval: float = 1.0,
                 max_display_points: int = 50):
        """
        参数:
        - y_target: 目标值数组
        - update_interval: 更新间隔（秒）
        - max_display_points: 残差图最多显示的点数
        """
        self.y_target = y_target
        self.update_interval = update_interval
        self.max_display_points = max_display_points
        
        self.state = OptimizationState()
        self.is_running = False
        self.start_time = None
        
        # 创建图形
        self._setup_figure()
    
    def _setup_figure(self):
        """设置图形和子图"""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('CMA-ES Inversion Optimization Monitor', fontsize=14, fontweight='bold')
        
        # 使用 GridSpec 实现灵活布局：2行3列
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 子图1: 损失曲线 (左上)
        self.ax_loss = self.fig.add_subplot(gs[0, 0])
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Convergence')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='Current')
        self.line_best, = self.ax_loss.plot([], [], 'r--', linewidth=1.5, label='Best')
        self.ax_loss.legend(loc='upper right')
        
        # 子图2: Sigma 变化 (中上)
        self.ax_sigma = self.fig.add_subplot(gs[0, 1])
        self.ax_sigma.set_xlabel('Iteration')
        self.ax_sigma.set_ylabel('Sigma (Step Size)')
        self.ax_sigma.set_title('CMA-ES Step Size')
        self.ax_sigma.grid(True, alpha=0.3)
        self.line_sigma, = self.ax_sigma.plot([], [], 'g-', linewidth=2)
        
        # 子图3: 欧式距离统计 (右上) - 新增！
        self.ax_euclidean = self.fig.add_subplot(gs[0, 2])
        self.ax_euclidean.set_xlabel('Iteration')
        self.ax_euclidean.set_ylabel('Euclidean Distance')
        self.ax_euclidean.set_title('Population Fitness (Pred vs Target)')
        self.ax_euclidean.grid(True, alpha=0.3)
        self.line_euc_best, = self.ax_euclidean.plot([], [], 'g-', linewidth=2, label='Best')
        self.line_euc_mean, = self.ax_euclidean.plot([], [], 'b-', linewidth=1.5, label='Mean')
        self.line_euc_worst, = self.ax_euclidean.plot([], [], 'r-', linewidth=1.5, label='Worst')
        # 填充区域表示标准差
        self.ax_euclidean.legend(loc='upper right')
        
        # 子图4: 残差分布 (左下)
        self.ax_residual = self.fig.add_subplot(gs[1, 0])
        self.ax_residual.set_xlabel('Measurement Index')
        self.ax_residual.set_ylabel('Residual (Pred - Target)')
        self.ax_residual.set_title('Residual Distribution')
        self.ax_residual.axhline(y=0, color='black', linestyle='-', linewidth=1)
        self.ax_residual.grid(True, alpha=0.3)
        
        # 子图5: 预测 vs 目标 (中下)
        self.ax_scatter = self.fig.add_subplot(gs[1, 1])
        self.ax_scatter.set_xlabel('Target Q (m3/s)')
        self.ax_scatter.set_ylabel('Predicted Q (m3/s)')
        self.ax_scatter.set_title('Predicted vs Target')
        self.ax_scatter.grid(True, alpha=0.3)
        
        # 子图6: 欧式距离分布箱线图 (右下) - 新增！
        self.ax_boxplot = self.fig.add_subplot(gs[1, 2])
        self.ax_boxplot.set_xlabel('Recent Iterations')
        self.ax_boxplot.set_ylabel('Euclidean Distance')
        self.ax_boxplot.set_title('Recent Population Distribution')
        self.ax_boxplot.grid(True, alpha=0.3)
        
        # 状态文本
        self.status_text = self.fig.text(
            0.02, 0.02, '', fontsize=9, 
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # 调整子图间距
        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)
    
    def update_state(self, 
                     iteration: int,
                     loss: float,
                     best_loss: float,
                     sigma: float,
                     y_pred: Optional[np.ndarray] = None,
                     x_best: Optional[np.ndarray] = None,
                     eval_count: int = 0,
                     euclidean_stats: Optional[Dict] = None):
        """
        更新优化状态（由优化器回调）
        
        参数:
        - euclidean_stats: 种群欧式距离统计 {'best': float, 'worst': float, 'mean': float, 'std': float}
        """
        with self.state.lock:
            self.state.iteration = iteration
            self.state.loss_history.append(loss)
            self.state.best_loss_history.append(best_loss)
            self.state.sigma_history.append(sigma)
            self.state.eval_count = eval_count
            
            if self.start_time:
                self.state.elapsed_time = time.time() - self.start_time
            
            if y_pred is not None:
                self.state.current_y_pred = y_pred.copy()
                self.state.current_y_target = self.y_target.copy()
                self.state.current_residuals = y_pred - self.y_target
            
            if x_best is not None:
                self.state.current_best_x = x_best.copy()
            
            # 更新欧式距离统计
            if euclidean_stats is not None:
                self.state.euclidean_best_history.append(euclidean_stats['best'])
                self.state.euclidean_worst_history.append(euclidean_stats['worst'])
                self.state.euclidean_mean_history.append(euclidean_stats['mean'])
                self.state.euclidean_std_history.append(euclidean_stats['std'])
    
    def _update_plots(self, frame):
        """更新所有图表"""
        with self.state.lock:
            iterations = list(range(1, len(self.state.loss_history) + 1))
            
            if len(iterations) == 0:
                return
            
            # 更新损失曲线
            self.line_loss.set_data(iterations, self.state.loss_history)
            self.line_best.set_data(iterations, self.state.best_loss_history)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()
            
            # 更新 sigma 曲线
            self.line_sigma.set_data(iterations, self.state.sigma_history)
            self.ax_sigma.relim()
            self.ax_sigma.autoscale_view()
            
            # 更新欧式距离曲线
            if len(self.state.euclidean_best_history) > 0:
                euc_iters = list(range(1, len(self.state.euclidean_best_history) + 1))
                self.line_euc_best.set_data(euc_iters, self.state.euclidean_best_history)
                self.line_euc_mean.set_data(euc_iters, self.state.euclidean_mean_history)
                self.line_euc_worst.set_data(euc_iters, self.state.euclidean_worst_history)
                
                # 清除之前的填充区域并重新绘制
                # 移除所有 PolyCollection (fill_between 创建的)
                for coll in list(self.ax_euclidean.collections):
                    coll.remove()
                
                mean_arr = np.array(self.state.euclidean_mean_history)
                std_arr = np.array(self.state.euclidean_std_history)
                self.ax_euclidean.fill_between(
                    euc_iters, 
                    mean_arr - std_arr, 
                    mean_arr + std_arr,
                    alpha=0.2, color='blue'
                )
                
                self.ax_euclidean.relim()
                self.ax_euclidean.autoscale_view()
                
                # 更新箱线图 (最近10次迭代)
                self.ax_boxplot.clear()
                self.ax_boxplot.set_xlabel('Recent Iterations')
                self.ax_boxplot.set_ylabel('Euclidean Distance')
                self.ax_boxplot.set_title('Recent Population Distribution')
                self.ax_boxplot.grid(True, alpha=0.3)
                
                n_recent = min(10, len(self.state.euclidean_best_history))
                if n_recent > 0:
                    recent_data = []
                    recent_labels = []
                    for i in range(-n_recent, 0):
                        # 使用 best, mean, worst 构造近似分布
                        best = self.state.euclidean_best_history[i]
                        mean = self.state.euclidean_mean_history[i]
                        worst = self.state.euclidean_worst_history[i]
                        std = self.state.euclidean_std_history[i]
                        # 生成近似数据点用于箱线图
                        approx_data = [best, mean - std, mean, mean + std, worst]
                        recent_data.append(approx_data)
                        iter_num = len(self.state.euclidean_best_history) + i + 1
                        recent_labels.append(str(iter_num))
                    
                    bp = self.ax_boxplot.boxplot(recent_data, tick_labels=recent_labels, patch_artist=True)
                    for patch in bp['boxes']:
                        patch.set_facecolor('lightblue')
                        patch.set_alpha(0.7)
            
            # 更新残差图
            if self.state.current_residuals is not None:
                self.ax_residual.clear()
                self.ax_residual.set_xlabel('Measurement Index')
                self.ax_residual.set_ylabel('Residual (Pred - Target)')
                self.ax_residual.set_title('Residual Distribution')
                self.ax_residual.axhline(y=0, color='black', linestyle='-', linewidth=1)
                self.ax_residual.grid(True, alpha=0.3)
                
                residuals = self.state.current_residuals
                n_points = min(len(residuals), self.max_display_points)
                indices = np.linspace(0, len(residuals)-1, n_points, dtype=int)
                
                colors = ['green' if r >= 0 else 'red' for r in residuals[indices]]
                self.ax_residual.bar(range(n_points), residuals[indices], 
                                    color=colors, alpha=0.7, width=0.8)
            
            # 更新预测 vs 目标散点图
            if self.state.current_y_pred is not None:
                self.ax_scatter.clear()
                self.ax_scatter.set_xlabel('Target Q (m3/s)')
                self.ax_scatter.set_ylabel('Predicted Q (m3/s)')
                self.ax_scatter.set_title('Predicted vs Target')
                self.ax_scatter.grid(True, alpha=0.3)
                
                y_pred = self.state.current_y_pred
                y_target = self.state.current_y_target
                
                self.ax_scatter.scatter(y_target, y_pred, c='blue', alpha=0.5, s=20)
                
                # 绘制理想对角线
                min_val = min(y_target.min(), y_pred.min())
                max_val = max(y_target.max(), y_pred.max())
                self.ax_scatter.plot([min_val, max_val], [min_val, max_val], 
                                    'r--', linewidth=2, label='Ideal')
                self.ax_scatter.legend()
            
            # 更新状态文本
            if len(self.state.loss_history) > 0:
                rmse = np.sqrt(np.mean(self.state.current_residuals**2)) if self.state.current_residuals is not None else 0
                rel_err = np.mean(np.abs(self.state.current_residuals) / (np.abs(self.y_target) + 0.1)) * 100 if self.state.current_residuals is not None else 0
                
                # 欧式距离统计
                euc_best = self.state.euclidean_best_history[-1] if self.state.euclidean_best_history else 0
                euc_mean = self.state.euclidean_mean_history[-1] if self.state.euclidean_mean_history else 0
                euc_worst = self.state.euclidean_worst_history[-1] if self.state.euclidean_worst_history else 0
                
                status = (
                    f"Iter: {self.state.iteration:4d} | "
                    f"Loss: {self.state.loss_history[-1]:.4e} | "
                    f"Best: {min(self.state.best_loss_history):.4e} | "
                    f"Sigma: {self.state.sigma_history[-1]:.4f} | "
                    f"Evals: {self.state.eval_count:5d} | "
                    f"EucDist(B/M/W): {euc_best:.2f}/{euc_mean:.2f}/{euc_worst:.2f} | "
                    f"RMSE: {rmse:.3f} | "
                    f"Time: {self.state.elapsed_time:.1f}s"
                )
                self.status_text.set_text(status)
        
        return self.line_loss, self.line_best, self.line_sigma
    
    def start(self):
        """启动可视化"""
        self.is_running = True
        self.start_time = time.time()
        
        # 使用动画更新
        self.ani = FuncAnimation(
            self.fig, 
            self._update_plots,
            interval=int(self.update_interval * 1000),
            blit=False,
            cache_frame_data=False
        )
        
        plt.ion()  # 交互模式
        plt.show(block=False)
    
    def stop(self):
        """停止可视化"""
        self.is_running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
    
    def save(self, filepath: str = "optimization_progress.png"):
        """保存当前图表"""
        self._update_plots(None)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"图表已保存: {filepath}")
    
    def show_final(self):
        """显示最终结果"""
        plt.ioff()
        self._update_plots(None)
        plt.show()


class IterationResultSaver:
    """
    迭代结果保存器
    
    将每次迭代的结果保存到 results 文件夹下的 JSON 文件
    """
    
    def __init__(self, 
                 output_dir: str = "results",
                 save_every: int = 1,
                 network_data = None,
                 inv_config = None):
        """
        参数:
        - output_dir: 输出目录
        - save_every: 每隔多少次迭代保存一次
        - network_data: 网络数据（用于保存巷道ID等信息）
        - inv_config: 反演配置
        """
        self.output_dir = output_dir
        self.save_every = save_every
        self.network_data = network_data
        self.inv_config = inv_config
        
        # 创建带时间戳的运行目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        
        # 创建目录
        os.makedirs(self.run_dir, exist_ok=True)
        print(f"结果将保存到: {self.run_dir}")
        
        # 保存运行配置
        self._save_run_config()
    
    def _save_run_config(self):
        """保存运行配置"""
        config_data = {
            "timestamp": self.timestamp,
            "save_every": self.save_every,
            "n_roads": len(self.network_data.roads) if self.network_data else 0,
            "n_measurements": len(self.inv_config.target_values) if self.inv_config else 0,
        }
        
        config_path = os.path.join(self.run_dir, "run_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def save_iteration(self,
                       iteration: int,
                       x_best: np.ndarray,
                       loss: float,
                       best_loss: float,
                       sigma: float,
                       eval_count: int,
                       elapsed_time: float,
                       y_pred: Optional[np.ndarray] = None,
                       y_target: Optional[np.ndarray] = None,
                       euclidean_stats: Optional[Dict] = None):
        """
        保存单次迭代结果
        """
        if iteration % self.save_every != 0:
            return
        
        # 构建结果数据
        result_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "loss": float(loss),
                "best_loss": float(best_loss),
                "sigma": float(sigma),
                "eval_count": eval_count,
                "elapsed_time": float(elapsed_time)
            }
        }
        
        # 添加欧式距离统计（种群预测与观测的方差）
        if euclidean_stats is not None:
            result_data["population_euclidean_distance"] = {
                "best": euclidean_stats['best'],
                "worst": euclidean_stats['worst'],
                "mean": euclidean_stats['mean'],
                "std": euclidean_stats['std']
            }
        
        # 添加预测误差统计
        if y_pred is not None and y_target is not None:
            residuals = y_pred - y_target
            result_data["metrics"]["rmse"] = float(np.sqrt(np.mean(residuals**2)))
            result_data["metrics"]["mae"] = float(np.mean(np.abs(residuals)))
            result_data["metrics"]["max_error"] = float(np.max(np.abs(residuals)))
            result_data["metrics"]["relative_error_percent"] = float(
                np.mean(np.abs(residuals) / (np.abs(y_target) + 0.1)) * 100
            )
            # 最佳个体的欧式距离 (参考 FuncFitness.py 第94行)
            # _fitness_value = math.sqrt(sum((q0 - q1) ** 2 for q0, q1 in zip(q0s, q1s)))
            result_data["metrics"]["euclidean_distance"] = float(
                np.sqrt(sum((q_pred - q_target) ** 2 for q_pred, q_target in zip(y_pred, y_target)))
            )
        
        # 解码并保存优化后的风阻参数
        if self.inv_config and self.network_data:
            R_opt, H_opt = self.inv_config.decode_parameters(x_best)
            
            # 保存风阻变化
            result_data["optimized_R"] = {}
            for i, road in enumerate(self.network_data.roads):
                r0 = road.r0
                r_opt = float(R_opt[i])
                change_percent = (r_opt - r0) / r0 * 100 if r0 != 0 else 0
                result_data["optimized_R"][road.id] = {
                    "r0": r0,
                    "r_optimized": r_opt,
                    "change_percent": float(change_percent)
                }
            
            # 保存风机压力变化（如果优化了风机）
            if self.inv_config.optimize_fan and len(self.inv_config.fan_to_optimize) > 0:
                result_data["optimized_H"] = {}
                for j, fan in enumerate(self.inv_config.fan_to_optimize):
                    fan_idx = self.network_data.fans.index(fan)
                    h0 = fan.h0
                    h_opt = float(H_opt[fan_idx])
                    change_percent = (h_opt - h0) / abs(h0) * 100 if h0 != 0 else 0
                    result_data["optimized_H"][fan.id] = {
                        "eid": fan.edge_id,
                        "h0": h0,
                        "h_optimized": h_opt,
                        "min_h": fan.min_h,
                        "max_h": fan.max_h,
                        "change_percent": float(change_percent)
                    }
            
            # 保存测点预测值
            if y_pred is not None and y_target is not None:
                result_data["measurements"] = {}
                for i, idx in enumerate(self.inv_config.measurement_indices):
                    road = self.network_data.roads[idx]
                    result_data["measurements"][road.id] = {
                        "target": float(y_target[i]),
                        "predicted": float(y_pred[i]),
                        "residual": float(y_pred[i] - y_target[i])
                    }
        
        # 保存到文件
        filename = f"iter_{iteration:05d}.json"
        filepath = os.path.join(self.run_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
    
    def save_final_summary(self, 
                           result,
                           total_iterations: int):
        """
        保存最终汇总结果
        """
        summary_data = {
            "run_id": self.timestamp,
            "completed_at": datetime.now().isoformat(),
            "total_iterations": total_iterations,
            "final_metrics": {
                "best_loss": float(result.loss_best),
                "total_evaluations": result.n_evaluations,
                "elapsed_time": float(result.elapsed_time)
            },
            "convergence_info": {
                k: (float(v) if isinstance(v, (int, float, np.number)) else str(v))
                for k, v in result.convergence_info.items()
            }
        }
        
        # 保存最终参数（完整）
        if self.inv_config and self.network_data:
            R_opt, H_opt = self.inv_config.decode_parameters(result.x_best)
            
            summary_data["final_R"] = [
                {
                    "id": road.id,
                    "r0": road.r0,
                    "r_optimized": float(R_opt[i]),
                    "min_r": road.min_r,
                    "max_r": road.max_r,
                    "change_percent": float((R_opt[i] - road.r0) / road.r0 * 100) if road.r0 != 0 else 0
                }
                for i, road in enumerate(self.network_data.roads)
            ]
        
        # 保存汇总文件
        summary_path = os.path.join(self.run_dir, "summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"最终结果已保存到: {summary_path}")
        
        return summary_path


class LiveCMAESOptimizer:
    """
    带实时可视化的 CMA-ES 优化器（支持 BIPOP-CMA-ES：大小种群交替）
    """
    
    def __init__(self, 
                 problem,
                 config,
                 visualizer: Optional[LiveVisualizer] = None,
                 result_saver: Optional[IterationResultSaver] = None):
        """
        参数:
        - problem: InversionProblem 实例
        - config: CMAESConfig 配置
        - visualizer: LiveVisualizer 实例（可选）
        - result_saver: IterationResultSaver 实例（可选）
        """
        self.problem = problem
        self.config = config
        self.visualizer = visualizer
        self.result_saver = result_saver
        self._cma = None
        
        # BIPOP-CMA-ES 状态
        self._bipop_state = {
            'current_popsize': None,
            'popsize_large': None,
            'popsize_small': None,
            'last_switch_iter': 0,
            'last_best_loss': None,
            'stagnation_count': 0,
            'use_large_pop': True
        }
        
        # 并行计算状态
        self._pool = None
        self._n_workers = None
        if self.config.use_parallel:
            self._n_workers = self.config.n_workers
            if self._n_workers is None:
                self._n_workers = max(1, mp.cpu_count() - 1)
            self._n_workers = min(self._n_workers, mp.cpu_count())
    
    def _import_cma(self):
        if self._cma is None:
            import cma
            self._cma = cma
        return self._cma
    
    def _initialize_bipop(self, dim: int, default_popsize: Optional[int] = None):
        """初始化 BIPOP-CMA-ES 参数"""
        if default_popsize is None:
            default_popsize = 4 + int(3 * np.log(dim))
        
        self._bipop_state['popsize_large'] = max(
            int(default_popsize * self.config.popsize_large_ratio),
            default_popsize + 1
        )
        self._bipop_state['popsize_small'] = max(
            int(default_popsize * self.config.popsize_small_ratio),
            4
        )
        
        self._bipop_state['current_popsize'] = self._bipop_state['popsize_large']
        self._bipop_state['use_large_pop'] = True
        self._bipop_state['last_switch_iter'] = 0
        self._bipop_state['last_best_loss'] = None
        self._bipop_state['stagnation_count'] = 0
    
    def _should_switch_population(self, current_iter: int, current_best_loss: float) -> bool:
        """判断是否应该切换种群大小"""
        if not self.config.use_bipop:
            return False
        
        state = self._bipop_state
        
        if self.config.switch_condition == 'interval':
            if current_iter - state['last_switch_iter'] >= self.config.switch_interval:
                return True
        elif self.config.switch_condition == 'stagnation':
            if state['last_best_loss'] is not None:
                relative_improvement = abs(state['last_best_loss'] - current_best_loss) / (
                    abs(state['last_best_loss']) + 1e-10
                )
                
                if relative_improvement < self.config.stagnation_threshold:
                    state['stagnation_count'] += 1
                else:
                    state['stagnation_count'] = 0
                
                if state['stagnation_count'] >= self.config.stagnation_window:
                    return True
        
        return False
    
    def _switch_population_size(self):
        """切换种群大小"""
        state = self._bipop_state
        state['use_large_pop'] = not state['use_large_pop']
        
        if state['use_large_pop']:
            state['current_popsize'] = state['popsize_large']
        else:
            state['current_popsize'] = state['popsize_small']
        
        state['last_switch_iter'] = 0
        state['stagnation_count'] = 0
    
    def _get_pool(self):
        """获取或创建进程池"""
        if not self.config.use_parallel:
            return None
        
        if self.config.reuse_pool:
            if self._pool is None:
                if self.config.parallel_backend == 'multiprocessing':
                    self._pool = mp.Pool(processes=self._n_workers)
                elif self.config.parallel_backend == 'concurrent.futures':
                    self._pool = ProcessPoolExecutor(max_workers=self._n_workers)
            return self._pool
        else:
            if self.config.parallel_backend == 'multiprocessing':
                return mp.Pool(processes=self._n_workers)
            elif self.config.parallel_backend == 'concurrent.futures':
                return ProcessPoolExecutor(max_workers=self._n_workers)
    
    def _close_pool(self, pool):
        """关闭进程池"""
        if pool is None:
            return
        
        if self.config.reuse_pool:
            return
        
        if isinstance(pool, ProcessPoolExecutor):
            pool.shutdown(wait=True)
        elif isinstance(pool, mp.pool.Pool):
            pool.close()
            pool.join()
    
    def _evaluate_fitness_parallel(self, solutions: List[np.ndarray], pool) -> List[float]:
        """并行评估适应度"""
        if pool is None:
            return [self.problem.objective(x) for x in solutions]
        
        # 尝试并行计算
        try:
            # 只在第一次测试序列化（避免每次都测试）
            if not hasattr(self, '_parallel_ok'):
                try:
                    # 临时清空缓存以便序列化测试
                    old_cache = getattr(self.problem, '_cache', {})
                    if hasattr(self.problem, '_cache'):
                        self.problem._cache = {}
                    
                    pickle.dumps(self.problem.objective)
                    self._parallel_ok = True
                    
                    # 恢复缓存
                    if hasattr(self.problem, '_cache'):
                        self.problem._cache = old_cache
                except Exception as e:
                    self._parallel_ok = False
                    print(f"[并行] 序列化测试失败，将使用串行模式: {e}")
            
            if not self._parallel_ok:
                return [self.problem.objective(x) for x in solutions]
            
            if isinstance(pool, ProcessPoolExecutor):
                futures = [pool.submit(self.problem.objective, x) for x in solutions]
                return [f.result() for f in futures]
            elif isinstance(pool, mp.pool.Pool):
                return pool.map(self.problem.objective, solutions)
            else:
                return [self.problem.objective(x) for x in solutions]
        except (pickle.PicklingError, AttributeError, TypeError, ValueError) as e:
            # 如果目标函数不可序列化，回退到串行
            self._parallel_ok = False
            print(f"[并行] 执行失败，回退到串行: {e}")
            return [self.problem.objective(x) for x in solutions]
        except Exception as e:
            # 其他错误也回退到串行
            print(f"[并行] 执行失败，回退到串行: {e}")
            return [self.problem.objective(x) for x in solutions]
    
    def _evaluate_fitness(self, solutions: List[np.ndarray]) -> List[float]:
        """评估适应度（自动选择并行或串行）"""
        if not self.config.use_parallel or len(solutions) == 1:
            return [self.problem.objective(x) for x in solutions]
        
        pool = self._get_pool()
        try:
            fitness = self._evaluate_fitness_parallel(solutions, pool)
        except Exception as e:
            fitness = [self.problem.objective(x) for x in solutions]
        finally:
            self._close_pool(pool if not self.config.reuse_pool else None)
        
        return fitness
    
    def run(self, x0: Optional[np.ndarray] = None, 
            forward_fn_for_vis: Optional[Callable] = None):
        """
        运行优化并实时可视化（支持 BIPOP-CMA-ES：大小种群交替）
        
        参数:
        - x0: 初始点
        - forward_fn_for_vis: 用于可视化的前向函数（返回预测值）
        """
        cma = self._import_cma()
        
        if x0 is None:
            x0 = self.problem.bounds.get_initial_guess()
        
        sigma0 = self.config.sigma0
        if sigma0 is None:
            sigma0 = self.problem.bounds.get_initial_sigma()
        
        self.problem.reset_eval_count()
        
        # 初始化 BIPOP 参数
        default_popsize = self.config.popsize
        if self.config.use_bipop:
            self._initialize_bipop(self.problem.dim, default_popsize)
            current_popsize = self._bipop_state['current_popsize']
        else:
            current_popsize = default_popsize
        
        opts = {
            'maxiter': self.config.maxiter,
            'maxfevals': self.config.maxfevals,
            'tolx': self.config.tolx,
            'tolfun': self.config.tolfun,
            'bounds': [self.problem.bounds.lower.tolist(),
                       self.problem.bounds.upper.tolist()],
            'verbose': -9,  # 静默模式
            'seed': self.config.seed,
            'popsize': current_popsize
        }
        
        # 启动可视化
        if self.visualizer:
            self.visualizer.start()
        
        start_time = time.time()
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        x_history = []
        loss_history = []
        popsize_history = []
        best_loss = float('inf')
        x_best = x0.copy()
        iteration_count = 0  # 手动跟踪迭代次数
        
        # 计时统计
        timing_stats = {
            'iteration_times': [],
            'step_times': {
                'bipop_check': [],
                'ask': [],
                'evaluate': [],
                'tell': [],
                'history_update': [],
                'euclidean_calc': [],
                'visualization': [],
                'save_results': []
            },
            'individual_eval_times': []  # 每个个体的评估时间
        }
        
        try:
            while not es.stop():
                iter_start_time = time.time()
                step_times = {}
                
                # ========== 步骤1: BIPOP 检查/切换 ==========
                t0 = time.time()
                if self.config.use_bipop and iteration_count > 0:
                    if self._should_switch_population(iteration_count, best_loss):
                        current_x = es.result.xbest
                        current_sigma = es.sigma
                        
                        self._switch_population_size()
                        new_popsize = self._bipop_state['current_popsize']
                        
                        new_opts = opts.copy()
                        new_opts['popsize'] = new_popsize
                        
                        es = cma.CMAEvolutionStrategy(current_x, current_sigma, new_opts)
                step_times['bipop_check'] = time.time() - t0
                
                # ========== 步骤2: 生成候选解 ==========
                t0 = time.time()
                solutions = es.ask()
                step_times['ask'] = time.time() - t0
                
                # ========== 步骤3: 评估适应度（并行或串行）==========
                t0 = time.time()
                individual_times = []
                if self.config.use_parallel:
                    # 并行模式：记录总时间
                    fitness = self._evaluate_fitness(solutions)
                    avg_time = (time.time() - t0) / len(solutions)
                    individual_times = [avg_time] * len(solutions)
                else:
                    # 串行模式：记录每个个体的评估时间
                    fitness = []
                    for sol in solutions:
                        t_ind = time.time()
                        f = self.problem.objective(sol)
                        individual_times.append(time.time() - t_ind)
                        fitness.append(f)
                step_times['evaluate'] = time.time() - t0
                timing_stats['individual_eval_times'].append(individual_times)
                
                # ========== 步骤4: 更新优化器 ==========
                t0 = time.time()
                es.tell(solutions, fitness)
                step_times['tell'] = time.time() - t0
                
                # ========== 步骤5: 更新历史记录 ==========
                t0 = time.time()
                # 获取当前最优
                current_best_idx = np.argmin(fitness)
                current_best_loss = fitness[current_best_idx]
                current_best_x = solutions[current_best_idx]
                
                if current_best_loss < best_loss:
                    best_loss = current_best_loss
                    x_best = current_best_x.copy()
                
                x_history.append(x_best.copy())
                loss_history.append(current_best_loss)
                popsize_history.append(len(solutions))
                
                # 更新迭代计数
                iteration_count += 1
                
                # 更新 BIPOP 状态
                if self.config.use_bipop:
                    self._bipop_state['last_switch_iter'] += 1
                    self._bipop_state['last_best_loss'] = best_loss
                step_times['history_update'] = time.time() - t0
                
                # ========== 步骤6: 计算欧式距离统计（优化：使用缓存的预测值）==========
                t0 = time.time()
                euclidean_stats = None
                y_pred = None
                y_target = None
                
                if self.visualizer:
                    try:
                        y_target = self.visualizer.y_target
                        
                        # 尝试使用缓存的预测值（避免重复调用昂贵的前向模型）
                        y_pred = self.problem.get_cached_prediction(x_best)
                        
                        if y_pred is None and forward_fn_for_vis:
                            # 缓存未命中，需要调用前向模型
                            y_pred = forward_fn_for_vis(x_best)
                        
                        if y_pred is not None:
                            best_euc_dist = np.sqrt(np.sum((y_pred - y_target) ** 2))
                            
                            # 使用 fitness 值估算欧式距离统计（避免重复计算）
                            # 假设 loss ≈ MSE，则 euclidean_dist ≈ sqrt(loss * n_measurements)
                            n_meas = len(y_target)
                            estimated_distances = [np.sqrt(f * n_meas) if f > 0 else 0 for f in fitness]
                            
                            euclidean_stats = {
                                'best': float(best_euc_dist),
                                'worst': float(max(estimated_distances)),
                                'mean': float(np.mean(estimated_distances)),
                                'std': float(np.std(estimated_distances)),
                                'best_individual_euc': float(best_euc_dist)
                            }
                    except Exception as e:
                        pass
                step_times['euclidean_calc'] = time.time() - t0
                
                # ========== 步骤7: 更新可视化 ==========
                t0 = time.time()
                if self.visualizer:
                    self.visualizer.update_state(
                        iteration=iteration_count,
                        loss=current_best_loss,
                        best_loss=best_loss,
                        sigma=es.sigma,
                        y_pred=y_pred,
                        x_best=x_best,
                        eval_count=self.problem._eval_count,
                        euclidean_stats=euclidean_stats
                    )
                    
                    # 刷新显示
                    plt.pause(0.01)
                step_times['visualization'] = time.time() - t0
                
                # ========== 步骤8: 保存迭代结果到JSON ==========
                if self.result_saver:
                    elapsed = time.time() - start_time
                    self.result_saver.save_iteration(
                        iteration=iteration_count,
                        x_best=x_best,
                        loss=current_best_loss,
                        best_loss=best_loss,
                        sigma=es.sigma,
                        eval_count=self.problem._eval_count,
                        elapsed_time=elapsed,
                        y_pred=y_pred,
                        y_target=y_target,
                        euclidean_stats=euclidean_stats
                    )
                step_times['save_results'] = time.time() - t0
                
                # ========== 计算迭代总时间并记录统计 ==========
                iter_total_time = time.time() - iter_start_time
                timing_stats['iteration_times'].append(iter_total_time)
                for key in step_times:
                    timing_stats['step_times'][key].append(step_times[key])
                
                # ========== 打印详细计时信息 ==========
                pop_size = len(solutions)
                eval_time = step_times['evaluate']
                avg_eval_time = eval_time / pop_size if pop_size > 0 else 0
                
                # 每次迭代都打印详细计时
                print(f"\n{'='*70}")
                print(f"迭代 {iteration_count} | Loss={current_best_loss:.4e} | Best={best_loss:.4e} | σ={es.sigma:.4f}")
                print(f"{'='*70}")
                print(f"  总用时: {iter_total_time*1000:.1f}ms | 种群大小: {pop_size}")
                print(f"  ┌─ 关键步骤用时:")
                print(f"  │  [1] BIPOP检查:    {step_times['bipop_check']*1000:>8.2f}ms")
                print(f"  │  [2] 生成候选解:   {step_times['ask']*1000:>8.2f}ms")
                print(f"  │  [3] 适应度评估:   {step_times['evaluate']*1000:>8.2f}ms ({avg_eval_time*1000:.2f}ms/个体)")
                print(f"  │  [4] 优化器更新:   {step_times['tell']*1000:>8.2f}ms")
                print(f"  │  [5] 历史记录:     {step_times['history_update']*1000:>8.2f}ms")
                print(f"  │  [6] 欧式距离:     {step_times['euclidean_calc']*1000:>8.2f}ms")
                print(f"  │  [7] 可视化更新:   {step_times['visualization']*1000:>8.2f}ms")
                print(f"  └─ [8] 结果保存:     {step_times['save_results']*1000:>8.2f}ms")
                
                # 显示个体评估时间统计（如果是串行模式）
                if not self.config.use_parallel and individual_times:
                    min_t = min(individual_times) * 1000
                    max_t = max(individual_times) * 1000
                    avg_t = sum(individual_times) / len(individual_times) * 1000
                    print(f"  ├─ 个体评估时间: min={min_t:.2f}ms, max={max_t:.2f}ms, avg={avg_t:.2f}ms")
        
        except KeyboardInterrupt:
            print("\n优化被用户中断")
        
        finally:
            elapsed_time = time.time() - start_time
            
            # 保存最终图表
            if self.visualizer:
                self.visualizer.save("optimization_final.png")
        
        # ========== 打印计时统计汇总 ==========
        if timing_stats['iteration_times']:
            print("\n" + "="*70)
            print("计时统计汇总")
            print("="*70)
            
            n_iters = len(timing_stats['iteration_times'])
            total_iter_time = sum(timing_stats['iteration_times'])
            avg_iter_time = total_iter_time / n_iters
            
            print(f"总迭代次数: {n_iters}")
            print(f"总运行时间: {elapsed_time:.2f}s")
            print(f"迭代总用时: {total_iter_time:.2f}s")
            print(f"平均每迭代: {avg_iter_time*1000:.1f}ms")
            
            print(f"\n各步骤平均用时:")
            step_names = {
                'bipop_check': 'BIPOP检查',
                'ask': '生成候选解',
                'evaluate': '适应度评估',
                'tell': '优化器更新',
                'history_update': '历史记录',
                'euclidean_calc': '欧式距离',
                'visualization': '可视化更新',
                'save_results': '结果保存'
            }
            
            total_step_time = 0
            for key, name in step_names.items():
                times = timing_stats['step_times'][key]
                if times:
                    avg_time = sum(times) / len(times) * 1000
                    total_time = sum(times) * 1000
                    pct = (sum(times) / total_iter_time * 100) if total_iter_time > 0 else 0
                    total_step_time += sum(times)
                    print(f"  {name:12s}: 平均={avg_time:>8.2f}ms, 总计={total_time:>8.1f}ms ({pct:>5.1f}%)")
            
            # 显示个体评估时间统计
            if timing_stats['individual_eval_times']:
                all_times = [t for times in timing_stats['individual_eval_times'] for t in times]
                if all_times:
                    print(f"\n个体评估时间统计 (共{len(all_times)}次评估):")
                    print(f"  最小: {min(all_times)*1000:.2f}ms")
                    print(f"  最大: {max(all_times)*1000:.2f}ms")
                    print(f"  平均: {sum(all_times)/len(all_times)*1000:.2f}ms")
                    print(f"  总计: {sum(all_times):.2f}s")
            
            print("="*70)
        
        # 构建结果
        from ventilation_inversion import InversionResult
        
        # 计算种群切换次数
        bipop_switches = 0
        if self.config.use_bipop and len(popsize_history) > 1:
            bipop_switches = len([i for i in range(1, len(popsize_history)) 
                                  if popsize_history[i] != popsize_history[i-1]])
        
        result = InversionResult(
            x_best=x_best,
            loss_best=best_loss,
            x_history=x_history,
            loss_history=loss_history,
            n_evaluations=self.problem._eval_count,
            convergence_info={
                'stop_conditions': es.stop(),
                'iterations': iteration_count,
                'final_sigma': es.sigma,
                'popsize_history': popsize_history if self.config.use_bipop else None,
                'bipop_switches': bipop_switches
            },
            elapsed_time=elapsed_time
        )
        
        # 保存最终汇总结果
        if self.result_saver:
            self.result_saver.save_final_summary(
                result=result,
                total_iterations=iteration_count
            )
        
        # 清理进程池
        if self.config.reuse_pool and self._pool is not None:
            if isinstance(self._pool, ProcessPoolExecutor):
                self._pool.shutdown(wait=True)
            elif isinstance(self._pool, mp.pool.Pool):
                self._pool.close()
                self._pool.join()
            self._pool = None
        
        return result


def run_with_live_visualization(
    json_path: str = "input.json",
    max_iter: int = 50,
    use_mvn_solver: bool = True,
    optimize_fan: bool = False,
    save_results: bool = True,
    save_every: int = 1,
    output_dir: str = "results",
    enable_bipop: bool = True,
    enable_parallel: bool = True,
    n_workers: Optional[int] = None,
    pop_large_ratio: float = 2.0,
    pop_small_ratio: float = 0.5
):
    """
    运行带实时可视化的反演
    
    参数:
    - json_path: 数据文件路径
    - max_iter: 最大迭代次数
    - use_mvn_solver: 是否使用MVN Solver
    - optimize_fan: 是否同时优化风机压力 (fanHs)
    - save_results: 是否保存每次迭代结果
    - save_every: 每隔多少次迭代保存一次
    - output_dir: 结果输出目录
    """
    from real_data_inversion import (
        DataLoader, RealDataInversionConfig,
        MVNSolverWrapper, SimplifiedForwardModel,
        HuberLoss
    )
    from ventilation_inversion import InversionBounds, CMAESConfig
    
    print("="*70)
    print("CMA-ES Inversion with Live Visualization")
    print("="*70)
    
    # 加载数据
    print("\n[1] Loading data...")
    network = DataLoader.load(json_path)
    
    # 显示风机信息
    variable_fans = [f for f in network.fans if f.min_h != f.max_h]
    print(f"  Total fans: {len(network.fans)}, Variable fans: {len(variable_fans)}")
    
    # 创建前向模型
    print("\n[2] Creating forward model...")
    if use_mvn_solver:
        try:
            forward_model = MVNSolverWrapper(
                network=network,
                json_path=json_path
            )
            print("  Using MVN Solver")
        except ImportError:
            print("  MVN Solver not available, using simplified model")
            forward_model = SimplifiedForwardModel(network)
    else:
        forward_model = SimplifiedForwardModel(network)
        print("  Using simplified model")
    
    # 配置反演
    print("\n[3] Configuring inversion problem...")
    if optimize_fan:
        print(f"  Fan pressure optimization: ENABLED ({len(variable_fans)} variable fans)")
    else:
        print("  Fan pressure optimization: DISABLED")
    
    inv_config = RealDataInversionConfig(
        network_data=network,
        forward_model=forward_model,
        optimize_fan_pressure=optimize_fan,
        loss_fn=HuberLoss(delta=1.0),
        use_log_scale=True
    )
    
    problem = inv_config.create_inversion_problem()
    
    # CMA-ES 特性
    print("\n[4] Configuring CMA-ES features...")
    print(f"  BIPOP alternating population: {'ENABLED' if enable_bipop else 'DISABLED'} "
          f"(large ratio={pop_large_ratio}, small ratio={pop_small_ratio})")
    if enable_parallel:
        worker_msg = n_workers if n_workers is not None else f"auto ({max(1, mp.cpu_count()-1)} cores)"
        print(f"  Parallel fitness evaluation: ENABLED (workers={worker_msg})")
    else:
        print("  Parallel fitness evaluation: DISABLED")
    
    # 创建可视化器
    print("\n[5] Initializing visualizer...")
    visualizer = LiveVisualizer(
        y_target=inv_config.target_values,
        update_interval=0.5
    )
    
    # 创建结果保存器
    result_saver = None
    if save_results:
        print(f"\n[6] Setting up result saver (save every {save_every} iterations)...")
        result_saver = IterationResultSaver(
            output_dir=output_dir,
            save_every=save_every,
            network_data=network,
            inv_config=inv_config
        )
    
    # 创建用于可视化的前向函数
    def forward_for_vis(x):
        R, H = inv_config.decode_parameters(x)
        Q_all = forward_model(R, H)
        return Q_all[inv_config.measurement_indices]
    
    # 配置 CMA-ES
    cma_config = CMAESConfig(
        sigma0=0.3,
        maxiter=max_iter,
        maxfevals=max_iter * 50,
        verbose=0,
        seed=42,
        use_bipop=enable_bipop,
        popsize_large_ratio=pop_large_ratio,
        popsize_small_ratio=pop_small_ratio,
        use_parallel=enable_parallel,
        n_workers=n_workers
    )
    
    # 创建优化器
    optimizer = LiveCMAESOptimizer(
        problem, 
        cma_config, 
        visualizer,
        result_saver=result_saver
    )
    
    # 运行优化
    print(f"\n[7] Starting optimization (max {max_iter} iterations)...")
    print("    Press Ctrl+C to interrupt\n")
    
    x0 = inv_config.get_initial_guess()
    result = optimizer.run(x0, forward_fn_for_vis=forward_for_vis)
    
    # 显示结果
    print("\n" + "="*70)
    print("反演完成")
    print("="*70)
    print(f"最终损失: {result.loss_best:.6e}")
    print(f"迭代次数: {result.convergence_info['iterations']}")
    print(f"函数评估: {result.n_evaluations}")
    print(f"运行时间: {result.elapsed_time:.2f}秒")
    
    # 显示最终图表
    print("\n显示最终结果图表...")
    visualizer.show_final()
    
    return result, inv_config, network


if __name__ == "__main__":
    import sys
    
    max_iter = 50
    use_mvn = False
    optimize_fan = False
    save_every = 1
    no_save = False
    enable_bipop = True
    enable_parallel = True
    n_workers = None
    pop_large_ratio = 2.0
    pop_small_ratio = 0.5
    
    for arg in sys.argv[1:]:
        if arg == "--mvn":
            use_mvn = True
        elif arg == "--optimize-fan":
            optimize_fan = True
        elif arg.startswith("--iter="):
            max_iter = int(arg.split("=")[1])
        elif arg.startswith("--save-every="):
            save_every = int(arg.split("=")[1])
        elif arg == "--no-save":
            no_save = True
        elif arg == "--no-bipop":
            enable_bipop = False
        elif arg == "--no-parallel":
            enable_parallel = False
        elif arg.startswith("--workers="):
            n_workers = int(arg.split("=")[1])
        elif arg.startswith("--pop-large="):
            pop_large_ratio = float(arg.split("=")[1])
        elif arg.startswith("--pop-small="):
            pop_small_ratio = float(arg.split("=")[1])
    
    print("\nCommand line options:")
    print(f"  --mvn: {use_mvn}")
    print(f"  --optimize-fan: {optimize_fan}")
    print(f"  --iter: {max_iter}")
    print(f"  --save-every: {save_every}")
    print(f"  --no-bipop: {not enable_bipop}")
    print(f"  --no-parallel: {not enable_parallel}")
    print(f"  --workers: {n_workers if n_workers is not None else 'auto'}")
    print(f"  --pop-large: {pop_large_ratio}")
    print(f"  --pop-small: {pop_small_ratio}")
    print()
    
    result, config, network = run_with_live_visualization(
        max_iter=max_iter,
        use_mvn_solver=use_mvn,
        optimize_fan=optimize_fan,
        save_results=not no_save,
        save_every=save_every,
        enable_bipop=enable_bipop,
        enable_parallel=enable_parallel,
        n_workers=n_workers,
        pop_large_ratio=pop_large_ratio,
        pop_small_ratio=pop_small_ratio
    )

