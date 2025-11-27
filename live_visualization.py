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
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle('CMA-ES Inversion Optimization Monitor', fontsize=14, fontweight='bold')
        
        # 子图1: 损失曲线
        self.ax_loss = self.axes[0, 0]
        self.ax_loss.set_xlabel('Iteration')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.set_title('Loss Convergence')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='Current')
        self.line_best, = self.ax_loss.plot([], [], 'r--', linewidth=1.5, label='Best')
        self.ax_loss.legend(loc='upper right')
        
        # 子图2: Sigma 变化
        self.ax_sigma = self.axes[0, 1]
        self.ax_sigma.set_xlabel('Iteration')
        self.ax_sigma.set_ylabel('Sigma (Step Size)')
        self.ax_sigma.set_title('CMA-ES Step Size')
        self.ax_sigma.grid(True, alpha=0.3)
        self.line_sigma, = self.ax_sigma.plot([], [], 'g-', linewidth=2)
        
        # 子图3: 残差分布
        self.ax_residual = self.axes[1, 0]
        self.ax_residual.set_xlabel('Measurement Index')
        self.ax_residual.set_ylabel('Residual (Pred - Target)')
        self.ax_residual.set_title('Residual Distribution')
        self.ax_residual.axhline(y=0, color='black', linestyle='-', linewidth=1)
        self.ax_residual.grid(True, alpha=0.3)
        
        # 子图4: 预测 vs 目标
        self.ax_scatter = self.axes[1, 1]
        self.ax_scatter.set_xlabel('Target Q (m3/s)')
        self.ax_scatter.set_ylabel('Predicted Q (m3/s)')
        self.ax_scatter.set_title('Predicted vs Target')
        self.ax_scatter.grid(True, alpha=0.3)
        
        # 状态文本
        self.status_text = self.fig.text(
            0.02, 0.02, '', fontsize=10, 
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    def update_state(self, 
                     iteration: int,
                     loss: float,
                     best_loss: float,
                     sigma: float,
                     y_pred: Optional[np.ndarray] = None,
                     x_best: Optional[np.ndarray] = None,
                     eval_count: int = 0):
        """
        更新优化状态（由优化器回调）
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
                
                status = (
                    f"Iter: {self.state.iteration:4d} | "
                    f"Loss: {self.state.loss_history[-1]:.4e} | "
                    f"Best: {min(self.state.best_loss_history):.4e} | "
                    f"Sigma: {self.state.sigma_history[-1]:.4f} | "
                    f"Evals: {self.state.eval_count:5d} | "
                    f"RMSE: {rmse:.3f} | "
                    f"RelErr: {rel_err:.1f}% | "
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
                       y_target: Optional[np.ndarray] = None):
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
        
        # 添加预测误差统计
        if y_pred is not None and y_target is not None:
            residuals = y_pred - y_target
            result_data["metrics"]["rmse"] = float(np.sqrt(np.mean(residuals**2)))
            result_data["metrics"]["mae"] = float(np.mean(np.abs(residuals)))
            result_data["metrics"]["max_error"] = float(np.max(np.abs(residuals)))
            result_data["metrics"]["relative_error_percent"] = float(
                np.mean(np.abs(residuals) / (np.abs(y_target) + 0.1)) * 100
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
    带实时可视化的 CMA-ES 优化器
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
    
    def _import_cma(self):
        if self._cma is None:
            import cma
            self._cma = cma
        return self._cma
    
    def run(self, x0: Optional[np.ndarray] = None, 
            forward_fn_for_vis: Optional[Callable] = None):
        """
        运行优化并实时可视化
        
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
        
        opts = {
            'maxiter': self.config.maxiter,
            'maxfevals': self.config.maxfevals,
            'tolx': self.config.tolx,
            'tolfun': self.config.tolfun,
            'bounds': [self.problem.bounds.lower.tolist(),
                       self.problem.bounds.upper.tolist()],
            'verbose': -9,  # 静默模式
            'seed': self.config.seed
        }
        
        if self.config.popsize:
            opts['popsize'] = self.config.popsize
        
        # 启动可视化
        if self.visualizer:
            self.visualizer.start()
        
        start_time = time.time()
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        
        x_history = []
        loss_history = []
        best_loss = float('inf')
        x_best = x0.copy()
        
        try:
            while not es.stop():
                solutions = es.ask()
                fitness = [self.problem.objective(x) for x in solutions]
                es.tell(solutions, fitness)
                
                # 获取当前最优
                current_best_idx = np.argmin(fitness)
                current_best_loss = fitness[current_best_idx]
                current_best_x = solutions[current_best_idx]
                
                if current_best_loss < best_loss:
                    best_loss = current_best_loss
                    x_best = current_best_x.copy()
                
                x_history.append(x_best.copy())
                loss_history.append(current_best_loss)
                
                # 计算预测值（用于可视化和保存）
                y_pred = None
                y_target = None
                if forward_fn_for_vis:
                    try:
                        y_pred = forward_fn_for_vis(x_best)
                        y_target = self.visualizer.y_target if self.visualizer else None
                    except:
                        pass
                
                # 更新可视化
                if self.visualizer:
                    self.visualizer.update_state(
                        iteration=es.countiter,
                        loss=current_best_loss,
                        best_loss=best_loss,
                        sigma=es.sigma,
                        y_pred=y_pred,
                        x_best=x_best,
                        eval_count=self.problem._eval_count
                    )
                    
                    # 刷新显示
                    plt.pause(0.01)
                
                # 保存迭代结果到JSON
                if self.result_saver:
                    elapsed = time.time() - start_time
                    self.result_saver.save_iteration(
                        iteration=es.countiter,
                        x_best=x_best,
                        loss=current_best_loss,
                        best_loss=best_loss,
                        sigma=es.sigma,
                        eval_count=self.problem._eval_count,
                        elapsed_time=elapsed,
                        y_pred=y_pred,
                        y_target=y_target
                    )
                
                # 打印进度
                if es.countiter % 10 == 0:
                    print(f"Iter {es.countiter}: Loss={current_best_loss:.4e}, "
                          f"Best={best_loss:.4e}, Sigma={es.sigma:.4f}")
        
        except KeyboardInterrupt:
            print("\n优化被用户中断")
        
        finally:
            elapsed_time = time.time() - start_time
            
            # 保存最终图表
            if self.visualizer:
                self.visualizer.save("optimization_final.png")
        
        # 构建结果
        from ventilation_inversion import InversionResult
        
        result = InversionResult(
            x_best=x_best,
            loss_best=best_loss,
            x_history=x_history,
            loss_history=loss_history,
            n_evaluations=self.problem._eval_count,
            convergence_info={
                'stop_conditions': es.stop(),
                'iterations': es.countiter,
                'final_sigma': es.sigma
            },
            elapsed_time=elapsed_time
        )
        
        # 保存最终汇总结果
        if self.result_saver:
            self.result_saver.save_final_summary(
                result=result,
                total_iterations=es.countiter
            )
        
        return result


def run_with_live_visualization(
    json_path: str = "input.json",
    max_iter: int = 50,
    use_mvn_solver: bool = True,
    save_results: bool = True,
    save_every: int = 1,
    output_dir: str = "results"
):
    """
    运行带实时可视化的反演
    
    参数:
    - json_path: 数据文件路径
    - max_iter: 最大迭代次数
    - use_mvn_solver: 是否使用MVN Solver
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
    inv_config = RealDataInversionConfig(
        network_data=network,
        forward_model=forward_model,
        loss_fn=HuberLoss(delta=1.0),
        use_log_scale=True
    )
    
    problem = inv_config.create_inversion_problem()
    
    # 创建可视化器
    print("\n[4] Initializing visualizer...")
    visualizer = LiveVisualizer(
        y_target=inv_config.target_values,
        update_interval=0.5
    )
    
    # 创建结果保存器
    result_saver = None
    if save_results:
        print(f"\n[5] Setting up result saver (save every {save_every} iterations)...")
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
        seed=42
    )
    
    # 创建优化器
    optimizer = LiveCMAESOptimizer(
        problem, 
        cma_config, 
        visualizer,
        result_saver=result_saver
    )
    
    # 运行优化
    print(f"\n[6] Starting optimization (max {max_iter} iterations)...")
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
    save_every = 1
    no_save = False
    
    for arg in sys.argv[1:]:
        if arg == "--mvn":
            use_mvn = True
        elif arg.startswith("--iter="):
            max_iter = int(arg.split("=")[1])
        elif arg.startswith("--save-every="):
            save_every = int(arg.split("=")[1])
        elif arg == "--no-save":
            no_save = True
    
    result, config, network = run_with_live_visualization(
        max_iter=max_iter,
        use_mvn_solver=use_mvn,
        save_results=not no_save,
        save_every=save_every
    )

