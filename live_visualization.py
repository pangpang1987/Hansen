"""
矿井通风阻力系数反演 - 实时可视化模块

在 CMA-ES 优化过程中实时显示关键指标
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Optional, Callable
import time
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
        self.fig.suptitle('CMA-ES 反演优化实时监控', fontsize=14, fontweight='bold')
        
        # 子图1: 损失曲线
        self.ax_loss = self.axes[0, 0]
        self.ax_loss.set_xlabel('迭代次数')
        self.ax_loss.set_ylabel('损失值')
        self.ax_loss.set_title('损失函数收敛曲线')
        self.ax_loss.set_yscale('log')
        self.ax_loss.grid(True, alpha=0.3)
        self.line_loss, = self.ax_loss.plot([], [], 'b-', linewidth=2, label='当前损失')
        self.line_best, = self.ax_loss.plot([], [], 'r--', linewidth=1.5, label='最佳损失')
        self.ax_loss.legend(loc='upper right')
        
        # 子图2: Sigma 变化
        self.ax_sigma = self.axes[0, 1]
        self.ax_sigma.set_xlabel('迭代次数')
        self.ax_sigma.set_ylabel('σ (步长)')
        self.ax_sigma.set_title('CMA-ES 步长变化')
        self.ax_sigma.grid(True, alpha=0.3)
        self.line_sigma, = self.ax_sigma.plot([], [], 'g-', linewidth=2)
        
        # 子图3: 残差分布
        self.ax_residual = self.axes[1, 0]
        self.ax_residual.set_xlabel('测点索引')
        self.ax_residual.set_ylabel('残差 (预测 - 目标)')
        self.ax_residual.set_title('测点残差分布')
        self.ax_residual.axhline(y=0, color='black', linestyle='-', linewidth=1)
        self.ax_residual.grid(True, alpha=0.3)
        
        # 子图4: 预测 vs 目标
        self.ax_scatter = self.axes[1, 1]
        self.ax_scatter.set_xlabel('目标值 (m³/s)')
        self.ax_scatter.set_ylabel('预测值 (m³/s)')
        self.ax_scatter.set_title('预测值 vs 目标值')
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
                self.ax_residual.set_xlabel('测点索引')
                self.ax_residual.set_ylabel('残差 (预测 - 目标)')
                self.ax_residual.set_title('测点残差分布')
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
                self.ax_scatter.set_xlabel('目标值 (m³/s)')
                self.ax_scatter.set_ylabel('预测值 (m³/s)')
                self.ax_scatter.set_title('预测值 vs 目标值')
                self.ax_scatter.grid(True, alpha=0.3)
                
                y_pred = self.state.current_y_pred
                y_target = self.state.current_y_target
                
                self.ax_scatter.scatter(y_target, y_pred, c='blue', alpha=0.5, s=20)
                
                # 绘制理想对角线
                min_val = min(y_target.min(), y_pred.min())
                max_val = max(y_target.max(), y_pred.max())
                self.ax_scatter.plot([min_val, max_val], [min_val, max_val], 
                                    'r--', linewidth=2, label='理想拟合')
                self.ax_scatter.legend()
            
            # 更新状态文本
            if len(self.state.loss_history) > 0:
                rmse = np.sqrt(np.mean(self.state.current_residuals**2)) if self.state.current_residuals is not None else 0
                rel_err = np.mean(np.abs(self.state.current_residuals) / (np.abs(self.y_target) + 0.1)) * 100 if self.state.current_residuals is not None else 0
                
                status = (
                    f"迭代: {self.state.iteration:4d} | "
                    f"损失: {self.state.loss_history[-1]:.4e} | "
                    f"最佳: {min(self.state.best_loss_history):.4e} | "
                    f"σ: {self.state.sigma_history[-1]:.4f} | "
                    f"评估: {self.state.eval_count:5d} | "
                    f"RMSE: {rmse:.3f} | "
                    f"相对误差: {rel_err:.1f}% | "
                    f"时间: {self.state.elapsed_time:.1f}s"
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


class LiveCMAESOptimizer:
    """
    带实时可视化的 CMA-ES 优化器
    """
    
    def __init__(self, 
                 problem,
                 config,
                 visualizer: Optional[LiveVisualizer] = None):
        """
        参数:
        - problem: InversionProblem 实例
        - config: CMAESConfig 配置
        - visualizer: LiveVisualizer 实例（可选）
        """
        self.problem = problem
        self.config = config
        self.visualizer = visualizer
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
                
                # 更新可视化
                if self.visualizer:
                    y_pred = None
                    if forward_fn_for_vis:
                        try:
                            y_pred = forward_fn_for_vis(x_best)
                        except:
                            pass
                    
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
                
                # 打印进度
                if es.countiter % 10 == 0:
                    print(f"迭代 {es.countiter}: 损失={current_best_loss:.4e}, "
                          f"最佳={best_loss:.4e}, σ={es.sigma:.4f}")
        
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
        
        return result


def run_with_live_visualization(
    json_path: str = "input.json",
    max_iter: int = 50,
    use_mvn_solver: bool = True
):
    """
    运行带实时可视化的反演
    """
    from real_data_inversion import (
        DataLoader, RealDataInversionConfig,
        MVNSolverWrapper, SimplifiedForwardModel,
        HuberLoss
    )
    from ventilation_inversion import InversionBounds, CMAESConfig
    
    print("="*70)
    print("矿井通风阻力系数反演 - 实时可视化模式")
    print("="*70)
    
    # 加载数据
    print("\n[1] 加载数据...")
    network = DataLoader.load(json_path)
    
    # 创建前向模型
    print("\n[2] 创建前向模型...")
    if use_mvn_solver:
        try:
            forward_model = MVNSolverWrapper(
                network=network,
                json_path=json_path
            )
            print("  使用 MVN Solver")
        except ImportError:
            print("  MVN Solver 不可用，使用简化模型")
            forward_model = SimplifiedForwardModel(network)
    else:
        forward_model = SimplifiedForwardModel(network)
        print("  使用简化模型")
    
    # 配置反演
    print("\n[3] 配置反演问题...")
    inv_config = RealDataInversionConfig(
        network_data=network,
        forward_model=forward_model,
        loss_fn=HuberLoss(delta=1.0),
        use_log_scale=True
    )
    
    problem = inv_config.create_inversion_problem()
    
    # 创建可视化器
    print("\n[4] 初始化可视化器...")
    visualizer = LiveVisualizer(
        y_target=inv_config.target_values,
        update_interval=0.5
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
    optimizer = LiveCMAESOptimizer(problem, cma_config, visualizer)
    
    # 运行优化
    print("\n[5] 开始优化（实时可视化）...")
    print("    按 Ctrl+C 可中断优化\n")
    
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
    
    for arg in sys.argv[1:]:
        if arg == "--mvn":
            use_mvn = True
        elif arg.startswith("--iter="):
            max_iter = int(arg.split("=")[1])
    
    result, config, network = run_with_live_visualization(
        max_iter=max_iter,
        use_mvn_solver=use_mvn
    )

