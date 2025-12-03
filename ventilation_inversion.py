"""
矿井通风网络阻力系数反演系统
使用 CMA-ES (Covariance Matrix Adaptation Evolution Strategy) 求解

作者: 高级研究员
版本: 1.0.0
描述: 针对矿井通风网络的阻力系数反演问题，提供完整的优化框架
"""

import numpy as np
from typing import Callable, List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 注意：由于 forward_fn 可能是闭包，无法被 pickle 序列化
# 并行计算会在检测到序列化失败时自动回退到串行模式


# ============================================================================
# 第一部分: 损失函数定义
# ============================================================================

class LossFunction(ABC):
    """损失函数抽象基类"""
    
    @abstractmethod
    def __call__(self, y_pred: np.ndarray, y_target: np.ndarray, 
                 weights: Optional[np.ndarray] = None) -> float:
        """计算损失值"""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """返回损失函数名称"""
        pass


class MSELoss(LossFunction):
    """
    均方误差损失 (Mean Squared Error)
    
    公式: L = (1/n) * Σ w_i * (y_pred_i - y_target_i)²
    
    适用场景:
    - 测量噪声为高斯分布
    - 对大误差敏感，需要严格拟合
    - 数据质量较高，无明显异常值
    """
    
    def __call__(self, y_pred: np.ndarray, y_target: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
        residuals = y_pred - y_target
        squared_residuals = residuals ** 2
        
        if weights is not None:
            weighted_squared = weights * squared_residuals
            return np.mean(weighted_squared)
        return np.mean(squared_residuals)
    
    def name(self) -> str:
        return "MSE"


class MAELoss(LossFunction):
    """
    平均绝对误差损失 (Mean Absolute Error)
    
    公式: L = (1/n) * Σ w_i * |y_pred_i - y_target_i|
    
    适用场景:
    - 数据存在异常值/离群点
    - 需要对所有误差等权处理
    - 对大误差不过度惩罚
    """
    
    def __call__(self, y_pred: np.ndarray, y_target: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
        residuals = np.abs(y_pred - y_target)
        
        if weights is not None:
            weighted_residuals = weights * residuals
            return np.mean(weighted_residuals)
        return np.mean(residuals)
    
    def name(self) -> str:
        return "MAE"


class HuberLoss(LossFunction):
    """
    Huber 损失函数 (推荐默认使用)
    
    公式:
    L = 0.5 * r² ,           当 |r| ≤ δ
    L = δ * (|r| - 0.5*δ) ,  当 |r| > δ
    
    其中 r = y_pred - y_target, δ 为阈值参数
    
    适用场景:
    - 矿井通风测量数据（推荐）
    - 数据可能存在测量误差和少量异常值
    - 兼顾 MSE 的平滑性和 MAE 的鲁棒性
    
    参数:
    - delta: 阈值参数，控制 MSE 与 MAE 的过渡点
             较小的 delta 更接近 MAE（更鲁棒）
             较大的 delta 更接近 MSE（更平滑）
    """
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    
    def __call__(self, y_pred: np.ndarray, y_target: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
        residuals = y_pred - y_target
        abs_residuals = np.abs(residuals)
        
        # 分段计算
        quadratic = 0.5 * residuals ** 2
        linear = self.delta * (abs_residuals - 0.5 * self.delta)
        
        # 条件选择
        loss = np.where(abs_residuals <= self.delta, quadratic, linear)
        
        if weights is not None:
            weighted_loss = weights * loss
            return np.mean(weighted_loss)
        return np.mean(loss)
    
    def name(self) -> str:
        return f"Huber(δ={self.delta})"


class WeightedCompositeLoss(LossFunction):
    """
    加权复合损失函数
    
    用于多测点误差合并，支持：
    - 不同测点的重要性权重
    - 不同物理量（风速、风量、风压）的归一化
    """
    
    def __init__(self, base_loss: LossFunction, 
                 measurement_weights: Optional[np.ndarray] = None,
                 normalization_factors: Optional[np.ndarray] = None):
        """
        参数:
        - base_loss: 基础损失函数
        - measurement_weights: 测点重要性权重
        - normalization_factors: 各测点的归一化因子（用于量纲统一）
        """
        self.base_loss = base_loss
        self.measurement_weights = measurement_weights
        self.normalization_factors = normalization_factors
    
    def __call__(self, y_pred: np.ndarray, y_target: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> float:
        # 归一化处理
        if self.normalization_factors is not None:
            y_pred_norm = y_pred / self.normalization_factors
            y_target_norm = y_target / self.normalization_factors
        else:
            y_pred_norm = y_pred
            y_target_norm = y_target
        
        # 应用测点权重
        effective_weights = self.measurement_weights if weights is None else weights
        
        return self.base_loss(y_pred_norm, y_target_norm, effective_weights)
    
    def name(self) -> str:
        return f"Weighted_{self.base_loss.name()}"


# ============================================================================
# 第二部分: 反演问题定义
# ============================================================================

@dataclass
class InversionBounds:
    """参数边界定义"""
    lower: np.ndarray  # 下界向量
    upper: np.ndarray  # 上界向量
    
    def __post_init__(self):
        assert len(self.lower) == len(self.upper), "边界维度不匹配"
        assert np.all(self.lower < self.upper), "下界必须小于上界"
    
    @property
    def dim(self) -> int:
        return len(self.lower)
    
    def clip(self, x: np.ndarray) -> np.ndarray:
        """将参数裁剪到边界内"""
        return np.clip(x, self.lower, self.upper)
    
    def is_within(self, x: np.ndarray) -> bool:
        """检查参数是否在边界内"""
        return np.all(x >= self.lower) and np.all(x <= self.upper)
    
    def get_initial_guess(self) -> np.ndarray:
        """返回边界中心作为初始猜测"""
        return 0.5 * (self.lower + self.upper)
    
    def get_initial_sigma(self) -> float:
        """返回建议的初始步长"""
        range_width = self.upper - self.lower
        return 0.3 * np.mean(range_width)


@dataclass
class InversionResult:
    """反演结果数据结构"""
    x_best: np.ndarray          # 最优参数向量
    loss_best: float            # 最优损失值
    x_history: List[np.ndarray] # 参数历史
    loss_history: List[float]   # 损失历史
    n_evaluations: int          # 函数评估次数
    convergence_info: Dict      # 收敛信息
    elapsed_time: float         # 运行时间(秒)
    
    def get_summary(self) -> str:
        """获取结果摘要"""
        return (
            f"\n{'='*60}\n"
            f"反演结果摘要\n"
            f"{'='*60}\n"
            f"最优损失值: {self.loss_best:.6e}\n"
            f"函数评估次数: {self.n_evaluations}\n"
            f"运行时间: {self.elapsed_time:.2f}秒\n"
            f"最优参数:\n{self.x_best}\n"
            f"{'='*60}"
        )


class InversionProblem:
    """
    通风反演问题定义类
    
    封装了反演问题的所有要素：
    - 参数维度和边界
    - 前向模型接口
    - 损失函数
    - 观测数据
    """
    
    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        y_target: np.ndarray,
        bounds: InversionBounds,
        loss_fn: Optional[LossFunction] = None,
        measurement_weights: Optional[np.ndarray] = None,
        param_names: Optional[List[str]] = None
    ):
        """
        参数:
        - forward_fn: 前向模型函数 x -> y_pred
        - y_target: 观测目标值
        - bounds: 参数边界
        - loss_fn: 损失函数（默认 Huber Loss）
        - measurement_weights: 测点权重
        - param_names: 参数名称列表（用于日志）
        """
        self.forward_fn = forward_fn
        self.y_target = np.asarray(y_target)
        self.bounds = bounds
        self.loss_fn = loss_fn if loss_fn is not None else HuberLoss(delta=1.0)
        self.measurement_weights = measurement_weights
        self.param_names = param_names or [f"k_{i}" for i in range(bounds.dim)]
        
        # 评估计数器
        self._eval_count = 0
        
        # 缓存评估结果（用于避免重复调用前向模型）
        self._cache = {}  # key: x的hash, value: (x, y_pred, loss)
        self._cache_max_size = 200  # 最大缓存数量
    
    @property
    def dim(self) -> int:
        """参数维度"""
        return self.bounds.dim
    
    @property
    def n_measurements(self) -> int:
        """测量点数量"""
        return len(self.y_target)
    
    def reset_eval_count(self):
        """重置评估计数器"""
        self._eval_count = 0
    
    def objective(self, x: np.ndarray) -> float:
        """
        目标函数（用于优化器调用）
        
        参数:
        - x: 待评估的参数向量
        
        返回:
        - 损失值（越小越好）
        """
        self._eval_count += 1
        
        # 边界约束处理（惩罚法）
        if not self.bounds.is_within(x):
            # 计算越界惩罚
            violation = np.sum(np.maximum(0, self.bounds.lower - x) ** 2)
            violation += np.sum(np.maximum(0, x - self.bounds.upper) ** 2)
            return 1e10 + violation * 1e6
        
        try:
            # 调用前向模型
            y_pred = self.forward_fn(x)
            
            # 计算损失
            loss = self.loss_fn(y_pred, self.y_target, self.measurement_weights)
            
            # 缓存结果（用于后续复用，避免重复调用前向模型）
            x_key = hash(x.tobytes())
            self._cache[x_key] = (x.copy(), y_pred.copy() if hasattr(y_pred, 'copy') else np.array(y_pred), loss)
            
            # 限制缓存大小
            if len(self._cache) > self._cache_max_size:
                # 删除一半的缓存（简单策略）
                keys = list(self._cache.keys())
                for k in keys[:len(keys)//2]:
                    del self._cache[k]
            
            return loss
        except Exception as e:
            logger.warning(f"前向模型评估失败: {e}")
            return 1e10
    
    def get_cached_prediction(self, x: np.ndarray) -> Optional[np.ndarray]:
        """
        获取缓存的预测值（如果 x 在缓存中）
        
        用于避免在欧式距离计算等场景中重复调用前向模型
        """
        x_key = hash(x.tobytes())
        if x_key in self._cache:
            cached_x, cached_y_pred, _ = self._cache[x_key]
            # 验证 x 确实匹配（防止hash冲突）
            if np.allclose(x, cached_x, rtol=1e-10, atol=1e-10):
                return cached_y_pred
        return None
    
    def clear_cache(self):
        """清空预测值缓存"""
        self._cache.clear()
    
    def evaluate_solution(self, x: np.ndarray) -> Dict:
        """
        详细评估某个解
        
        返回包含预测值、残差等详细信息的字典
        """
        y_pred = self.forward_fn(x)
        residuals = y_pred - self.y_target
        
        return {
            'x': x,
            'y_pred': y_pred,
            'y_target': self.y_target,
            'residuals': residuals,
            'abs_residuals': np.abs(residuals),
            'relative_errors': np.abs(residuals) / (np.abs(self.y_target) + 1e-10),
            'loss': self.loss_fn(y_pred, self.y_target, self.measurement_weights),
            'max_abs_error': np.max(np.abs(residuals)),
            'mean_abs_error': np.mean(np.abs(residuals)),
            'rmse': np.sqrt(np.mean(residuals ** 2))
        }


# ============================================================================
# 第三部分: CMA-ES 优化器封装
# ============================================================================

@dataclass
class CMAESConfig:
    """CMA-ES 配置参数"""
    sigma0: Optional[float] = None      # 初始步长（None 则自动计算）
    maxiter: int = 1000                  # 最大迭代次数
    maxfevals: int = 50000               # 最大函数评估次数
    tolx: float = 1e-8                   # 参数收敛阈值
    tolfun: float = 1e-10                # 函数值收敛阈值
    popsize: Optional[int] = None        # 种群大小（None 则自动）
    seed: Optional[int] = None           # 随机种子
    verbose: int = 1                     # 日志级别 (0=静默, 1=进度, 2=详细)
    
    # 高级选项
    restart_from_best: bool = True       # 是否从最佳点重启
    bounds_handling: str = 'penalty'     # 边界处理方式: 'penalty' 或 'clip'
    
    # BIPOP-CMA-ES 参数（大小种群交替）
    use_bipop: bool = True               # 是否使用 BIPOP-CMA-ES（大小种群交替）
    popsize_large_ratio: float = 2.0     # 大种群相对于默认种群的比例
    popsize_small_ratio: float = 0.5     # 小种群相对于默认种群的比例
    switch_interval: int = 20            # 种群大小切换间隔（迭代次数）
    switch_condition: str = 'interval'   # 切换条件: 'interval'（固定间隔）或 'stagnation'（停滞时切换）
    stagnation_threshold: float = 1e-6   # 停滞判断阈值（相对改进）
    stagnation_window: int = 10          # 停滞判断窗口（迭代次数）
    
    # 并行计算参数
    use_parallel: bool = True             # 是否使用并行计算
    n_workers: Optional[int] = None      # 并行进程数（None 则自动检测，使用 CPU 核心数）
    parallel_backend: str = 'multiprocessing'  # 并行后端: 'multiprocessing' 或 'concurrent.futures'
    reuse_pool: bool = True              # 是否重用进程池（避免重复创建开销）


class CMAESOptimizer:
    """
    CMA-ES 优化器封装类（支持 BIPOP-CMA-ES：大小种群交替）
    
    提供：
    - 与 InversionProblem 的无缝集成
    - BIPOP-CMA-ES：大小种群交替策略（最强版本）
    - 多次运行支持（检测多个局部最优）
    - 详细的收敛日志
    - 结果后处理
    """
    
    def __init__(self, problem: InversionProblem, config: Optional[CMAESConfig] = None):
        """
        参数:
        - problem: 反演问题实例
        - config: CMA-ES 配置
        """
        self.problem = problem
        self.config = config if config is not None else CMAESConfig()
        
        # 延迟导入 cma 库
        self._cma = None
        
        # BIPOP-CMA-ES 状态
        self._bipop_state = {
            'current_popsize': None,
            'popsize_large': None,
            'popsize_small': None,
            'last_switch_iter': 0,
            'last_best_loss': None,
            'stagnation_count': 0,
            'use_large_pop': True  # 初始使用大种群
        }
        
        # 并行计算状态
        self._pool = None
        self._n_workers = None
        if self.config.use_parallel:
            self._n_workers = self.config.n_workers
            if self._n_workers is None:
                self._n_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给主进程
            self._n_workers = min(self._n_workers, mp.cpu_count())
    
    def _import_cma(self):
        """延迟导入 CMA-ES 库"""
        if self._cma is None:
            try:
                import cma
                self._cma = cma
            except ImportError:
                raise ImportError(
                    "请安装 cma 库: pip install cma\n"
                    "CMA-ES 是本反演系统的核心优化器"
                )
        return self._cma
    
    def _initialize_bipop(self, dim: int, default_popsize: Optional[int] = None):
        """初始化 BIPOP-CMA-ES 参数"""
        if default_popsize is None:
            # 计算默认种群大小（CMA-ES 标准公式）
            default_popsize = 4 + int(3 * np.log(dim))
        
        # 计算大小种群
        self._bipop_state['popsize_large'] = max(
            int(default_popsize * self.config.popsize_large_ratio),
            default_popsize + 1
        )
        self._bipop_state['popsize_small'] = max(
            int(default_popsize * self.config.popsize_small_ratio),
            4  # 最小种群大小
        )
        
        # 初始使用大种群
        self._bipop_state['current_popsize'] = self._bipop_state['popsize_large']
        self._bipop_state['use_large_pop'] = True
        self._bipop_state['last_switch_iter'] = 0
        self._bipop_state['last_best_loss'] = None
        self._bipop_state['stagnation_count'] = 0
        
        if self.config.verbose >= 1:
            logger.info(
                f"BIPOP-CMA-ES 初始化: "
                f"大种群={self._bipop_state['popsize_large']}, "
                f"小种群={self._bipop_state['popsize_small']}, "
                f"默认={default_popsize}"
            )
    
    def _should_switch_population(self, current_iter: int, current_best_loss: float) -> bool:
        """判断是否应该切换种群大小"""
        if not self.config.use_bipop:
            return False
        
        state = self._bipop_state
        
        if self.config.switch_condition == 'interval':
            # 固定间隔切换
            if current_iter - state['last_switch_iter'] >= self.config.switch_interval:
                return True
        elif self.config.switch_condition == 'stagnation':
            # 停滞时切换
            if state['last_best_loss'] is not None:
                # 计算相对改进
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
        
        state['last_switch_iter'] = 0  # 重置切换计数器
        state['stagnation_count'] = 0
        
        if self.config.verbose >= 1:
            pop_type = "大种群" if state['use_large_pop'] else "小种群"
            logger.info(
                f"切换到{pop_type} (大小={state['current_popsize']})"
            )
    
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
            # 每次创建新池（不重用）
            if self.config.parallel_backend == 'multiprocessing':
                return mp.Pool(processes=self._n_workers)
            elif self.config.parallel_backend == 'concurrent.futures':
                return ProcessPoolExecutor(max_workers=self._n_workers)
    
    def _close_pool(self, pool):
        """关闭进程池"""
        if pool is None:
            return
        
        if self.config.reuse_pool:
            # 重用池时不关闭
            return
        
        if isinstance(pool, ProcessPoolExecutor):
            pool.shutdown(wait=True)
        elif isinstance(pool, mp.pool.Pool):
            pool.close()
            pool.join()
    
    def _evaluate_fitness_parallel(self, solutions: List[np.ndarray], pool) -> List[float]:
        """并行评估适应度"""
        if pool is None:
            # 回退到串行
            return [self.problem.objective(x) for x in solutions]
        
        # 尝试并行计算
        # 注意：如果 problem.objective 包含不可序列化的闭包，会自动回退到串行
        try:
            # 只在第一次测试序列化（避免每次都测试，且缓存可能导致序列化失败）
            if not hasattr(self, '_parallel_ok'):
                try:
                    # 临时清空缓存以便序列化测试
                    old_cache = getattr(self.problem, '_cache', {})
                    if hasattr(self.problem, '_cache'):
                        self.problem._cache = {}
                    
                    import pickle
                    pickle.dumps(self.problem.objective)
                    self._parallel_ok = True
                    
                    # 恢复缓存
                    if hasattr(self.problem, '_cache'):
                        self.problem._cache = old_cache
                        
                    if self.config.verbose >= 1:
                        logger.info("并行计算序列化测试通过")
                except Exception as e:
                    self._parallel_ok = False
                    if self.config.verbose >= 1:
                        logger.warning(
                            f"并行计算序列化测试失败，将使用串行模式: {e}\n"
                            f"提示：如果 forward_fn 是闭包，建议在 RealDataInversionConfig 中使用类方法而非闭包"
                        )
            
            if not self._parallel_ok:
                return [self.problem.objective(x) for x in solutions]
            
            if isinstance(pool, ProcessPoolExecutor):
                # 使用 concurrent.futures（保持顺序）
                futures = [pool.submit(self.problem.objective, x) for x in solutions]
                return [f.result() for f in futures]  # 保持顺序
            elif isinstance(pool, mp.pool.Pool):
                # 使用 multiprocessing.Pool
                return pool.map(self.problem.objective, solutions)
            else:
                # 未知类型，回退到串行
                return [self.problem.objective(x) for x in solutions]
        except (pickle.PicklingError, AttributeError, TypeError, ValueError) as e:
            # 如果目标函数不可序列化，回退到串行
            self._parallel_ok = False
            if self.config.verbose >= 1:
                logger.warning(f"并行计算执行失败，回退到串行: {e}")
            return [self.problem.objective(x) for x in solutions]
        except Exception as e:
            # 其他错误也回退到串行
            if self.config.verbose >= 1:
                logger.warning(f"并行计算失败，回退到串行: {e}")
            return [self.problem.objective(x) for x in solutions]
    
    def _evaluate_fitness(self, solutions: List[np.ndarray]) -> List[float]:
        """评估适应度（自动选择并行或串行）"""
        if not self.config.use_parallel or len(solutions) == 1:
            # 串行评估
            return [self.problem.objective(x) for x in solutions]
        
        # 并行评估
        pool = self._get_pool()
        try:
            fitness = self._evaluate_fitness_parallel(solutions, pool)
        except Exception as e:
            if self.config.verbose >= 1:
                logger.warning(f"并行计算失败，回退到串行: {e}")
            fitness = [self.problem.objective(x) for x in solutions]
        finally:
            self._close_pool(pool if not self.config.reuse_pool else None)
        
        return fitness
    
    def __del__(self):
        """析构函数：清理进程池"""
        if self._pool is not None:
            if isinstance(self._pool, ProcessPoolExecutor):
                self._pool.shutdown(wait=False)
            elif isinstance(self._pool, mp.pool.Pool):
                self._pool.terminate()
                self._pool.join()
    
    def run(self, x0: Optional[np.ndarray] = None) -> InversionResult:
        """
        运行单次 CMA-ES 优化（支持 BIPOP-CMA-ES：大小种群交替）
        
        参数:
        - x0: 初始点（None 则使用边界中心）
        
        返回:
        - InversionResult 结果对象
        """
        cma = self._import_cma()
        
        # 初始化
        if x0 is None:
            x0 = self.problem.bounds.get_initial_guess()
        
        sigma0 = self.config.sigma0
        if sigma0 is None:
            sigma0 = self.problem.bounds.get_initial_sigma()
        
        # 重置评估计数
        self.problem.reset_eval_count()
        
        # 初始化 BIPOP 参数
        default_popsize = self.config.popsize
        if self.config.use_bipop:
            self._initialize_bipop(self.problem.dim, default_popsize)
            current_popsize = self._bipop_state['current_popsize']
        else:
            current_popsize = default_popsize
        
        # 配置 CMA-ES 选项
        opts = {
            'maxiter': self.config.maxiter,
            'maxfevals': self.config.maxfevals,
            'tolx': self.config.tolx,
            'tolfun': self.config.tolfun,
            'bounds': [self.problem.bounds.lower.tolist(), 
                       self.problem.bounds.upper.tolist()],
            'verbose': self.config.verbose - 1,  # cma 的 verbose 从 -1 开始
            'seed': self.config.seed,
            'popsize': current_popsize
        }
        
        # 记录历史
        x_history = []
        loss_history = []
        popsize_history = []  # 记录种群大小变化
        
        start_time = time.time()
        
        # 创建优化器并运行
        if self.config.verbose >= 1:
            mode_str = "BIPOP-CMA-ES (大小种群交替)" if self.config.use_bipop else "CMA-ES"
            parallel_info = f", 并行进程={self._n_workers}" if self.config.use_parallel else ""
            logger.info(
                f"开始 {mode_str} 优化 "
                f"(维度={self.problem.dim}, σ₀={sigma0:.4f}, "
                f"初始种群={current_popsize}{parallel_info})"
            )
        
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        best_loss_so_far = float('inf')
        best_x_so_far = x0.copy()
        
        iteration_count = 0  # 手动跟踪迭代次数（因为切换时会重置 es.countiter）
        
        # 计时统计
        timing_stats = {
            'iteration_times': [],
            'step_times': {
                'bipop_check': [],
                'ask': [],
                'evaluate': [],
                'tell': [],
                'history_update': []
            },
            'individual_eval_times': []
        }
        
        while not es.stop():
            iter_start_time = time.time()
            step_times = {}
            
            # ========== 步骤1: BIPOP 检查/切换 ==========
            t0 = time.time()
            if self.config.use_bipop and iteration_count > 0:
                if self._should_switch_population(iteration_count, best_loss_so_far):
                    # 保存当前状态
                    current_x = es.result.xbest
                    current_sigma = es.sigma
                    
                    # 切换种群大小
                    self._switch_population_size()
                    new_popsize = self._bipop_state['current_popsize']
                    
                    # 创建新的优化器，从当前最佳点重启
                    new_opts = opts.copy()
                    new_opts['popsize'] = new_popsize
                    
                    # 从当前最佳点重启，保持 sigma
                    es = cma.CMAEvolutionStrategy(current_x, current_sigma, new_opts)
                    
                    if self.config.verbose >= 1:
                        pop_type = "大种群" if self._bipop_state['use_large_pop'] else "小种群"
                        logger.info(
                            f"迭代 {iteration_count}: 切换到{pop_type} "
                            f"(大小={new_popsize}), 当前最优损失={best_loss_so_far:.6e}"
                        )
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
            
            # ========== 步骤5: 记录历史 ==========
            t0 = time.time()
            x_best_gen = solutions[np.argmin(fitness)]
            loss_best_gen = min(fitness)
            
            # 更新全局最优
            if loss_best_gen < best_loss_so_far:
                best_loss_so_far = loss_best_gen
                best_x_so_far = x_best_gen.copy()
            
            x_history.append(x_best_gen.copy())
            loss_history.append(loss_best_gen)
            
            # 记录当前种群大小
            current_popsize_iter = len(solutions)
            popsize_history.append(current_popsize_iter)
            
            # 更新迭代计数
            iteration_count += 1
            
            # 更新 BIPOP 状态
            if self.config.use_bipop:
                self._bipop_state['last_switch_iter'] += 1
                self._bipop_state['last_best_loss'] = best_loss_so_far
            step_times['history_update'] = time.time() - t0
            
            # ========== 计算迭代总时间并记录统计 ==========
            iter_total_time = time.time() - iter_start_time
            timing_stats['iteration_times'].append(iter_total_time)
            for key in step_times:
                timing_stats['step_times'][key].append(step_times[key])
            
            # 日志输出（详细计时信息）
            if self.config.verbose >= 2:
                pop_size = len(solutions)
                eval_time = step_times['evaluate']
                avg_eval_time = eval_time / pop_size if pop_size > 0 else 0
                popsize_info = f", 种群={current_popsize_iter}" if self.config.use_bipop else ""
                logger.info(
                    f"迭代 {iteration_count}: "
                    f"最优损失={loss_best_gen:.6e}, "
                    f"sigma={es.sigma:.4e}{popsize_info}, "
                    f"用时={iter_total_time*1000:.1f}ms (评估={eval_time*1000:.1f}ms, {avg_eval_time*1000:.2f}ms/个体)"
                )
            elif self.config.verbose >= 1 and iteration_count % 10 == 0:
                # 每10次迭代输出一次简要信息
                pop_size = len(solutions)
                eval_time = step_times['evaluate']
                logger.info(
                    f"迭代 {iteration_count}: 损失={loss_best_gen:.6e}, "
                    f"用时={iter_total_time*1000:.1f}ms (评估={eval_time*1000:.1f}ms)"
                )
        
        elapsed_time = time.time() - start_time
        
        # 获取最终结果（使用全局最优）
        x_best = best_x_so_far
        loss_best = best_loss_so_far
        
        # 计算种群切换次数
        bipop_switches = 0
        if self.config.use_bipop and len(popsize_history) > 1:
            bipop_switches = len([i for i in range(1, len(popsize_history)) 
                                  if popsize_history[i] != popsize_history[i-1]])
        
        # 收敛信息
        convergence_info = {
            'stop_conditions': es.stop(),
            'iterations': iteration_count,
            'final_sigma': es.sigma,
            'condition_number': es.D.max() / es.D.min() if hasattr(es, 'D') else None,
            'popsize_history': popsize_history if self.config.use_bipop else None,
            'bipop_switches': bipop_switches
        }
        
        if self.config.verbose >= 1:
            switch_info = f", 种群切换={bipop_switches}次" if self.config.use_bipop else ""
            parallel_info = f", 并行加速" if self.config.use_parallel else ""
            logger.info(
                f"优化完成: 损失={loss_best:.6e}, 迭代={iteration_count}{switch_info}{parallel_info}"
            )
            
            # 打印计时统计汇总
            if timing_stats['iteration_times']:
                n_iters = len(timing_stats['iteration_times'])
                total_iter_time = sum(timing_stats['iteration_times'])
                avg_iter_time = total_iter_time / n_iters
                
                logger.info("="*60)
                logger.info("计时统计汇总")
                logger.info("="*60)
                logger.info(f"总迭代次数: {n_iters}")
                logger.info(f"总运行时间: {elapsed_time:.2f}s")
                logger.info(f"迭代总用时: {total_iter_time:.2f}s")
                logger.info(f"平均每迭代: {avg_iter_time*1000:.1f}ms")
                
                step_names = {
                    'bipop_check': 'BIPOP检查',
                    'ask': '生成候选解',
                    'evaluate': '适应度评估',
                    'tell': '优化器更新',
                    'history_update': '历史记录'
                }
                
                logger.info("各步骤平均用时:")
                for key, name in step_names.items():
                    times = timing_stats['step_times'][key]
                    if times:
                        avg_time = sum(times) / len(times) * 1000
                        total_time = sum(times) * 1000
                        pct = (sum(times) / total_iter_time * 100) if total_iter_time > 0 else 0
                        logger.info(f"  {name:12s}: 平均={avg_time:>8.2f}ms, 总计={total_time:>8.1f}ms ({pct:>5.1f}%)")
                
                # 显示个体评估时间统计
                if timing_stats['individual_eval_times']:
                    all_times = [t for times in timing_stats['individual_eval_times'] for t in times]
                    if all_times:
                        logger.info(f"个体评估时间统计 (共{len(all_times)}次评估):")
                        logger.info(f"  最小: {min(all_times)*1000:.2f}ms")
                        logger.info(f"  最大: {max(all_times)*1000:.2f}ms")
                        logger.info(f"  平均: {sum(all_times)/len(all_times)*1000:.2f}ms")
                        logger.info(f"  总计: {sum(all_times):.2f}s")
                
                logger.info("="*60)
        
        # 清理进程池
        if self.config.reuse_pool and self._pool is not None:
            if isinstance(self._pool, ProcessPoolExecutor):
                self._pool.shutdown(wait=True)
            elif isinstance(self._pool, mp.pool.Pool):
                self._pool.close()
                self._pool.join()
            self._pool = None
        
        return InversionResult(
            x_best=x_best,
            loss_best=loss_best,
            x_history=x_history,
            loss_history=loss_history,
            n_evaluations=self.problem._eval_count,
            convergence_info=convergence_info,
            elapsed_time=elapsed_time
        )
    
    def run_multi(self, n_runs: int = 5, 
                  random_init: bool = True) -> List[InversionResult]:
        """
        多次运行 CMA-ES（用于检测多个局部最优）
        
        参数:
        - n_runs: 运行次数
        - random_init: 是否使用随机初始点
        
        返回:
        - 所有运行结果的列表
        """
        results = []
        
        for i in range(n_runs):
            if self.config.verbose >= 1:
                logger.info(f"\n{'='*40}")
                logger.info(f"第 {i+1}/{n_runs} 次运行")
                logger.info(f"{'='*40}")
            
            # 生成初始点
            if random_init:
                bounds = self.problem.bounds
                x0 = bounds.lower + np.random.rand(self.problem.dim) * (
                    bounds.upper - bounds.lower
                )
            else:
                x0 = None
            
            # 设置不同的随机种子
            original_seed = self.config.seed
            if self.config.seed is not None:
                self.config.seed = self.config.seed + i * 1000
            
            result = self.run(x0)
            results.append(result)
            
            # 恢复原始种子
            self.config.seed = original_seed
        
        # 按损失值排序
        results.sort(key=lambda r: r.loss_best)
        
        if self.config.verbose >= 1:
            logger.info(f"\n多次运行完成，最佳损失: {results[0].loss_best:.6e}")
        
        return results
    
    def analyze_results(self, results: List[InversionResult]) -> Dict:
        """
        分析多次运行结果
        
        返回:
        - 统计信息和局部最优检测结果
        """
        losses = [r.loss_best for r in results]
        x_bests = np.array([r.x_best for r in results])
        
        # 检测不同的局部最优
        unique_solutions = []
        tolerance = 0.01 * np.mean(self.problem.bounds.upper - self.problem.bounds.lower)
        
        for i, x in enumerate(x_bests):
            is_unique = True
            for ux in unique_solutions:
                if np.linalg.norm(x - ux) < tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique_solutions.append(x)
        
        return {
            'n_runs': len(results),
            'best_loss': min(losses),
            'worst_loss': max(losses),
            'mean_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'n_unique_solutions': len(unique_solutions),
            'unique_solutions': unique_solutions,
            'loss_spread': max(losses) - min(losses),
            'is_consistent': np.std(losses) / (np.mean(losses) + 1e-10) < 0.1
        }


# ============================================================================
# 第四部分: 示例风网模型
# ============================================================================

class VentilationNetwork:
    """
    简化的矿井通风网络模型
    
    采用简化的非线性风阻模型：
    - 风阻公式: h = R * Q²  (h: 风压降, R: 风阻, Q: 风量)
    - 阻力系数与风阻关系: R = k * L * P / (A³)
      其中 k: 摩擦阻力系数, L: 长度, P: 周长, A: 断面积
    
    本示例使用简化的非线性方程组求解
    """
    
    def __init__(
        self,
        n_branches: int,
        incidence_matrix: np.ndarray,
        branch_lengths: np.ndarray,
        branch_areas: np.ndarray,
        branch_perimeters: np.ndarray,
        fan_pressure: float = 1000.0,
        fan_branch: int = 0
    ):
        """
        参数:
        - n_branches: 巷道数量
        - incidence_matrix: 节点-巷道关联矩阵 (n_nodes × n_branches)
        - branch_lengths: 各巷道长度 (m)
        - branch_areas: 各巷道断面积 (m²)
        - branch_perimeters: 各巷道周长 (m)
        - fan_pressure: 风机压力 (Pa)
        - fan_branch: 风机所在巷道编号
        """
        self.n_branches = n_branches
        self.A = incidence_matrix
        self.n_nodes = incidence_matrix.shape[0]
        self.lengths = np.asarray(branch_lengths)
        self.areas = np.asarray(branch_areas)
        self.perimeters = np.asarray(branch_perimeters)
        self.fan_pressure = fan_pressure
        self.fan_branch = fan_branch
        
        # 预计算几何因子
        self._geo_factors = self.lengths * self.perimeters / (self.areas ** 3)
    
    def compute_resistance(self, k: np.ndarray) -> np.ndarray:
        """
        计算各巷道风阻
        
        参数:
        - k: 阻力系数向量
        
        返回:
        - R: 风阻向量
        """
        return k * self._geo_factors
    
    def solve_flow(self, k: np.ndarray, max_iter: int = 100, 
                   tol: float = 1e-6) -> np.ndarray:
        """
        求解风网流量分布（简化的迭代法）
        
        使用 Hardy-Cross 思想的简化实现
        
        参数:
        - k: 阻力系数向量
        - max_iter: 最大迭代次数
        - tol: 收敛容差
        
        返回:
        - Q: 各巷道风量 (m³/s)
        """
        R = self.compute_resistance(k)
        
        # 初始化流量（假设均匀分布）
        Q = np.ones(self.n_branches) * 10.0  # m³/s
        
        # 简化的迭代求解
        for iteration in range(max_iter):
            Q_old = Q.copy()
            
            # 计算各巷道风压降
            h = R * Q * np.abs(Q)  # h = R * Q²，保留符号
            
            # 添加风机压力
            h_with_fan = h.copy()
            h_with_fan[self.fan_branch] -= self.fan_pressure
            
            # 基于压力平衡调整流量（简化）
            # 这里使用一个简化的更新规则
            pressure_imbalance = np.sum(h_with_fan * Q) / (np.sum(R * Q ** 2) + 1e-10)
            
            # 更新流量
            correction = 0.5 * pressure_imbalance
            Q = Q * (1 - 0.1 * np.sign(h_with_fan) * np.abs(correction))
            Q = np.maximum(Q, 0.1)  # 保证正流量
            
            # 检查收敛
            if np.max(np.abs(Q - Q_old)) < tol:
                break
        
        return Q
    
    def forward(self, k: np.ndarray) -> np.ndarray:
        """
        前向模型：从阻力系数计算可观测量
        
        参数:
        - k: 阻力系数向量
        
        返回:
        - y: 观测量（这里返回各测点的风量）
        """
        Q = self.solve_flow(k)
        return Q
    
    def create_forward_function(self, 
                                measurement_indices: Optional[List[int]] = None
                                ) -> Callable[[np.ndarray], np.ndarray]:
        """
        创建前向函数（可插拔接口）
        
        参数:
        - measurement_indices: 测量点索引（None 表示所有巷道）
        
        返回:
        - forward_fn: 可调用的前向函数
        """
        indices = measurement_indices
        
        def forward_fn(k: np.ndarray) -> np.ndarray:
            Q = self.solve_flow(k)
            if indices is not None:
                return Q[indices]
            return Q
        
        return forward_fn


def create_demo_network() -> VentilationNetwork:
    """
    创建演示用的 8 巷道通风网络
    
    网络拓扑:
        节点0 (入风) --- 巷道0 ---> 节点1
                                   |
                           巷道1   |   巷道2
                                   v
                         节点2 <---+---> 节点3
                           |               |
                       巷道3           巷道4
                           v               v
                         节点4 ----巷道5---> 节点5
                                           |
                                       巷道6
                                           v
                         节点6 <---巷道7--- 节点7 (回风)
    """
    n_branches = 8
    n_nodes = 8
    
    # 关联矩阵（简化表示）
    # 实际矿井通风计算中会使用标准的图论关联矩阵
    A = np.zeros((n_nodes, n_branches))
    
    # 巷道几何参数
    lengths = np.array([100, 80, 90, 120, 110, 70, 100, 85])  # m
    areas = np.array([8, 7, 7.5, 9, 8.5, 8, 7, 7.5])  # m²
    perimeters = np.array([12, 11, 11.5, 13, 12.5, 12, 11, 11.5])  # m
    
    return VentilationNetwork(
        n_branches=n_branches,
        incidence_matrix=A,
        branch_lengths=lengths,
        branch_areas=areas,
        branch_perimeters=perimeters,
        fan_pressure=1200.0,  # Pa
        fan_branch=7  # 风机在回风巷道
    )


# ============================================================================
# 第五部分: 简化线性模型（用于演示和测试）
# ============================================================================

class LinearVentilationModel:
    """
    简化的线性通风模型（用于算法验证）
    
    假设风量与阻力系数的关系可以用线性模型近似：
    y = G @ (1/sqrt(k)) + noise
    
    这模拟了风量 Q ∝ 1/sqrt(R) 的物理关系
    其中 G 是观测矩阵，表示各测点对各巷道参数的敏感度
    """
    
    def __init__(self, n_params: int, n_measurements: int, 
                 sensitivity_matrix: Optional[np.ndarray] = None,
                 noise_std: float = 0.0):
        """
        参数:
        - n_params: 阻力系数数量
        - n_measurements: 测量点数量
        - sensitivity_matrix: 敏感度矩阵 G (n_measurements × n_params)
        - noise_std: 测量噪声标准差
        """
        self.n_params = n_params
        self.n_measurements = n_measurements
        self.noise_std = noise_std
        
        if sensitivity_matrix is None:
            # 生成良定的敏感度矩阵（对角占优，确保可辨识性）
            np.random.seed(42)  # 可重复性
            # 使用对角占优矩阵确保问题是良定的
            self.G = np.eye(n_measurements, n_params) * 2.0
            # 添加少量交叉敏感性
            self.G += np.random.randn(n_measurements, n_params) * 0.3
            self.G = np.abs(self.G)  # 确保正值（物理意义）
        else:
            self.G = sensitivity_matrix
    
    def forward(self, k: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        前向模型
        
        参数:
        - k: 阻力系数向量
        - add_noise: 是否添加测量噪声
        
        返回:
        - y: 观测量（风量）
        """
        # 使用 1/sqrt(k) 来模拟物理关系 Q ∝ 1/sqrt(R) ∝ 1/sqrt(k)
        k_inv_sqrt = 1.0 / np.sqrt(np.maximum(k, 1e-10))
        y = self.G @ k_inv_sqrt
        
        if add_noise and self.noise_std > 0:
            y += np.random.randn(len(y)) * self.noise_std
        
        return y
    
    def create_forward_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """创建前向函数接口"""
        return lambda k: self.forward(k, add_noise=False)


# ============================================================================
# 第六部分: 主运行函数
# ============================================================================

def run_inversion_demo():
    """
    运行完整的反演演示
    
    演示内容：
    1. 构建示例风网
    2. 设定真实阻力系数
    3. 生成观测数据
    4. 运行 CMA-ES 反演
    5. 分析和输出结果
    """
    print("\n" + "="*70)
    print("矿井通风网络阻力系数反演演示")
    print("使用 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)")
    print("="*70 + "\n")
    
    # -------------------- 参数设置 --------------------
    n_branches = 8  # 巷道数量
    n_measurements = 8  # 测量点数量（这里假设可以测量所有巷道）
    
    # -------------------- 真实阻力系数 --------------------
    np.random.seed(42)
    k_true = np.array([
        0.0025,  # 巷道0
        0.0030,  # 巷道1
        0.0028,  # 巷道2
        0.0022,  # 巷道3
        0.0035,  # 巷道4
        0.0020,  # 巷道5
        0.0032,  # 巷道6
        0.0027   # 巷道7
    ])
    
    print(f"真实阻力系数 k_true:")
    for i, k in enumerate(k_true):
        print(f"  k_{i} = {k:.4f}")
    print()
    
    # -------------------- 构建模型 --------------------
    # 使用简化线性模型进行演示（更稳定的验证）
    model = LinearVentilationModel(
        n_params=n_branches,
        n_measurements=n_measurements,
        noise_std=0.1  # 测量噪声
    )
    
    # 生成观测数据（使用真实参数 + 噪声）
    y_target = model.forward(k_true, add_noise=True)
    
    print(f"观测数据 y_target:")
    for i, y in enumerate(y_target):
        print(f"  y_{i} = {y:.4f}")
    print()
    
    # -------------------- 定义反演问题 --------------------
    # 参数边界（阻力系数的物理范围）
    bounds = InversionBounds(
        lower=np.ones(n_branches) * 0.001,  # 下界
        upper=np.ones(n_branches) * 0.010   # 上界
    )
    
    # 创建前向函数
    forward_fn = model.create_forward_function()
    
    # 使用 Huber Loss（推荐）
    loss_fn = HuberLoss(delta=0.5)
    
    # 创建反演问题实例
    problem = InversionProblem(
        forward_fn=forward_fn,
        y_target=y_target,
        bounds=bounds,
        loss_fn=loss_fn,
        param_names=[f"k_{i}" for i in range(n_branches)]
    )
    
    print(f"反演问题配置:")
    print(f"  参数维度: {problem.dim}")
    print(f"  测量点数: {problem.n_measurements}")
    print(f"  损失函数: {loss_fn.name()}")
    print(f"  参数范围: [{bounds.lower[0]:.4f}, {bounds.upper[0]:.4f}]")
    print()
    
    # -------------------- CMA-ES 优化 --------------------
    config = CMAESConfig(
        maxiter=300,
        maxfevals=10000,
        tolx=1e-8,
        tolfun=1e-10,
        verbose=1,
        seed=123
    )
    
    optimizer = CMAESOptimizer(problem, config)
    
    print("开始单次 CMA-ES 优化...")
    result = optimizer.run()
    
    # -------------------- 结果分析 --------------------
    print(result.get_summary())
    
    # 详细评估
    evaluation = problem.evaluate_solution(result.x_best)
    
    print("\n反演结果对比:")
    print("-" * 50)
    print(f"{'参数':<10} {'真实值':<12} {'反演值':<12} {'相对误差':<12}")
    print("-" * 50)
    
    for i in range(n_branches):
        k_t = k_true[i]
        k_e = result.x_best[i]
        rel_err = abs(k_e - k_t) / k_t * 100
        print(f"k_{i:<8} {k_t:<12.5f} {k_e:<12.5f} {rel_err:<10.2f}%")
    
    print("-" * 50)
    
    # 计算整体误差
    total_rel_error = np.mean(np.abs(result.x_best - k_true) / k_true) * 100
    print(f"\n平均相对误差: {total_rel_error:.2f}%")
    print(f"RMSE: {evaluation['rmse']:.6f}")
    print(f"最大绝对误差: {evaluation['max_abs_error']:.6f}")
    
    # -------------------- 多次运行（检测局部最优） --------------------
    print("\n" + "="*50)
    print("执行多次运行以检测局部最优...")
    print("="*50)
    
    config.verbose = 0  # 减少输出
    config.seed = None  # 使用随机种子
    
    results = optimizer.run_multi(n_runs=5, random_init=True)
    analysis = optimizer.analyze_results(results)
    
    print(f"\n多次运行分析:")
    print(f"  运行次数: {analysis['n_runs']}")
    print(f"  最佳损失: {analysis['best_loss']:.6e}")
    print(f"  最差损失: {analysis['worst_loss']:.6e}")
    print(f"  损失均值: {analysis['mean_loss']:.6e}")
    print(f"  损失标准差: {analysis['std_loss']:.6e}")
    print(f"  发现的唯一解数量: {analysis['n_unique_solutions']}")
    print(f"  结果一致性: {'是' if analysis['is_consistent'] else '否'}")
    
    print("\n演示完成！")
    
    return result, results, analysis


def compare_loss_functions():
    """
    比较不同损失函数的性能
    
    演示场景：存在测量异常值时，不同损失函数的鲁棒性对比
    """
    print("\n" + "="*70)
    print("损失函数对比实验")
    print("="*70 + "\n")
    
    # 设置
    n_params = 6
    n_measurements = 8  # 测量点多于参数，构成超定问题
    
    np.random.seed(42)
    k_true = np.random.uniform(0.002, 0.005, n_params)
    
    # 创建敏感度矩阵（超定系统）
    np.random.seed(123)
    G = np.zeros((n_measurements, n_params))
    for i in range(n_params):
        G[i, i] = 2.0  # 对角元素
    G[n_params:, :] = np.random.rand(n_measurements - n_params, n_params) * 0.5 + 0.5
    G += np.random.randn(n_measurements, n_params) * 0.2
    G = np.abs(G)
    
    # 创建无噪声模型用于生成真实数据
    model = LinearVentilationModel(
        n_params=n_params,
        n_measurements=n_measurements,
        sensitivity_matrix=G,
        noise_std=0.0
    )
    
    # 生成干净的目标值
    y_clean = model.forward(k_true, add_noise=False)
    
    # 添加少量高斯噪声 + 两个严重离群点
    np.random.seed(456)
    y_target = y_clean.copy()
    y_target += np.random.randn(n_measurements) * 0.5  # 小噪声
    y_target[2] *= 2.0   # 离群点1：放大100%
    y_target[5] *= 0.5   # 离群点2：缩小50%
    
    forward_fn = model.create_forward_function()
    bounds = InversionBounds(
        lower=np.ones(n_params) * 0.001,
        upper=np.ones(n_params) * 0.010
    )
    
    # 测试三种损失函数
    loss_functions = [
        ("MSE", MSELoss()),
        ("MAE", MAELoss()),
        ("Huber(δ=1.0)", HuberLoss(delta=1.0)),
        ("Huber(δ=5.0)", HuberLoss(delta=5.0)),
    ]
    
    config = CMAESConfig(maxiter=300, verbose=0, seed=42)
    
    print(f"测试场景描述:")
    print(f"  参数维度: {n_params}")
    print(f"  测量点数: {n_measurements}")
    print(f"  离群点1: y_target[2] 放大 100%")
    print(f"  离群点2: y_target[5] 缩小 50%")
    print()
    
    print("-" * 60)
    print(f"{'损失函数':<20} {'最终损失':<15} {'参数相对误差':<15}")
    print("-" * 60)
    
    results_comparison = []
    
    for name, loss_fn in loss_functions:
        problem = InversionProblem(
            forward_fn=forward_fn,
            y_target=y_target,
            bounds=bounds,
            loss_fn=loss_fn
        )
        
        optimizer = CMAESOptimizer(problem, config)
        result = optimizer.run()
        
        # 计算与真实值的误差
        rel_error = np.mean(np.abs(result.x_best - k_true) / k_true) * 100
        
        results_comparison.append({
            'name': name,
            'loss': result.loss_best,
            'rel_error': rel_error,
            'k_est': result.x_best
        })
        
        print(f"{name:<20} {result.loss_best:<15.6e} {rel_error:<13.2f}%")
    
    print("-" * 60)
    
    # 结论
    best_method = min(results_comparison, key=lambda x: x['rel_error'])
    print(f"\n结论: {best_method['name']} 在有离群点时表现最好（误差最小）")
    print("\n建议:")
    print("  - MSE: 对离群点非常敏感，不适合有异常值的数据")
    print("  - MAE: 对离群点鲁棒，但梯度不连续可能影响收敛")
    print("  - Huber: 兼顾鲁棒性和平滑性，推荐作为默认选择")
    print("  - δ值越小越接近MAE（更鲁棒），越大越接近MSE（更平滑）")


# ============================================================================
# 程序入口
# ============================================================================

if __name__ == "__main__":
    # 运行主演示
    result, results, analysis = run_inversion_demo()
    
    # 运行损失函数对比
    compare_loss_functions()

