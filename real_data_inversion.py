"""
矿井通风网络阻力系数反演 - 实际数据适配模块

读取 input.json 格式的实际风网数据，配置反演问题

数据结构:
- roads: 巷道数据 (r0, minR, maxR, targetQ 等)
- fanHs: 风机数据 (h0, minH, maxH)
- structureHs: 结构物阻力 (风门、风窗等)
"""

import json
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
import logging
import sys
import os
import copy
import pickle

# 添加 cloud 目录到 Python 路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
_cloud_path = os.path.join(_current_dir, 'cloud')
if _cloud_path not in sys.path:
    sys.path.insert(0, _cloud_path)

# 导入反演核心模块
from ventilation_inversion import (
    InversionProblem, InversionBounds, 
    CMAESOptimizer, CMAESConfig,
    HuberLoss, MSELoss, MAELoss,
    InversionResult, LossFunction
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

@dataclass
class RoadData:
    """巷道数据"""
    id: str                     # 巷道ID
    source: str                 # 起点节点
    target: str                 # 终点节点
    r0: float                   # 初始风阻（已优化）
    min_r: float                # 最小风阻
    max_r: float                # 最大风阻
    init_q: float               # 初始风量
    target_q: Optional[float]   # 目标风量（测点）
    ex: float = 2.0             # 风阻指数 (h = R * Q^ex)
    weight: float = 1.0         # 权重
    index: int = 0              # 原始序号


@dataclass
class FanData:
    """风机数据"""
    id: str                     # 风机ID
    edge_id: str                # 对应巷道ID
    h0: float                   # 初始风机压力
    min_h: float                # 最小压力
    max_h: float                # 最大压力
    use: str = "LOCAL"          # 用途: "LOCAL" 或 "MAIN"


@dataclass 
class StructureData:
    """结构物阻力数据（风门、风窗等）"""
    id: str                     # 结构物ID
    edge_id: str                # 对应巷道ID
    h: float                    # 结构物阻力值


@dataclass
class VentilationNetworkData:
    """完整的通风网络数据"""
    roads: List[RoadData]
    fans: List[FanData]
    structures: List[StructureData]
    
    # 快速查找表
    road_by_id: Dict[str, RoadData] = field(default_factory=dict)
    measurement_roads: List[RoadData] = field(default_factory=list)
    
    def __post_init__(self):
        # 构建查找表
        self.road_by_id = {r.id: r for r in self.roads}
        # 找出有测量值的巷道
        self.measurement_roads = [r for r in self.roads if r.target_q is not None]
    
    @property
    def n_roads(self) -> int:
        return len(self.roads)
    
    @property
    def n_measurements(self) -> int:
        return len(self.measurement_roads)
    
    @property
    def n_fans(self) -> int:
        return len(self.fans)


# ============================================================================
# 数据加载器
# ============================================================================

class DataLoader:
    """
    从 input.json 加载通风网络数据
    """
    
    @staticmethod
    def load(filepath: str, fix_bounds: bool = True) -> VentilationNetworkData:
        """
        加载JSON数据文件
        
        参数:
        - filepath: JSON文件路径
        - fix_bounds: 是否自动修复越界的初始值
        
        返回:
        - VentilationNetworkData 对象
        """
        logger.info(f"加载数据文件: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 统计边界问题
        r0_below_min = 0
        r0_above_max = 0
        
        # 解析巷道数据
        roads = []
        for i, r in enumerate(data.get('roads', [])):
            r0 = r['r0']
            min_r = r['minR']
            max_r = r['maxR']
            
            # 检查并修复边界问题
            if r0 < min_r:
                r0_below_min += 1
                if fix_bounds:
                    # 将 r0 调整到边界内，或扩展边界
                    min_r = r0 * 0.5  # 扩展下界
            if r0 > max_r:
                r0_above_max += 1
                if fix_bounds:
                    # 将 r0 调整到边界内，或扩展边界
                    max_r = r0 * 2.0  # 扩展上界
            
            road = RoadData(
                id=r['id'],
                source=r.get('s', ''),
                target=r.get('t', ''),
                r0=r0,
                min_r=min_r,
                max_r=max_r,
                init_q=r.get('initQ', 0.0),
                target_q=r.get('targetQ'),
                ex=r.get('ex', 2.0),
                weight=r.get('weight', 1.0),
                index=int(r.get('序号', i + 1))
            )
            roads.append(road)
        
        # 解析风机数据
        fans = []
        h0_out_of_bounds = 0
        for f in data.get('fanHs', []):
            h0 = f['h0']
            min_h = min(f['minH'], f['maxH'])  # 确保 min < max
            max_h = max(f['minH'], f['maxH'])
            
            # 检查并修复边界问题
            if h0 < min_h or h0 > max_h:
                h0_out_of_bounds += 1
                if fix_bounds:
                    # 扩展边界以包含 h0
                    if h0 < min_h:
                        min_h = h0 * 1.1 if h0 < 0 else h0 * 0.9
                    if h0 > max_h:
                        max_h = h0 * 0.9 if h0 < 0 else h0 * 1.1
            
            fan = FanData(
                id=f['id'],
                edge_id=f['eid'],
                h0=h0,
                min_h=min_h,
                max_h=max_h,
                use=f.get('use', 'LOCAL')
            )
            fans.append(fan)
        
        # 解析结构物数据
        structures = []
        for s in data.get('structureHs', []):
            struct = StructureData(
                id=s['id'],
                edge_id=s['eid'],
                h=s['h']
            )
            structures.append(struct)
        
        network = VentilationNetworkData(
            roads=roads,
            fans=fans,
            structures=structures
        )
        
        logger.info(f"数据加载完成:")
        logger.info(f"  巷道数: {network.n_roads}")
        logger.info(f"  测点数: {network.n_measurements}")
        logger.info(f"  风机数: {network.n_fans}")
        logger.info(f"  结构物数: {len(structures)}")
        
        # 报告边界问题
        if r0_below_min > 0 or r0_above_max > 0:
            logger.warning(f"边界问题:")
            logger.warning(f"  R0 低于下界: {r0_below_min}")
            logger.warning(f"  R0 高于上界: {r0_above_max}")
            if fix_bounds:
                logger.info(f"  已自动扩展边界以包含初始值")
        
        if h0_out_of_bounds > 0:
            logger.warning(f"  H0 超出边界: {h0_out_of_bounds}")
            if fix_bounds:
                logger.info(f"  已自动扩展边界以包含初始值")
        
        return network


# ============================================================================
# 反演问题配置器
# ============================================================================

class SerializableForwardFn:
    """
    可序列化的前向函数包装类（支持并行计算）
    
    必须在模块级别定义，才能被 pickle 序列化。
    存储必要的可序列化引用，而不是整个 config 对象。
    """
    def __init__(self, config):
        # 只存储必要的可序列化引用
        self.forward_model = config.forward_model
        self.measurement_indices = config.measurement_indices
        
        # 存储解码参数所需的信息
        self.n_r_params = config.n_r_params
        self.n_h_params = config.n_h_params
        self.h_start_idx = config.h_start_idx
        self.use_log_scale = config.use_log_scale
        self.network_roads = config.network.roads
        self.network_fans = config.network.fans
        self.fan_to_optimize = config.fan_to_optimize if hasattr(config, 'fan_to_optimize') else []
    
    def _decode_parameters(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """解码参数（从 config 中提取的逻辑）"""
        # 解码风阻
        if self.use_log_scale:
            R = np.exp(x[:self.n_r_params])
        else:
            R = x[:self.n_r_params].copy()
        
        # 解码/填充风机压力
        H = np.array([f.h0 for f in self.network_fans])
        if self.n_h_params > 0:
            for j, fan in enumerate(self.fan_to_optimize):
                idx = self.h_start_idx + j
                fan_idx = self.network_fans.index(fan)
                H[fan_idx] = x[idx]
        
        return R, H
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """前向函数调用"""
        R, H = self._decode_parameters(x)
        Q_all = self.forward_model(R, H)
        return Q_all[self.measurement_indices]
    
    def __getstate__(self):
        """确保 forward_model 可序列化，否则抛出异常"""
        try:
            pickle.dumps(self.forward_model)
        except Exception as e:
            raise pickle.PicklingError(
                f"forward_model 无法序列化: {e}"
            )
        return self.__dict__


class RealDataInversionConfig:
    """
    实际数据反演问题配置器
    
    支持两种反演模式:
    1. 仅反演风阻 R
    2. 同时反演风阻 R 和风机压力 H
    """
    
    def __init__(
        self,
        network_data: VentilationNetworkData,
        forward_model: Callable[[np.ndarray, np.ndarray], np.ndarray],
        optimize_fan_pressure: bool = False,
        loss_fn: Optional[LossFunction] = None,
        use_log_scale: bool = True,
        relative_bounds: bool = True
    ):
        """
        参数:
        - network_data: 加载的网络数据
        - forward_model: 前向模型函数 (R_array, H_array) -> Q_pred_array
        - optimize_fan_pressure: 是否同时优化风机压力
        - loss_fn: 损失函数（默认 Huber）
        - use_log_scale: 是否使用对数尺度优化（推荐用于跨度大的参数）
        - relative_bounds: 是否使用相对边界（基于r0）
        """
        self.network = network_data
        self.forward_model = forward_model
        self.optimize_fan = optimize_fan_pressure
        self.loss_fn = loss_fn or HuberLoss(delta=1.0)
        self.use_log_scale = use_log_scale
        self.relative_bounds = relative_bounds
        
        # 构建参数映射
        self._build_parameter_mapping()
        
        # 构建目标值
        self._build_targets()
    
    def _build_parameter_mapping(self):
        """构建参数索引映射"""
        # 风阻参数
        self.r_indices = list(range(self.network.n_roads))
        self.n_r_params = self.network.n_roads
        
        # 风机参数（如果需要优化）
        if self.optimize_fan:
            # 只优化有范围的风机
            self.fan_to_optimize = [
                f for f in self.network.fans 
                if f.min_h != f.max_h
            ]
            self.n_h_params = len(self.fan_to_optimize)
            self.h_start_idx = self.n_r_params
        else:
            self.fan_to_optimize = []
            self.n_h_params = 0
            self.h_start_idx = self.n_r_params
        
        self.total_params = self.n_r_params + self.n_h_params
        
        logger.info(f"参数配置:")
        logger.info(f"  风阻参数: {self.n_r_params}")
        logger.info(f"  风机参数: {self.n_h_params}")
        logger.info(f"  总参数数: {self.total_params}")
    
    def _build_targets(self):
        """构建观测目标值"""
        self.measurement_indices = []
        self.target_values = []
        self.measurement_weights = []
        
        for i, road in enumerate(self.network.roads):
            if road.target_q is not None:
                self.measurement_indices.append(i)
                self.target_values.append(road.target_q)
                # 使用倒数作为权重（大风量测点权重低）
                self.measurement_weights.append(1.0 / (abs(road.target_q) + 1.0))
        
        self.target_values = np.array(self.target_values)
        self.measurement_weights = np.array(self.measurement_weights)
        # 归一化权重
        self.measurement_weights /= np.sum(self.measurement_weights)
        
        logger.info(f"目标值配置:")
        logger.info(f"  测点数: {len(self.target_values)}")
        logger.info(f"  目标风量范围: [{self.target_values.min():.2f}, {self.target_values.max():.2f}]")
    
    def get_initial_guess(self) -> np.ndarray:
        """获取初始猜测值（使用r0和h0）"""
        x0 = np.zeros(self.total_params)
        
        # 风阻初始值
        for i, road in enumerate(self.network.roads):
            if self.use_log_scale:
                x0[i] = np.log(road.r0)
            else:
                x0[i] = road.r0
        
        # 风机初始值
        if self.optimize_fan:
            for j, fan in enumerate(self.fan_to_optimize):
                idx = self.h_start_idx + j
                x0[idx] = fan.h0
        
        return x0
    
    def get_bounds(self) -> InversionBounds:
        """获取参数边界"""
        lower = np.zeros(self.total_params)
        upper = np.zeros(self.total_params)
        
        # 风阻边界
        for i, road in enumerate(self.network.roads):
            # 确保边界有效
            min_r = max(road.min_r, 1e-12)  # 避免零或负值
            max_r = max(road.max_r, min_r * 1.01)  # 确保 max > min
            
            # 确保 r0 在边界内
            if road.r0 < min_r:
                min_r = road.r0 * 0.5
            if road.r0 > max_r:
                max_r = road.r0 * 2.0
            
            if self.use_log_scale:
                lower[i] = np.log(min_r)
                upper[i] = np.log(max_r)
            else:
                lower[i] = min_r
                upper[i] = max_r
        
        # 风机边界
        if self.optimize_fan:
            for j, fan in enumerate(self.fan_to_optimize):
                idx = self.h_start_idx + j
                # 注意：风机压力通常为负值
                min_h = min(fan.min_h, fan.max_h)
                max_h = max(fan.min_h, fan.max_h)
                
                # 确保 h0 在边界内
                if fan.h0 < min_h:
                    min_h = fan.h0 * 1.1 if fan.h0 < 0 else fan.h0 * 0.9
                if fan.h0 > max_h:
                    max_h = fan.h0 * 0.9 if fan.h0 < 0 else fan.h0 * 1.1
                
                # 确保边界有效
                if min_h >= max_h:
                    # 如果边界无效，创建一个小范围
                    center = fan.h0
                    half_range = abs(center) * 0.1 + 1.0
                    min_h = center - half_range
                    max_h = center + half_range
                
                lower[idx] = min_h
                upper[idx] = max_h
        
        return InversionBounds(lower=lower, upper=upper)
    
    def decode_parameters(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将优化变量解码为物理参数
        
        返回:
        - R_array: 风阻数组
        - H_array: 风机压力数组
        """
        # 解码风阻
        if self.use_log_scale:
            R = np.exp(x[:self.n_r_params])
        else:
            R = x[:self.n_r_params].copy()
        
        # 解码/填充风机压力
        H = np.array([f.h0 for f in self.network.fans])
        if self.optimize_fan:
            for j, fan in enumerate(self.fan_to_optimize):
                idx = self.h_start_idx + j
                # 找到该风机在完整列表中的位置
                fan_idx = self.network.fans.index(fan)
                H[fan_idx] = x[idx]
        
        return R, H
    
    def create_objective_function(self) -> Callable[[np.ndarray], float]:
        """
        创建目标函数
        
        返回:
        - objective_fn: 目标函数 x -> loss
        """
        def objective(x: np.ndarray) -> float:
            try:
                # 解码参数
                R, H = self.decode_parameters(x)
                
                # 调用前向模型
                Q_pred_all = self.forward_model(R, H)
                
                # 提取测点预测值
                Q_pred = Q_pred_all[self.measurement_indices]
                
                # 计算损失
                loss = self.loss_fn(Q_pred, self.target_values, self.measurement_weights)
                
                return loss
            except Exception as e:
                logger.warning(f"目标函数评估失败: {e}")
                return 1e10
        
        return objective
    
    def create_inversion_problem(self) -> InversionProblem:
        """
        创建完整的反演问题实例
        
        注意：为了支持并行计算，forward_fn 使用模块级别的可序列化类而非闭包
        """
        bounds = self.get_bounds()
        
        # 使用模块级别的可序列化类（支持并行计算）
        forward_fn = SerializableForwardFn(self)
        
        # 参数名称
        param_names = [f"R_{r.id}" for r in self.network.roads]
        if self.optimize_fan:
            param_names += [f"H_{f.id}" for f in self.fan_to_optimize]
        
        return InversionProblem(
            forward_fn=forward_fn,
            y_target=self.target_values,
            bounds=bounds,
            loss_fn=self.loss_fn,
            measurement_weights=self.measurement_weights,
            param_names=param_names
        )


# ============================================================================
# 简化的前向模型（用于测试）
# ============================================================================

class SimplifiedForwardModel:
    """
    简化的前向模型（用于测试和演示）
    
    基于假设: h = R * Q^2，在总压力不变的情况下
    Q_i ∝ 1 / sqrt(R_i)
    
    实际应用中需要替换为真正的风网求解器
    """
    
    def __init__(self, network: VentilationNetworkData):
        self.network = network
        # 使用初始风量和风阻作为参考
        self.q_ref = np.array([r.init_q for r in network.roads])
        self.r_ref = np.array([r.r0 for r in network.roads])
        
        # 计算参考压力降 h_ref = R_ref * Q_ref^2
        self.h_ref = self.r_ref * self.q_ref ** 2
        
    def __call__(self, R: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        简化的前向计算
        
        参数:
        - R: 风阻数组 (n_roads,)
        - H: 风机压力数组 (n_fans,)
        
        返回:
        - Q: 风量数组 (n_roads,)
        """
        # 假设压力降保持不变: h = R * Q^2 = h_ref
        # 则 Q = sqrt(h_ref / R)
        Q = np.sqrt(np.maximum(self.h_ref, 1e-10) / (R + 1e-10))
        
        # 确保风量为正
        Q = np.maximum(Q, 0.01)
        
        return Q


class IdentityForwardModel:
    """
    恒等前向模型（用于验证反演框架）
    
    直接返回初始风量，用于测试反演框架是否正确
    """
    
    def __init__(self, network: VentilationNetworkData):
        self.network = network
        self.q_init = np.array([r.init_q for r in network.roads])
        
    def __call__(self, R: np.ndarray, H: np.ndarray) -> np.ndarray:
        return self.q_init.copy()


# ============================================================================
# MVN Solver 集成（真实风网求解器）
# ============================================================================

class MVNSolverWrapper:
    """
    MVN Solver 包装器
    
    集成 jl.vn.ns.ns.mvn_solver 风网求解器
    
    使用方式:
    ```python
    from real_data_inversion import MVNSolverWrapper, DataLoader, run_real_data_demo
    
    network = DataLoader.load("input.json")
    forward_model = MVNSolverWrapper(network, json_path="input.json")
    
    result, config, network = run_real_data_demo(
        json_path="input.json",
        forward_model=forward_model
    )
    ```
    """
    
    def __init__(
        self, 
        network: VentilationNetworkData,
        json_path: str = "input.json",
        config_ns: Optional[Dict] = None,
        calcul_weight_fn: Optional[Callable] = None
    ):
        """
        参数:
        - network: 加载的网络数据
        - json_path: 原始JSON数据路径（用于获取完整数据）
        - config_ns: 网络解算控制参数
        - calcul_weight_fn: 权重计算函数（可选）
        """
        self.network = network
        self.json_path = json_path
        
        # 加载原始JSON数据
        with open(json_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 网络解算配置
        self.config_ns = config_ns or {
            "loopN": 20,
            "iterN": 30,
            "minQ": 0.05,
            "minH": 0.1
        }
        
        # 权重计算函数
        self.calcul_weight_fn = calcul_weight_fn
        
        # 构建ID到索引的映射
        self.road_id_to_idx = {r.id: i for i, r in enumerate(network.roads)}
        self.fan_id_to_idx = {f.id: i for i, f in enumerate(network.fans)}
        
        # 缓存初始数据
        self._prepare_base_data()
        
        logger.info(f"MVNSolverWrapper 初始化完成")
        logger.info(f"  巷道数: {len(self.network.roads)}")
        logger.info(f"  风机数: {len(self.network.fans)}")
    
    def _prepare_base_data(self):
        """准备基础数据结构"""
        logger.info(f"准备基础数据结构")
        # 复制roads数据，转换字段名
        self.base_roads = []
        for road_data in self.raw_data.get('roads', []):
            road = {
                'id': road_data['id'],
                's': road_data.get('s', ''),
                't': road_data.get('t', ''),
                'r': road_data['r0'],  # 使用 r0 作为初始风阻
                'ex': road_data.get('ex', 2.0),
                'initQ': road_data.get('initQ', 1.0),
                'weight': road_data.get('weight', 1.0)
            }
            # 如果有固定风量
            if 'fixedQ' in road_data:
                road['fixedQ'] = road_data['fixedQ']
            self.base_roads.append(road)
        
        # 复制fanHs数据
        self.base_fan_hs = []
        for fan_data in self.raw_data.get('fanHs', []):
            fan = {
                'id': fan_data['id'],
                'eid': fan_data['eid'],
                'h': fan_data['h0'],  # 使用 h0 作为初始风机压力
                'direction': fan_data.get('direction', 'forward'),
                'pitotLocation': fan_data.get('pitotLocation', 'in')
            }
            self.base_fan_hs.append(fan)
        
        # 复制structureHs数据
        self.base_structure_hs = []
        for struct_data in self.raw_data.get('structureHs', []):
            struct = {
                'id': struct_data['id'],
                'eid': struct_data['eid'],
                'h': struct_data['h']
            }
            self.base_structure_hs.append(struct)
        
        # 构建巷道ID到roads列表索引的映射
        self.road_id_to_list_idx = {r['id']: i for i, r in enumerate(self.base_roads)}
        
        # 构建风机ID到fanHs列表索引的映射
        self.fan_id_to_list_idx = {f['id']: i for i, f in enumerate(self.base_fan_hs)}
    
    def _create_solver_data(self, R: np.ndarray, H: np.ndarray) -> Dict:
        """
        根据优化变量创建求解器输入数据
        
        参数:
        - R: 风阻数组 (n_roads,)
        - H: 风机压力数组 (n_fans,)
        
        返回:
        - dataNS: 求解器输入数据字典
        """

        
        # 深拷贝基础数据
        roads = copy.deepcopy(self.base_roads)
        fan_hs = copy.deepcopy(self.base_fan_hs) if self.base_fan_hs else None
        structure_hs = copy.deepcopy(self.base_structure_hs) if self.base_structure_hs else None
        
        # 更新风阻值
        for i, road in enumerate(roads):
            road['r'] = float(R[i])
        
        # 更新风机压力
        if fan_hs and len(H) > 0:
            for i, fan in enumerate(fan_hs):
                if i < len(H):
                    fan['h'] = float(H[i])
        
        # 计算权重（如果提供了权重计算函数）
        if self.calcul_weight_fn is not None:
            init_qs = {road['id']: road['initQ'] for road in roads}
            weights = self.calcul_weight_fn(initQs=init_qs, roads=roads)
            for road in roads:
                road['weight'] = weights.get(road['id'], road.get('weight', 1.0))
        
        return {
            'roads': roads,
            'fanHs': fan_hs,
            'structureHs': structure_hs
        }
    
    def _default_calcul_weight(self, initQs: Dict, roads: List) -> Dict:
        """默认的权重计算函数（避免使用 lambda，以便序列化）"""
        from jl.vn.ns.calcul_weight import calcul_weight
        return calcul_weight(initQs=initQs, weightType="R*Q", roads=roads)
    
    def __call__(self, R: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        调用 mvn_solver 求解风网
        
        参数:
        - R: 风阻数组 (n_roads,)
        - H: 风机压力数组 (n_fans,)
        
        返回:
        - Q: 风量数组 (n_roads,)
        """
        # 导入求解器（延迟导入）
        from jl.vn.ns.ns import mvn_solver
        
        # 如果没有设置权重计算函数，使用默认方法（避免 lambda）
        if self.calcul_weight_fn is None:
            self.calcul_weight_fn = self._default_calcul_weight
        
        # 创建求解器输入数据
        data_ns = self._create_solver_data(R, H)
        
        # 调用求解器
        result = mvn_solver(
            roads=data_ns['roads'],
            fanHs=data_ns.get('fanHs'),
            structureHs=data_ns.get('structureHs'),
            configNS=self.config_ns
        )
        
        # 检查返回状态
        if result.get('state') == 'error':
            raise RuntimeError(f"MVN Solver 返回错误: {result.get('message', 'Unknown error')}")
        
        # 提取风量结果
        road_qs = result.get('roadQs', {})
        
        # 转换为数组格式（按照network.roads的顺序）
        Q = np.zeros(len(self.network.roads))
        for i, road in enumerate(self.network.roads):
            Q[i] = road_qs.get(road.id, self.network.roads[i].init_q)
        
        return Q


class MVNSolverWrapperSimple:
    """
    MVN Solver 简化包装器
    
    直接使用字典回调方式，适合快速集成
    
    使用方式:
    ```python
    from jl.vn.ns.ns import mvn_solver
    from jl.vn.ns.calcul_weight import calcul_weight
    
    def solve_network(R_dict, H_dict):
        # 准备数据并调用 mvn_solver
        result = mvn_solver(...)
        return result['roadQs']
    
    wrapper = MVNSolverWrapperSimple(network, solve_network)
    ```
    """
    
    def __init__(
        self,
        network: VentilationNetworkData,
        solver_callback: Callable[[Dict[str, float], Dict[str, float]], Dict[str, float]]
    ):
        """
        参数:
        - network: 网络数据
        - solver_callback: 求解器回调函数
            输入: R_dict (巷道ID -> 风阻), H_dict (风机ID -> 压力)
            输出: Q_dict (巷道ID -> 风量)
        """
        self.network = network
        self.solver_callback = solver_callback
        self.road_ids = [r.id for r in network.roads]
        self.fan_ids = [f.id for f in network.fans]
    
    def __call__(self, R: np.ndarray, H: np.ndarray) -> np.ndarray:
        # 转换为字典格式
        R_dict = {self.road_ids[i]: float(R[i]) for i in range(len(R))}
        H_dict = {self.fan_ids[i]: float(H[i]) for i in range(len(H))}
        
        # 调用求解器
        Q_dict = self.solver_callback(R_dict, H_dict)
        
        # 转换回数组格式
        Q = np.array([Q_dict.get(rid, 0.0) for rid in self.road_ids])
        
        return Q


# ============================================================================
# 高级风网求解器接口（需要用户实现）
# ============================================================================

class VentilationSolverInterface:
    """
    风网求解器接口（抽象类）
    
    用户需要实现此接口，连接到实际的风网求解器
    例如：VENTSIM, ICAMPS, 或自定义的 Hardy-Cross 求解器
    """
    
    def __init__(self, network: VentilationNetworkData):
        self.network = network
    
    def solve(self, R: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        求解风网方程
        
        参数:
        - R: 各巷道风阻 (n_roads,)
        - H: 各风机压力 (n_fans,)
        
        返回:
        - Q: 各巷道风量 (n_roads,)
        """
        raise NotImplementedError("请实现具体的风网求解器")


class ExternalSolverWrapper:
    """
    外部求解器包装器
    
    通过文件/API调用外部风网求解程序
    
    使用示例:
    ```python
    def my_solver_callback(R_dict, H_dict):
        # 调用外部程序
        # 返回 Q_dict
        pass
    
    wrapper = ExternalSolverWrapper(network, my_solver_callback)
    Q = wrapper(R, H)
    ```
    """
    
    def __init__(self, 
                 network: VentilationNetworkData,
                 solver_callback: Callable[[Dict[str, float], Dict[str, float]], Dict[str, float]]):
        """
        参数:
        - network: 网络数据
        - solver_callback: 求解器回调函数
            输入: R_dict (巷道ID -> 风阻), H_dict (风机ID -> 压力)
            输出: Q_dict (巷道ID -> 风量)
        """
        self.network = network
        self.solver_callback = solver_callback
        self.road_ids = [r.id for r in network.roads]
        self.fan_ids = [f.id for f in network.fans]
    
    def __call__(self, R: np.ndarray, H: np.ndarray) -> np.ndarray:
        # 转换为字典格式
        R_dict = {self.road_ids[i]: float(R[i]) for i in range(len(R))}
        H_dict = {self.fan_ids[i]: float(H[i]) for i in range(len(H))}
        
        # 调用外部求解器
        Q_dict = self.solver_callback(R_dict, H_dict)
        
        # 转换回数组格式
        Q = np.array([Q_dict.get(rid, 0.0) for rid in self.road_ids])
        
        return Q


# ============================================================================
# 主运行函数
# ============================================================================

def run_real_data_demo(
    json_path: str = "input.json",
    forward_model: Optional[Callable] = None,
    optimize_fan: bool = False,
    max_iter: int = 500,
    use_log_scale: bool = True
):
    """
    运行实际数据反演
    
    参数:
    - json_path: JSON数据文件路径
    - forward_model: 自定义前向模型（None则使用简化模型）
    - optimize_fan: 是否同时优化风机压力
    - max_iter: 最大迭代次数
    - use_log_scale: 是否使用对数尺度
    """
    print("="*70)
    print("矿井通风网络阻力系数反演 - 实际数据模式")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    network = DataLoader.load(json_path)
    
    # 2. 创建前向模型
    print("\n[2] 创建前向模型...")
    if forward_model is None:
        forward_model = SimplifiedForwardModel(network)
        print("  使用简化前向模型（仅用于演示）")
        print("  注意：实际应用需要接入真实风网求解器")
    else:
        print("  使用自定义前向模型")
    
    # 3. 配置反演问题
    print("\n[3] 配置反演问题...")
    inv_config = RealDataInversionConfig(
        network_data=network,
        forward_model=forward_model,
        optimize_fan_pressure=optimize_fan,
        loss_fn=HuberLoss(delta=1.0),
        use_log_scale=use_log_scale,
    )
    
    # 4. 获取初始值和边界
    x0 = inv_config.get_initial_guess()
    bounds = inv_config.get_bounds()
    
    print(f"\n参数统计:")
    print(f"  总参数数: {inv_config.total_params}")
    print(f"  风阻参数: {inv_config.n_r_params}")
    print(f"  风机参数: {inv_config.n_h_params}")
    if use_log_scale:
        print(f"  使用对数尺度优化")
        print(f"  log(R) 初始范围: [{x0[:inv_config.n_r_params].min():.4f}, {x0[:inv_config.n_r_params].max():.4f}]")
    else:
        print(f"  R 初始范围: [{x0[:inv_config.n_r_params].min():.6e}, {x0[:inv_config.n_r_params].max():.6e}]")
    
    # 5. 创建反演问题
    problem = inv_config.create_inversion_problem()
    
    # 6. 配置 CMA-ES
    print("\n[4] 配置 CMA-ES 优化器...")
    
    # 根据问题维度调整参数
    dim = inv_config.total_params
    if dim > 500:
        # 高维问题使用较小的种群
        popsize = min(50, 4 + int(3 * np.log(dim)))
    else:
        popsize = None  # 自动选择
    
    cma_config = CMAESConfig(
        sigma0=0.3 if use_log_scale else None,  # 对数尺度使用较小步长
        maxiter=max_iter,
        maxfevals=max_iter * 50,
        tolx=1e-8,
        tolfun=1e-10,
        popsize=popsize,
        verbose=1,
        seed=42
    )
    
    optimizer = CMAESOptimizer(problem, cma_config)
    
    # 7. 运行优化
    print("\n[5] 运行 CMA-ES 优化...")
    print(f"  参数维度: {dim}")
    print(f"  测点数量: {len(inv_config.target_values)}")
    print(f"  最大迭代: {max_iter}")
    
    result = optimizer.run(x0)
    
    # 8. 解码结果
    R_opt, H_opt = inv_config.decode_parameters(result.x_best)
    
    # 9. 输出结果
    print("\n" + "="*70)
    print("反演结果")
    print("="*70)
    print(f"最终损失: {result.loss_best:.6e}")
    print(f"迭代次数: {result.convergence_info['iterations']}")
    print(f"函数评估: {result.n_evaluations}")
    print(f"运行时间: {result.elapsed_time:.2f}秒")
    
    # 比较优化前后的风阻
    print("\n风阻变化统计:")
    r0_values = np.array([r.r0 for r in network.roads])
    r_change = (R_opt - r0_values) / (r0_values + 1e-10) * 100
    
    print(f"  平均变化: {np.mean(np.abs(r_change)):.2f}%")
    print(f"  最大变化: {np.max(np.abs(r_change)):.2f}%")
    print(f"  变化超过10%的巷道数: {np.sum(np.abs(r_change) > 10)}/{network.n_roads}")
    print(f"  变化超过50%的巷道数: {np.sum(np.abs(r_change) > 50)}/{network.n_roads}")
    
    # 输出变化最大的几条巷道
    top_change_idx = np.argsort(np.abs(r_change))[-10:][::-1]
    print("\n变化最大的10条巷道:")
    for idx in top_change_idx:
        road = network.roads[idx]
        print(f"  {road.id}: {road.r0:.6e} -> {R_opt[idx]:.6e} ({r_change[idx]:+.1f}%)")
    
    # 检查测点拟合效果
    Q_pred = forward_model(R_opt, H_opt)
    Q_pred_meas = Q_pred[inv_config.measurement_indices]
    
    residuals = Q_pred_meas - inv_config.target_values
    rel_errors = np.abs(residuals) / (np.abs(inv_config.target_values) + 0.1) * 100
    
    print("\n测点拟合效果:")
    print(f"  平均相对误差: {np.mean(rel_errors):.2f}%")
    print(f"  最大相对误差: {np.max(rel_errors):.2f}%")
    print(f"  误差<5%的测点: {np.sum(rel_errors < 5)}/{len(rel_errors)}")
    print(f"  误差<10%的测点: {np.sum(rel_errors < 10)}/{len(rel_errors)}")
    print(f"  RMSE: {np.sqrt(np.mean(residuals**2)):.4f} m³/s")
    
    # 显示拟合最差的几个测点
    worst_idx = np.argsort(rel_errors)[-5:][::-1]
    print("\n拟合最差的5个测点:")
    for idx in worst_idx:
        road_idx = inv_config.measurement_indices[idx]
        road = network.roads[road_idx]
        print(f"  {road.id}: 目标={inv_config.target_values[idx]:.2f}, "
              f"预测={Q_pred_meas[idx]:.2f}, 误差={rel_errors[idx]:.1f}%")
    
    return result, inv_config, network


def export_results(result: InversionResult, 
                   config: RealDataInversionConfig,
                   network: VentilationNetworkData,
                   output_path: str = "inversion_results.json"):
    """
    导出反演结果到JSON文件
    """
    R_opt, H_opt = config.decode_parameters(result.x_best)
    
    results_data = {
        "summary": {
            "final_loss": result.loss_best,
            "iterations": result.convergence_info['iterations'],
            "elapsed_time": result.elapsed_time,
            "n_roads": network.n_roads,
            "n_measurements": network.n_measurements
        },
        "optimized_roads": [],
        "optimized_fans": []
    }
    
    # 导出优化后的风阻
    for i, road in enumerate(network.roads):
        results_data["optimized_roads"].append({
            "id": road.id,
            "r0": road.r0,
            "r_optimized": float(R_opt[i]),
            "change_percent": float((R_opt[i] - road.r0) / road.r0 * 100),
            "min_r": road.min_r,
            "max_r": road.max_r
        })
    
    # 导出优化后的风机压力
    if config.optimize_fan:
        for j, fan in enumerate(config.fan_to_optimize):
            fan_idx = network.fans.index(fan)
            results_data["optimized_fans"].append({
                "id": fan.id,
                "h0": fan.h0,
                "h_optimized": float(H_opt[fan_idx]),
                "min_h": fan.min_h,
                "max_h": fan.max_h
            })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果已导出到: {output_path}")


# ============================================================================
# 使用示例
# ============================================================================

def example_with_mvn_solver():
    """
    示例：使用 mvn_solver 真实风网求解器进行反演
    
    这是推荐的使用方式，直接集成 jl.vn.ns.ns.mvn_solver
    """
    print("="*70)
    print("示例：使用 MVN Solver 进行反演")
    print("="*70)
    
    json_path = "input.json"
    
    # 1. 加载网络数据
    network = DataLoader.load(json_path)
    
    # 2. 创建 MVN Solver 包装器
    try:
        forward_model = MVNSolverWrapper(
            network=network,
            json_path=json_path,
            config_ns={
                "loopN": 20,
                "iterN": 30,
                "minQ": 0.05,
                "minH": 0.1
            }
        )
        print("  MVN Solver 包装器创建成功")
    except ImportError as e:
        print(f"  无法导入 MVN Solver: {e}")
        print("  将使用简化模型代替")
        forward_model = SimplifiedForwardModel(network)
    
    # 3. 运行反演
    result, config, network = run_real_data_demo(
        json_path=json_path,
        forward_model=forward_model,
        optimize_fan=False,
        max_iter=200,
        use_log_scale=True
    )
    
    return result, config, network


def example_with_mvn_solver_manual():
    """
    示例：手动集成 mvn_solver（更灵活的方式）
    
    展示如何参考 FuncFitness.py 的方式集成求解器
    """
    print("="*70)
    print("示例：手动集成 MVN Solver")
    print("="*70)
    
    import copy
    
    json_path = "input.json"
    
    # 1. 加载原始数据
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 2. 加载网络结构
    network = DataLoader.load(json_path)
    
    # 3. 准备基础数据（参考 FuncFitness.py）
    base_roads = []
    for road_data in raw_data.get('roads', []):
        road = {
            'id': road_data['id'],
            's': road_data.get('s', ''),
            't': road_data.get('t', ''),
            'r': road_data['r0'],          # r0 -> r
            'ex': road_data.get('ex', 2.0),
            'initQ': road_data.get('initQ', 1.0),
            'weight': road_data.get('weight', 1.0)
        }
        base_roads.append(road)
    
    base_fan_hs = []
    for fan_data in raw_data.get('fanHs', []):
        fan = {
            'id': fan_data['id'],
            'eid': fan_data['eid'],
            'h': fan_data['h0'],           # h0 -> h
            'direction': fan_data.get('direction', 'forward'),
            'pitotLocation': fan_data.get('pitotLocation', 'in')
        }
        base_fan_hs.append(fan)
    
    base_structure_hs = raw_data.get('structureHs', [])
    
    config_ns = {
        "loopN": 20,
        "iterN": 30,
        "minQ": 0.05,
        "minH": 0.1
    }
    
    # 4. 定义前向模型函数
    def forward_with_mvn_solver(R: np.ndarray, H: np.ndarray) -> np.ndarray:
        """
        使用 mvn_solver 的前向模型
        
        参考 FuncFitness.py 中的 FuncFinness 函数
        """
        try:
            from jl.vn.ns.ns import mvn_solver
            from jl.vn.ns.calcul_weight import calcul_weight
        except ImportError as e:
            logger.error(f"无法导入 mvn_solver: {e}")
            raise
        
        # 深拷贝数据
        roads = copy.deepcopy(base_roads)
        fan_hs = copy.deepcopy(base_fan_hs) if base_fan_hs else None
        structure_hs = copy.deepcopy(base_structure_hs) if base_structure_hs else None
        
        # 将优化变量 R 赋给 road['r']（参考 FuncFitness.py 第61-63行）
        for i, road in enumerate(roads):
            road['r'] = float(R[i])
        
        # 将优化变量 H 赋给 fanH['h']（参考 FuncFitness.py 第64-67行）
        if fan_hs and len(H) > 0:
            for i, fan in enumerate(fan_hs):
                if i < len(H):
                    fan['h'] = float(H[i])
        
        # 计算风路权重（参考 FuncFitness.py 第70-73行）
        init_qs = {road['id']: road['initQ'] for road in roads}
        weights = calcul_weight(initQs=init_qs, weightType="R*Q", roads=roads)
        for road in roads:
            road['weight'] = weights[road['id']]
        
        # 调用网络解算（参考 FuncFitness.py 第76-79行）
        result = mvn_solver(
            roads=roads,
            fanHs=fan_hs,
            structureHs=structure_hs,
            configNS=config_ns
        )
        
        # 提取风量结果
        road_qs = result['roadQs']
        
        # 转换为数组格式
        Q = np.zeros(len(network.roads))
        for i, road in enumerate(network.roads):
            Q[i] = road_qs.get(road.id, 0.0)
        
        return Q
    
    # 5. 运行反演
    print("  前向模型定义完成，开始反演...")
    
    result, config, network = run_real_data_demo(
        json_path=json_path,
        forward_model=forward_with_mvn_solver,
        optimize_fan=False,
        max_iter=200,
        use_log_scale=True
    )
    
    return result, config, network


def example_with_custom_solver():
    """
    示例：使用自定义风网求解器进行反演（通用接口）
    
    这个示例展示了如何接入任意风网求解器
    """
    print("="*70)
    print("示例：接入自定义风网求解器")
    print("="*70)
    
    # 1. 加载网络数据
    network = DataLoader.load("input.json")
    
    # 2. 定义您的求解器回调函数
    def my_solver_callback(R_dict: Dict[str, float], 
                           H_dict: Dict[str, float]) -> Dict[str, float]:
        """
        您的风网求解器接口
        
        参数:
        - R_dict: {巷道ID: 风阻值}
        - H_dict: {风机ID: 风机压力}
        
        返回:
        - Q_dict: {巷道ID: 风量}
        """
        # 示例实现（请替换为您的实际求解器）
        Q_dict = {}
        for road_id, R in R_dict.items():
            Q_dict[road_id] = 10.0  # 占位值
        return Q_dict
    
    # 3. 创建求解器包装器
    forward_model = ExternalSolverWrapper(network, my_solver_callback)
    
    # 4. 运行反演
    result, config, network = run_real_data_demo(
        json_path="input.json",
        forward_model=forward_model,
        optimize_fan=False,
        max_iter=100
    )
    
    return result


def example_parameter_extraction():
    """
    示例：仅提取和显示数据，不运行优化
    """
    print("="*70)
    print("数据提取示例")
    print("="*70)
    
    # 加载数据
    network = DataLoader.load("input.json")
    
    # 提取所有风阻参数
    r0_values = np.array([r.r0 for r in network.roads])
    min_r_values = np.array([r.min_r for r in network.roads])
    max_r_values = np.array([r.max_r for r in network.roads])
    
    print(f"\n风阻参数 (R0):")
    print(f"  数量: {len(r0_values)}")
    print(f"  范围: [{r0_values.min():.6e}, {r0_values.max():.6e}]")
    print(f"  中位数: {np.median(r0_values):.6e}")
    
    # 计算边界宽度
    bound_ratio = max_r_values / min_r_values
    print(f"\n边界比例 (maxR/minR):")
    print(f"  平均: {np.mean(bound_ratio):.2f}")
    print(f"  所有巷道边界比例相同: {np.allclose(bound_ratio, bound_ratio[0])}")
    
    # 提取测点信息
    target_roads = [r for r in network.roads if r.target_q is not None]
    print(f"\n测点信息:")
    print(f"  测点数: {len(target_roads)}")
    print(f"  目标风量范围: [{min(r.target_q for r in target_roads):.2f}, "
          f"{max(r.target_q for r in target_roads):.2f}] m³/s")
    
    # 提取风机信息
    print(f"\n风机信息:")
    print(f"  风机数: {len(network.fans)}")
    main_fans = [f for f in network.fans if f.use == "MAIN"]
    local_fans = [f for f in network.fans if f.use == "LOCAL"]
    print(f"  主扇: {len(main_fans)}")
    print(f"  局扇: {len(local_fans)}")
    
    return network


# ============================================================================
# 程序入口
# ============================================================================

def main():
    """主函数"""
    import sys
    
    # 解析命令行参数
    json_path = "input.json"
    use_mvn_solver = False
    max_iter = 200
    optimize_fan = False
    
    for arg in sys.argv[1:]:
        if arg == "--mvn":
            use_mvn_solver = True
        elif arg.startswith("--iter="):
            max_iter = int(arg.split("=")[1])
        elif arg == "--optimize-fan":
            optimize_fan = True
        elif arg.endswith(".json"):
            json_path = arg
    
    try:
        # 先显示数据统计
        print("\n" + "="*70)
        print("第一步：数据分析")
        print("="*70)
        network = example_parameter_extraction()
        
        print("\n" + "="*70)
        print("第二步：运行反演")
        print("="*70)
        
        # 选择前向模型
        if use_mvn_solver:
            print("\n使用 MVN Solver 真实风网求解器")
            try:
                forward_model = MVNSolverWrapper(
                    network=network,
                    json_path=json_path,
                    config_ns={
                        "loopN": 20,
                        "iterN": 30,
                        "minQ": 0.05,
                        "minH": 0.1
                    }
                )
            except ImportError as e:
                print(f"无法导入 MVN Solver: {e}")
                print("请确保 jl.vn.ns 模块在 Python 路径中")
                print("回退到简化模型...")
                forward_model = None
        else:
            print("\n使用简化前向模型（演示用）")
            print("如需使用真实求解器，请添加 --mvn 参数")
            forward_model = None
        
        if optimize_fan:
            print("\n启用风机压力优化（fanHs）")
        
        result, config, network = run_real_data_demo(
            json_path=json_path,
            forward_model=forward_model,
            optimize_fan=optimize_fan,
            max_iter=max_iter,
            use_log_scale=True
        )
        
        # 导出结果
        export_results(result, config, network, "inversion_results.json")
        
        print("\n" + "="*70)
        print("完成！")
        print("="*70)
        print("结果已保存到 inversion_results.json")
        
        if not use_mvn_solver:
            print("\n提示：使用真实风网求解器运行命令：")
            print(f"  python real_data_inversion.py --mvn --iter={max_iter}")
        
        if not optimize_fan:
            print("\n提示：同时优化风机压力运行命令：")
            print(f"  python real_data_inversion.py --mvn --optimize-fan --iter={max_iter}")
        
    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {json_path}")
        print("请确保 input.json 存在于当前目录")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

