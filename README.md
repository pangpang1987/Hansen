# 矿井通风网络阻力系数反演系统

使用 **CMA-ES (Covariance Matrix Adaptation Evolution Strategy)** 求解矿井通风网络的阻力系数反演问题。

## 📋 项目结构

```
Hansen/
├── ventilation_inversion.py    # 核心反演模块
├── real_data_inversion.py      # 实际数据反演模块 ⭐
├── visualization.py            # 可视化工具
├── input.json                  # 实际风网数据
├── TECHNICAL_SPECIFICATION.md  # 详细技术规范文档
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行演示

```bash
# 使用实际数据运行反演 ⭐
python real_data_inversion.py

# 运行简化模型演示
python ventilation_inversion.py

# 生成可视化图表
python visualization.py
```

## 📖 技术概述

### 问题定义

**目标**：给定通风网络的观测数据（风量/风压/风速），反演各巷道的摩擦阻力系数 $k_i$

**数学模型**：
$$\mathbf{k}^* = \arg\min_{\mathbf{k}} \mathcal{L}(f(\mathbf{k}), \mathbf{y}_{\text{target}})$$

其中：
- $\mathbf{k} = [k_1, k_2, ..., k_N]^T$：待反演的阻力系数向量
- $f(\cdot)$：前向模型（风网求解器）
- $\mathbf{y}_{\text{target}}$：观测值
- $\mathcal{L}(\cdot)$：损失函数

### 为什么选择 CMA-ES？

| 问题特点 | CMA-ES 优势 |
|---------|-------------|
| 高维连续参数 | 专为连续优化设计，10-100维表现优异 |
| 非凸黑箱问题 | 无需梯度，自适应搜索方向 |
| 参数耦合 | 协方差矩阵捕捉参数相关性 |
| 测量噪声 | 种群评估提供噪声鲁棒性 |

## 🔧 核心模块

### 1. 损失函数

```python
from ventilation_inversion import MSELoss, MAELoss, HuberLoss

# 推荐使用 Huber Loss（兼顾鲁棒性和平滑性）
loss_fn = HuberLoss(delta=1.0)
```

| 损失函数 | 适用场景 |
|---------|---------|
| MSE | 高质量数据，无异常值 |
| MAE | 存在离群点，需要鲁棒性 |
| **Huber** | **推荐默认**，兼顾两者优点 |

### 2. 反演问题定义

```python
from ventilation_inversion import InversionProblem, InversionBounds

# 定义参数边界
bounds = InversionBounds(
    lower=np.array([0.001] * n_params),
    upper=np.array([0.010] * n_params)
)

# 创建反演问题
problem = InversionProblem(
    forward_fn=your_forward_model,  # 前向模型函数
    y_target=measured_data,          # 观测数据
    bounds=bounds,
    loss_fn=HuberLoss(delta=1.0)
)
```

### 3. CMA-ES 优化

```python
from ventilation_inversion import CMAESOptimizer, CMAESConfig

# 配置优化器
config = CMAESConfig(
    maxiter=500,      # 最大迭代次数
    maxfevals=20000,  # 最大函数评估次数
    verbose=1         # 日志级别
)

# 创建优化器并运行
optimizer = CMAESOptimizer(problem, config)
result = optimizer.run()

# 获取结果
print(f"最优参数: {result.x_best}")
print(f"最终损失: {result.loss_best}")
```

### 4. 多次运行（检测局部最优）

```python
# 执行多次运行
results = optimizer.run_multi(n_runs=10, random_init=True)

# 分析结果
analysis = optimizer.analyze_results(results)
print(f"发现 {analysis['n_unique_solutions']} 个唯一解")
```

## 📊 可视化

运行 `visualization.py` 生成以下图表：

1. **损失曲线** (`loss_curve.png`) - 优化过程收敛情况
2. **参数收敛轨迹** (`parameter_convergence.png`) - 各参数的演化过程
3. **残差分析** (`residual_analysis.png`) - 预测值与观测值对比
4. **多次运行分析** (`multi_run_analysis.png`) - 结果稳定性评估

## 📚 详细文档

请参阅 [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) 获取：

- 完整的数学建模
- 损失函数设计原理
- 系统架构设计
- 工程可解释性分析
- 收敛判据和调参建议

## 📊 实际数据格式 (input.json)

数据文件包含三个主要部分：

### roads（巷道数据）
```json
{
    "id": "e1",           // 巷道ID
    "s": "v2",            // 起点节点
    "t": "v3370",         // 终点节点
    "r0": 3.225,          // 初始风阻（已优化）⭐
    "minR": 1.612,        // 最小风阻边界
    "maxR": 6.450,        // 最大风阻边界
    "initQ": 13.67,       // 初始风量
    "targetQ": 13.67,     // 目标风量（测点）
    "ex": 2.0             // 风阻指数
}
```

### fanHs（风机数据）
```json
{
    "id": "f1",
    "eid": "ne494",       // 对应巷道ID
    "h0": -14.63,         // 初始风机压力
    "minH": -15.0,        // 最小压力
    "maxH": -1.65,        // 最大压力
    "use": "LOCAL"        // "LOCAL" 或 "MAIN"
}
```

### structureHs（结构物阻力）
```json
{
    "id": "s109",
    "eid": "e89",         // 对应巷道ID
    "h": 20.0             // 结构物阻力
}
```

## 🔌 接入 MVN Solver（推荐）

### 方法1：使用 MVNSolverWrapper（推荐）

```bash
# 使用 MVN Solver 运行反演
python real_data_inversion.py --mvn --iter=200
```

```python
from real_data_inversion import (
    DataLoader, MVNSolverWrapper, run_real_data_demo
)

# 加载数据
network = DataLoader.load("input.json")

# 创建 MVN Solver 包装器
forward_model = MVNSolverWrapper(
    network=network,
    json_path="input.json",
    config_ns={
        "loopN": 20,
        "iterN": 30,
        "minQ": 0.05,
        "minH": 0.1
    }
)

# 运行反演
result, config, network = run_real_data_demo(
    json_path="input.json",
    forward_model=forward_model,
    max_iter=200
)
```

### 方法2：手动集成 mvn_solver

参考 `FuncFitness.py` 的方式：

```python
import copy
from jl.vn.ns.ns import mvn_solver
from jl.vn.ns.calcul_weight import calcul_weight

def forward_with_mvn_solver(R: np.ndarray, H: np.ndarray) -> np.ndarray:
    # 深拷贝数据
    roads = copy.deepcopy(base_roads)
    fan_hs = copy.deepcopy(base_fan_hs)
    
    # 将优化变量 R 赋给 road['r']
    for i, road in enumerate(roads):
        road['r'] = float(R[i])
    
    # 将优化变量 H 赋给 fanH['h']
    if fan_hs:
        for i, fan in enumerate(fan_hs):
            fan['h'] = float(H[i])
    
    # 计算风路权重
    init_qs = {road['id']: road['initQ'] for road in roads}
    weights = calcul_weight(initQs=init_qs, weightType="R*Q", roads=roads)
    for road in roads:
        road['weight'] = weights[road['id']]
    
    # 调用网络解算
    result = mvn_solver(
        roads=roads,
        fanHs=fan_hs,
        structureHs=structure_hs,
        configNS=config_ns
    )
    
    # 返回风量
    return np.array([result['roadQs'][r.id] for r in network.roads])
```

### 方法3：通用求解器接口

```python
from real_data_inversion import ExternalSolverWrapper

def my_solver(R_dict, H_dict):
    """
    R_dict: {巷道ID: 风阻值}
    H_dict: {风机ID: 风机压力}
    返回: {巷道ID: 风量}
    """
    # 调用您的风网求解程序
    return your_solver.solve(R_dict, H_dict)

forward_model = ExternalSolverWrapper(network, my_solver)
```

## 📈 性能指标

在演示问题（8维参数）上的典型性能：

| 指标 | 值 |
|-----|-----|
| 平均相对误差 | < 1% |
| 函数评估次数 | ~1500 |
| 运行时间 | < 0.1秒 |
| 收敛迭代次数 | ~150 |

## ⚙️ 依赖

- Python >= 3.8
- numpy >= 1.20.0
- cma >= 3.2.0
- matplotlib >= 3.5.0 (可视化)

## 📄 许可证

MIT License

---

*开发者：高级研究员 | 版本：1.0.0*

