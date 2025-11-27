# 矿井通风网络阻力系数反演技术规范

## 使用 CMA-ES 求解通风阻力系数反演

---

## 1. 问题建模（通风反演数学模型）

### 1.1 阻力系数定义与物理范围

矿井通风中的摩擦阻力系数 $k$ 是描述巷道表面粗糙度的无量纲参数，其物理意义为单位长度、单位周长、单位断面积的风阻贡献。

**阻力系数的典型范围：**

| 巷道类型 | $k$ 值范围 (Ns²/m⁴) |
|---------|---------------------|
| 光滑混凝土巷道 | 0.0010 - 0.0020 |
| 砌碹巷道 | 0.0020 - 0.0035 |
| 锚喷巷道 | 0.0025 - 0.0045 |
| 毛巷道 | 0.0040 - 0.0080 |
| 破碎带巷道 | 0.0060 - 0.0120 |

**风阻计算公式：**

$$R_i = k_i \cdot \frac{L_i \cdot P_i}{A_i^3}$$

其中：
- $R_i$：第 $i$ 条巷道的风阻 (Ns²/m⁸ 或 kg/m⁷)
- $k_i$：第 $i$ 条巷道的阻力系数
- $L_i$：巷道长度 (m)
- $P_i$：巷道周长 (m)
- $A_i$：巷道断面积 (m²)

### 1.2 反演模型公式

**正问题（Forward Problem）：**

给定阻力系数向量 $\mathbf{k} = [k_1, k_2, ..., k_N]^T$，通过求解风网方程得到风量/风压分布。

**风网基本方程：**

1. **质量守恒（节点流量平衡）：**
   $$\sum_{j \in \text{连接节点} i} Q_j = 0$$

2. **能量守恒（回路压力平衡）：**
   $$\sum_{j \in \text{回路} l} h_j = \sum_{j \in \text{回路} l} H_{\text{fan},j}$$

3. **风阻定律：**
   $$h_i = R_i \cdot Q_i \cdot |Q_i|$$

**反问题（Inverse Problem）：**

已知观测值 $\mathbf{y}_{\text{target}} = [y_1, y_2, ..., y_M]^T$，反演阻力系数 $\mathbf{k}$：

$$\mathbf{k}^* = \arg\min_{\mathbf{k}} \mathcal{L}(f(\mathbf{k}), \mathbf{y}_{\text{target}})$$

其中：
- $f(\cdot)$：前向模型（风网求解器）
- $\mathcal{L}(\cdot)$：损失函数

### 1.3 Forward(x) 的定义

```
forward: ℝ^N → ℝ^M

输入: x = [k₁, k₂, ..., k_N]  (N维阻力系数向量)
输出: y = [y₁, y₂, ..., y_M]  (M维观测量向量)

处理过程:
1. 根据 k 计算各巷道风阻 R
2. 求解非线性风网方程组，得到风量分布 Q
3. 根据测点位置提取观测量 y（风量/风速/风压）
```

### 1.4 输入/输出维度

| 变量 | 符号 | 维度 | 说明 |
|-----|------|------|------|
| 阻力系数 | $\mathbf{k}$ | $N$ | 待反演参数，N为巷道数 |
| 观测量 | $\mathbf{y}$ | $M$ | 测点观测值，M为测量点数 |
| 风量分布 | $\mathbf{Q}$ | $N$ | 各巷道风量 |
| 风压降 | $\mathbf{h}$ | $N$ | 各巷道风压降 |

**典型规模：**
- 小型风网：$N = 10-50$，$M = 5-20$
- 中型风网：$N = 50-200$，$M = 20-50$
- 大型风网：$N = 200-1000$，$M = 50-200$

### 1.5 使用 CMA-ES 的理由

**问题特点与 CMA-ES 的匹配性：**

| 问题特点 | CMA-ES 优势 |
|---------|-------------|
| **高维连续参数** | CMA-ES 专为连续优化设计，在 10-100 维表现优异 |
| **非凸黑箱问题** | 不需要梯度信息，利用协方差矩阵自适应搜索方向 |
| **多峰景观** | 通过协方差矩阵捕捉参数相关性，有效探索复杂景观 |
| **噪声/不确定性** | 种群评估提供噪声鲁棒性，支持不确定性下的优化 |
| **边界约束** | 内置边界处理机制 |

**技术要点：**

1. **协方差矩阵自适应**：自动学习目标函数的局部结构
2. **无梯度优化**：适用于不可微或昂贵的前向模型
3. **尺度不变性**：对参数缩放不敏感
4. **全局搜索能力**：通过种群多样性探索解空间

---

## 2. 损失函数设计

### 2.1 MSE（均方误差）

**公式：**
$$\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^{M} w_i (y_{\text{pred},i} - y_{\text{target},i})^2$$

**特点：**
- 对大误差敏感（平方放大）
- 梯度光滑，便于优化
- 假设误差服从高斯分布

**适用场景：**
- 测量设备精度高，数据质量好
- 无明显异常值
- 需要严格拟合所有测点

### 2.2 MAE（平均绝对误差）

**公式：**
$$\mathcal{L}_{\text{MAE}} = \frac{1}{M} \sum_{i=1}^{M} w_i |y_{\text{pred},i} - y_{\text{target},i}|$$

**特点：**
- 对异常值鲁棒
- 梯度在零点不连续
- 对所有误差等权处理

**适用场景：**
- 数据存在少量离群点
- 测量噪声分布偏离高斯
- 关注中位数误差而非均值

### 2.3 Huber Loss（推荐默认）

**公式：**
$$\mathcal{L}_{\text{Huber}} = \frac{1}{M} \sum_{i=1}^{M} w_i \cdot L_\delta(r_i)$$

其中：
$$L_\delta(r) = \begin{cases} 
\frac{1}{2}r^2 & \text{if } |r| \leq \delta \\
\delta(|r| - \frac{1}{2}\delta) & \text{if } |r| > \delta
\end{cases}$$

**参数选择指南：**
- $\delta = 0.1$：高鲁棒性，接近 MAE
- $\delta = 1.0$：平衡选择（推荐）
- $\delta = 10.0$：接近 MSE

**适用场景：**
- **矿井通风测量数据（强烈推荐）**
- 数据可能存在测量误差和少量异常值
- 需要兼顾拟合精度和鲁棒性

### 2.4 噪声风网数据的最佳选择

**推荐：Huber Loss**

理由：
1. 矿井传感器可能存在故障、漂移
2. 测量过程受环境干扰（温度、湿度变化）
3. 部分测点数据可能失准
4. Huber Loss 在小误差时保持 MSE 的平滑性，大误差时具有 MAE 的鲁棒性

### 2.5 多测点误差合并策略

**方法一：简单平均**
$$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} L(y_{\text{pred},i}, y_{\text{target},i})$$

**方法二：加权平均**
$$\mathcal{L} = \frac{\sum_{i=1}^{M} w_i \cdot L(y_{\text{pred},i}, y_{\text{target},i})}{\sum_{i=1}^{M} w_i}$$

权重设计原则：
- 按测量置信度加权：$w_i \propto 1/\sigma_i^2$
- 按测点重要性加权：关键通风路径权重更大
- 按物理量类型归一化：避免量纲差异

**方法三：归一化误差**
$$\mathcal{L} = \frac{1}{M} \sum_{i=1}^{M} L\left(\frac{y_{\text{pred},i}}{s_i}, \frac{y_{\text{target},i}}{s_i}\right)$$

其中 $s_i$ 为归一化因子（可取目标值本身或物理量典型值）。

---

## 3. 反演系统结构设计

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    VentilationInversionSystem                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
│  │ InversionProblem │   │  CMAESOptimizer  │   │   结果分析    │ │
│  ├──────────────────┤   ├──────────────────┤   ├──────────────┤ │
│  │ - forward_fn     │──▶│ - problem        │──▶│ - 误差分析   │ │
│  │ - y_target       │   │ - config         │   │ - 收敛诊断   │ │
│  │ - bounds         │   │ - run()          │   │ - 多解检测   │ │
│  │ - loss_fn        │   │ - run_multi()    │   └──────────────┘ │
│  │ - objective()    │   └──────────────────┘                    │
│  └──────────────────┘            │                              │
│           ▲                      │                              │
│           │                      ▼                              │
│  ┌──────────────────┐   ┌──────────────────┐                   │
│  │   LossFunction   │   │ InversionResult  │                   │
│  ├──────────────────┤   ├──────────────────┤                   │
│  │ - MSELoss        │   │ - x_best         │                   │
│  │ - MAELoss        │   │ - loss_best      │                   │
│  │ - HuberLoss      │   │ - x_history      │                   │
│  └──────────────────┘   │ - loss_history   │                   │
│                         └──────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              可插拔前向模型接口 (forward_fn)              │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │ • VentilationNetwork.forward()  - 完整风网模型           │   │
│  │ • LinearVentilationModel.forward() - 简化线性模型        │   │
│  │ • ExternalSolver.forward()      - 外部求解器接口         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 核心类设计

#### InversionProblem 类

```python
class InversionProblem:
    """
    通风反演问题核心类
    
    职责：
    1. 封装反演问题的所有配置
    2. 提供统一的目标函数接口
    3. 处理边界约束
    4. 评估解的质量
    """
    
    属性:
        forward_fn: Callable   # 前向模型函数
        y_target: np.ndarray   # 观测目标值
        bounds: InversionBounds # 参数边界
        loss_fn: LossFunction  # 损失函数
        dim: int               # 参数维度
        n_measurements: int    # 测量点数
    
    方法:
        objective(x) -> float  # 目标函数（用于优化器）
        evaluate_solution(x)   # 详细评估某个解
        reset_eval_count()     # 重置评估计数
```

#### CMAESOptimizer 类

```python
class CMAESOptimizer:
    """
    CMA-ES 优化器封装
    
    职责：
    1. 配置和运行 CMA-ES
    2. 记录优化历史
    3. 支持多次运行
    4. 结果后处理和分析
    """
    
    属性:
        problem: InversionProblem
        config: CMAESConfig
    
    方法:
        run(x0) -> InversionResult      # 单次运行
        run_multi(n_runs) -> List       # 多次运行
        analyze_results(results) -> Dict # 结果分析
```

### 3.3 可插拔接口设计

**前向模型接口规范：**

```python
def forward_fn(k: np.ndarray) -> np.ndarray:
    """
    前向模型接口
    
    输入:
        k: shape (N,) 阻力系数向量
    
    输出:
        y: shape (M,) 观测量向量
    
    要求:
        - 必须是确定性函数（相同输入产生相同输出）
        - 对于无效输入应抛出异常而非返回 NaN
        - 计算时间应尽量稳定
    """
    pass
```

**外部求解器集成示例：**

```python
class ExternalSolverWrapper:
    """外部风网求解器封装"""
    
    def __init__(self, solver_path: str, config_file: str):
        self.solver_path = solver_path
        self.config = self._load_config(config_file)
    
    def forward(self, k: np.ndarray) -> np.ndarray:
        # 1. 写入参数文件
        self._write_params(k)
        # 2. 调用外部求解器
        subprocess.run([self.solver_path, self.config_file])
        # 3. 读取结果
        return self._read_results()
```

---

## 4. 完整 Python 实现

详见 `ventilation_inversion.py` 文件，包含：

- 所有类定义（LossFunction, InversionProblem, CMAESOptimizer 等）
- 三种损失函数实现（MSE, MAE, Huber）
- CMA-ES 主循环
- 8 巷道演示风网
- 完整的反演流程演示

**运行方式：**

```bash
pip install -r requirements.txt
python ventilation_inversion.py
```

---

## 5. 工程可解释性与反演判据

### 5.1 CMA-ES 如何适应黑箱通风反演？

CMA-ES 是一种自适应进化策略，特别适合通风反演问题：

1. **无需梯度信息**
   - 通风网络的前向模型（如 Hardy-Cross 迭代）难以解析求导
   - CMA-ES 仅需要目标函数值，将前向模型视为黑箱

2. **自动学习搜索方向**
   - 协方差矩阵 $\mathbf{C}$ 编码了参数空间的局部形状
   - 随着迭代进行，自动发现"容易"的搜索方向

3. **处理参数耦合**
   - 通风网络中不同巷道的阻力系数相互影响
   - CMA-ES 通过协方差矩阵自动捕捉这种耦合关系

4. **尺度自适应**
   - 步长 $\sigma$ 自动调整，适应不同阶段的搜索需求
   - 初期大范围探索，后期精细调整

### 5.2 协方差矩阵为何能捕捉"参数相关性"？

**数学原理：**

协方差矩阵 $\mathbf{C}$ 表示采样分布的形状：
$$\mathbf{x} \sim \mathcal{N}(\mathbf{m}, \sigma^2 \mathbf{C})$$

其中：
- 对角元素 $C_{ii}$：第 $i$ 个参数的方差（搜索范围）
- 非对角元素 $C_{ij}$：参数 $i$ 和 $j$ 的协方差（相关性）

**通风反演中的体现：**

1. **串联巷道**：上游阻力变化会影响下游风量，CMA-ES 学习到这种关联，同步调整相关参数

2. **并联巷道**：分流关系导致参数反相关，协方差矩阵捕捉负相关性

3. **主通风路径**：关键巷道的参数敏感度高，协方差矩阵相应元素较大

**自适应更新机制：**

每代更新后，CMA-ES 根据成功的搜索方向更新 $\mathbf{C}$：
$$\mathbf{C}^{(g+1)} = (1-c_1-c_\mu)\mathbf{C}^{(g)} + c_1 \mathbf{p}_c \mathbf{p}_c^T + c_\mu \sum_i w_i \mathbf{d}_i \mathbf{d}_i^T$$

这使得算法能够"记住"有效的搜索方向。

### 5.3 如何判断反演是否成功？

**定量判据：**

| 判据 | 公式 | 阈值建议 | 说明 |
|-----|------|---------|------|
| 相对误差 | $\epsilon_{rel} = \|y_{pred} - y_{target}\| / \|y_{target}\|$ | < 5% | 整体拟合质量 |
| 最大绝对误差 | $\epsilon_{max} = \max_i |y_{pred,i} - y_{target,i}|$ | 取决于量纲 | 最差测点 |
| 参数收敛性 | $\|\mathbf{k}^{(t)} - \mathbf{k}^{(t-1)}\| < \tau$ | $\tau = 10^{-6}$ | 迭代稳定性 |
| 损失下降率 | $(L^{(t-1)} - L^{(t)}) / L^{(t-1)}$ | < 0.1% 持续多代 | 收敛饱和 |

**定性判据：**

1. **物理合理性**：反演得到的 $k$ 值是否在物理范围内
2. **一致性**：多次运行是否收敛到相似解
3. **残差分布**：残差是否呈随机分布（无系统性偏差）
4. **敏感性**：小扰动是否导致结果剧烈变化

**反演失败的警示信号：**

- 损失值始终较大（> 测量噪声水平）
- 参数收敛到边界
- 多次运行结果差异大
- 残差呈现规律性模式

### 5.4 如何检测多个局部最优？

**方法一：多次独立运行**

```python
results = optimizer.run_multi(n_runs=10, random_init=True)
analysis = optimizer.analyze_results(results)

if analysis['n_unique_solutions'] > 1:
    print("警告：检测到多个局部最优！")
```

**方法二：参数空间聚类**

对多次运行的结果进行聚类分析：
1. 使用 K-Means 或 DBSCAN 聚类
2. 计算类内方差和类间距离
3. 不同聚类中心代表不同局部最优

**方法三：损失景观可视化**

对于低维问题（2-3个关键参数），可以：
1. 固定其他参数
2. 在关键参数平面上采样
3. 绘制损失等高线图

**处理策略：**

1. **全局最优选择**：选择损失最小的解
2. **集成方法**：对多个局部最优加权平均
3. **物理约束**：根据先验知识排除不合理的解
4. **增加测点**：更多观测数据通常能减少局部最优数量

### 5.5 如何提升收敛速度？

**策略一：优化初始点**

- 使用工程经验值作为初始猜测
- 基于历史反演结果热启动
- 使用简化模型预求解

**策略二：调整 CMA-ES 参数**

| 参数 | 默认 | 调整建议 |
|-----|------|---------|
| `sigma0` | 边界范围的 1/3 | 减小可加速收敛，但可能陷入局部最优 |
| `popsize` | $4 + 3\ln(N)$ | 增大提高全局搜索能力，但增加计算量 |
| `tolx` | $10^{-11}$ | 放宽可提早停止 |

**策略三：问题重构**

- **参数缩放**：将参数归一化到 [0,1] 区间
- **降维**：对高度相关的参数进行 PCA 降维
- **分层反演**：先反演敏感参数，再反演次要参数

**策略四：并行化**

CMA-ES 的种群评估天然支持并行：

```python
from multiprocessing import Pool

def parallel_objective(solutions):
    with Pool(n_cores) as pool:
        fitness = pool.map(problem.objective, solutions)
    return fitness
```

---

## 6. 可视化建议

### 6.1 损失曲线

**目的**：展示优化过程中损失值的下降趋势

**内容**：
- X 轴：迭代次数或函数评估次数
- Y 轴：损失值（建议对数坐标）
- 可添加：sigma 变化曲线（次Y轴）

**解读**：
- 平滑下降 → 正常收敛
- 阶梯状下降 → 跳出局部最优
- 震荡不降 → 可能需要调整参数

### 6.2 阻力系数收敛轨迹

**目的**：跟踪各参数的演化过程

**内容**：
- X 轴：迭代次数
- Y 轴：参数值
- 每条线代表一个 $k_i$
- 虚线标注真实值（如果已知）

**解读**：
- 参数稳定在某值 → 收敛
- 参数持续振荡 → 可能识别性不足
- 参数收敛到边界 → 约束可能不合理

### 6.3 y_pred vs y_target 残差图

**类型一：散点图**
- X 轴：$y_{target}$
- Y 轴：$y_{pred}$
- 理想情况：点沿 45° 对角线分布

**类型二：残差图**
- X 轴：测点编号或 $y_{target}$
- Y 轴：残差 $y_{pred} - y_{target}$
- 理想情况：残差随机分布在零附近

**类型三：直方图**
- 残差的分布直方图
- 理想情况：接近正态分布，均值接近零

### 6.4 多 Runs 结果分布图

**类型一：箱线图**
- 每个参数一个箱线图
- 展示多次运行的分布

**类型二：平行坐标图**
- 每次运行一条折线
- X 轴：参数索引
- Y 轴：归一化参数值

**类型三：热力图**
- 解的相似度矩阵
- 用于检测聚类结构

---

## 附录：快速开始指南

### 安装

```bash
pip install numpy cma matplotlib
```

### 最小示例

```python
from ventilation_inversion import (
    InversionProblem, InversionBounds, 
    CMAESOptimizer, CMAESConfig, HuberLoss
)
import numpy as np

# 1. 定义前向模型
def my_forward(k):
    return some_solver.solve(k)

# 2. 设置边界
bounds = InversionBounds(
    lower=np.array([0.001] * 5),
    upper=np.array([0.01] * 5)
)

# 3. 创建问题
problem = InversionProblem(
    forward_fn=my_forward,
    y_target=measured_data,
    bounds=bounds,
    loss_fn=HuberLoss(delta=1.0)
)

# 4. 运行优化
optimizer = CMAESOptimizer(problem, CMAESConfig(maxiter=500))
result = optimizer.run()

# 5. 获取结果
print(f"最优参数: {result.x_best}")
print(f"最终损失: {result.loss_best}")
```

---

*文档版本: 1.0.0*
*最后更新: 2024*

