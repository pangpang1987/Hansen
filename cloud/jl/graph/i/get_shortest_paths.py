#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph模块 最短路
#
###########################################################################################################################

#   若有权重 → 使用 Dijkstra 算法（非负权重）或 Bellman-Ford 算法（支持负权重）

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   
#
def get_shortest_paths(
    g,
    source,             # 源顶点索引
    to=None,            # 目标顶点（默认为所有顶点）
    mode="all",         # 遍历方向："out", "in", "all"
    weights=None,       # 边的权重列表
    output="epath",     # 输出格式："vpath", "epath", "both"
    algorithm="auto"    # 算法选择（通常自动）
) -> list:
    paths = g.get_shortest_paths(
        source,             # 源顶点索引
        to          =   to,            # 目标顶点（默认为所有顶点）
        mode        =   mode,         # 遍历方向："out", "in", "all"
        weights     =   weights,       # 边的权重列表
        output      =   output,     # 输出格式："vpath", "epath", "both"
        algorithm   =   algorithm    # 算法选择（通常自动）
    )
    return paths

"""
核心参数详解

source (必需)
类型：整数
说明：计算路径的起始顶点索引（从 0 开始）。
to (目标顶点)
类型：整数、列表或 None
默认：None（计算到图中所有顶点的路径）
示例：
to=4 → 计算到顶点 4 的路径
to=[2, 5] → 计算到顶点 2 和 5 的路径
mode (遍历方向)
选项：
"out"：从源顶点出发的路径（有向图的默认设置）
"in"：指向源顶点的路径
"all"：忽略方向（无向图的默认设置）
示例：
python
# 在有向图中查找指向 source 的路径
get_shortest_paths(source, mode="in")
weights (边权重)
类型：列表或 None
默认：None（所有边权重为 1，即无权图）
规则：
权重值越大表示路径越长。
若需表示实际距离，可用正数；负数权重可能导致未定义行为。
示例：
python
weights = [0.5, 1.0, 2.0]  # 为每条边赋予权重
output (输出格式)
选项：
"vpath"：返回顶点路径（默认）
"epath"：返回边路径
"both"：同时返回顶点和边路径
不可达路径返回 None。
返回值解析

返回值取决于 output 参数：

output="vpath"
返回列表，每个元素是从 source 到目标顶点的 顶点索引列表。
示例输出：[[0, 1, 3], [0, 2], None]
（表示到顶点 3 的路径为 0→1→3，到顶点 2 的路径为 0→2，某个目标不可达）
output="epath"
返回列表，每个元素是路径上的 边索引列表。
示例输出：[[0, 2], [1], None]
（边索引对应图的 get_edgelist() 顺序）
output="both"
返回元组 (vpath, epath)，包含顶点路径和边路径。
算法选择

自动模式 (algorithm="auto")：
若 weights=None → 使用 BFS（广度优先搜索）（高效，复杂度 O(|V| + |E|)）。
若有权重 → 使用 Dijkstra 算法（非负权重）或 Bellman-Ford 算法（支持负权重）。
手动覆盖：可通过 algorithm 参数指定算法（如 "dijkstra"）。
使用示例

1. 无权图（BFS）

python
from igraph import Graph

# 创建图：0 → 1 → 2 ← 3
g = Graph(directed=True)
g.add_vertices(4)
g.add_edges([(0,1), (1,2), (3,2)])

# 计算顶点 0 到所有顶点的最短路径
paths = g.get_shortest_paths(0)
print(paths)  # 输出: [[0], [0,1], [0,1,2], []] 
              # 顶点 3 不可达 → 空列表
2. 有权图（Dijkstra）

python
# 添加权重：边 (0,1)=1.0, (1,2)=2.0, (0,2)=4.0
weights = [1.0, 2.0, 4.0]

# 计算到顶点 2 的最短路径（权重最小）
path_to_2 = g.get_shortest_paths(0, to=2, weights=weights)
print(path_to_2)  # 输出: [[0,1,2]] （权重和=3 < 4）
3. 获取边路径

python
# 获取路径的边索引
epath = g.get_shortest_paths(0, to=2, weights=weights, output="epath")
print(epath)  # 输出: [[0, 1]] （对应边 (0,1) 和 (1,2)）
4. 多目标路径

python
# 计算到顶点 1 和 2 的路径
paths = g.get_shortest_paths(0, to=[1, 2], weights=weights)
print(paths)  # 输出: [[0,1], [0,1,2]]
常见问题与技巧

不可达顶点：
目标不可达时返回空列表（[]）或 None（取决于版本）。使用前需检查：
python
if not paths[target]: 
    print("Unreachable!")
性能优化：
对大型图，指定 to 参数避免全图计算。
非负权重时，Dijkstra 比 Bellman-Ford 更高效。
负权重处理：
若存在负权重环，使用 algorithm="bellman-ford"。
注意：负权重可能导致无界最短路径（需检查图中无负环）。
路径重建：
通过 epath 可结合 g.es[edge_id]["attr"] 获取路径属性（如距离、类型）。
"""
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


