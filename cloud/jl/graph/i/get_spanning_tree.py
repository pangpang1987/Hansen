#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph 模块
#
###########################################################################################################################

from .create_igraph     import  create_igraph

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   生成树
#
#   igraph.spanning_tree() 是 igraph 库中的一个函数，用于生成图的生成树。
#   生成树是一个包含原图所有顶点的无环连通子图，通常用于寻找网络的最小连接结构。
def get_spanning_tree(edges, weights=None, return_tree=False, root=None) :
    """
    参数说明
    weights: 可选参数, 指定边的权重列表。如果提供此参数, 将生成最小生成树(MST)。
    return_tree: 布尔值，指定是否返回生成树对象。若为 False, 则返回边的索引列表。
    root: 指定生成树的根节点（整数或顶点 ID)。      不好使，暂不用
    """
    g,mapv,mape = create_igraph(edges)
    mintree_index = g.spanning_tree(weights=weights,return_tree=return_tree)
    mintree = []
    cotree = []
    for e in g.es:
        if e.index in mintree_index:
            mintree.append(e['id'])
        else:
            cotree.append(e['id'])
    return mintree,cotree
     



#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


"""
基本用法
python
运行
from igraph import Graph

# 创建一个示例图
g = Graph.Famous("petersen")

# 计算生成树
spanning_tree = g.spanning_tree()

# 查看生成树的边数（对于n个顶点的树，边数为n-1）
print(f"生成树边数: {spanning_tree.ecount()}")
print(f"原图顶点数: {g.vcount()}")




关键参数
weights:
可选参数，指定边的权重，用于计算最小生成树
可以是边属性名称字符串，也可以是数值列表
mode:
对于有向图，指定生成树的类型
可选值: "in"（入树）、"out"（出树）、"all"（忽略方向）
计算最小生成树示例
python
运行
# 创建带权重的图
g = Graph(edges=[(0,1), (0,2), (1,2), (1,3), (2,3)], directed=False)
g.es["weight"] = [1, 5, 3, 4, 2]  # 设置边权重

# 计算最小生成树（使用Kruskal算法）
min_spanning_tree = g.spanning_tree(weights="weight")

# 输出最小生成树的边及其权重
print("最小生成树的边及其权重:")
for e in min_spanning_tree.es:
    source, target = e.tuple
    weight = g.es.find(_between=(source, target))["weight"]
    print(f"边 ({source}, {target}): 权重 {weight}")
注意事项
对于无向图，默认使用 Kruskal 算法
对于有向图，默认使用 BFS 算法寻找生成树
生成树是原图的一个子图，保留了原图的顶点但只包含部分边
如果原图不连通，则无法生成包含所有顶点的生成树，函数会从最大连通分量中生成树
"""

