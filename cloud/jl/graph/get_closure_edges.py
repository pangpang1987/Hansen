#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	图类模块
#
###########################################################################################################################



def get_closure_related_edges(g, closure_edges, starts=None, targets=None):
    """
    获取与不能通行边相关联的所有边，考虑路径约束
    
    参数:
    g: igraph 图对象
    bad_edges: 不能通行的边索引列表
    starts: 起点顶点名称（可以是 None、单个字符串或字符串列表）
    targets: 终点顶点名称（可以是 None、单个字符串或字符串列表）
    
    返回:
    相关边的索引列表（排序）
    """
    # 1. 处理起点参数
    if starts is None:
        # 默认：所有入度为0的顶点（源点）
        start_vertices = [v.index for v in g.vs if v.indegree() == 0]
    else:
        # 统一转换为列表
        if isinstance(starts, str):
            starts = [starts]
        
        # 将顶点名称转换为索引
        start_vertices = []
        for name in starts:
            # 查找具有指定名称的顶点
            vertex = g.vs.find(name=name)
            start_vertices.append(vertex.index)
    
    # 2. 处理终点参数
    if targets is None:
        # 默认：所有出度为0的顶点（汇点）
        target_vertices = [v.index for v in g.vs if v.outdegree() == 0]
    else:
        # 统一转换为列表
        if isinstance(targets, str):
            targets = [targets]
        
        # 将顶点名称转换为索引
        target_vertices = []
        for name in targets:
            # 查找具有指定名称的顶点
            vertex = g.vs.find(name=name)
            target_vertices.append(vertex.index)
    
    # 如果没有起点或终点，返回空集
    if not start_vertices or not target_vertices:
        return []
    
    # 3. 计算在路径上的顶点集合
    # 3.1 正向遍历：从起点出发可达的顶点
    forward_reachable = set()
    for start in start_vertices:
        # 使用BFS获取从start可达的所有顶点
        queue = [start]
        visited = set(queue)
        while queue:
            current = queue.pop(0)
            forward_reachable.add(current)
            # 获取当前顶点的出边邻居
            neighbors = g.successors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    # 3.2 反向遍历：可到达终点的顶点
    backward_reachable = set()
    for target in target_vertices:
        # 使用BFS获取可到达target的所有顶点
        queue = [target]
        visited = set(queue)
        while queue:
            current = queue.pop(0)
            backward_reachable.add(current)
            # 获取当前顶点的入边邻居
            neighbors = g.predecessors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    
    # 3.3 求交集：在路径上的顶点
    path_vertices = forward_reachable & backward_reachable
    
    # 4. 获取坏边的顶点集合
    bad_vertices = set()
    for eid in closure_edges:
        edge = g.es[eid]
        bad_vertices.add(edge.source)
        bad_vertices.add(edge.target)
    
    # 5. 获取路径上与坏边相关的顶点
    relevant_vertices = path_vertices & bad_vertices
    
    # 6. 收集相关顶点的所有邻接边
    related_edges = set()
    for v in relevant_vertices:
        # 获取顶点的所有邻接边（入边+出边）
        related_edges.update(g.incident(v, mode="all"))
    
    return sorted(related_edges)

