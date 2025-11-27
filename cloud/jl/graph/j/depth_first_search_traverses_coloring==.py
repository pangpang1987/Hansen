#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	j_graph
#
###########################################################################################################################

from .process_input_data_type   import  process_input_data_type

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   搜索、避让始末点
def get_start_target_nodes_and_avoid_edges(self, startNodes=None, targetNodes=None, avoidEdges=None) :
    # 1. 搜索起始点 
    if process_input_data_type(startNodes)  is  None    :   startNodes  =   self.sourceVIds
    else                                                :   startNodes  =   process_input_data_type(startNodes)
    # 2. 搜索目标点
    if process_input_data_type(targetNodes) is  None    :   targetNodes =   self.sinkVIds       # 默认全部源点
    else                                                :   targetNodes =   process_input_data_type(targetNodes)
    # 3. 避让分支id
    if process_input_data_type(avoidEdges)  is  None    :   avoidEdges  =   []
    else                                                :   avoidEdges  =   process_input_data_type(avoidEdges)
    return startNodes, targetNodes, avoidEdges

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   深度优先搜索遍历着色
#
def depth_first_search_traverses_coloring(
        self, 
        startNodes  =   None,       # None/str/list, 搜索起始点
        targetNodes =   None,       # None/str/list, 搜索目标点
        avoidEdges  =   None        # None/str/list, 避让分支
) :

    def _func_node_(vid, successEs, targetNodes, avoidEdges, track, snake) :

        v = self.mapv[vid]                        # 当前节点类对象        
        track.extend(['+',vid])              # 记录节点轨迹

        # 2. 当前节点为目标节点目标节点
        if vid in targetNodes :                  # 节点为目标节点
            v.ok       =   True                # 搜索成功
            track.extend(['-',vid])          # 退栈  如果目标点不是汇点，能否出现目标点单向回路？？？
            return

        # 3. 单向回路回退   ？？？？？？   注意连续回退情况，暂未完成，目前仅限于无单向回路
        if vid in snake :           # 蛇头咬蛇身
            track.extend(['-',vid])      # 回退的是刚加入的，蛇身的未加入
            return
        else :
            snake.append(vid)    # 当前节点加入蛇头通路（无此步，前方单向回路不能避免）不能再判断单向回路之前

        # 4. 着色节点回退
        if v.color is True:
            track.extend(['-',vid])
            return

        # 6. 节点无出边回退
        if not len(self.mapv[vid].outs) :
            track.extend(['-',vid])
            v.ok = False
            return

        #7. 节点出边循环体
        for e in self.mapv[vid].outs :           # 节点出边循环
            track.extend(['+',e.id])         # 记录轨迹

            if e.id in avoidEdges :            #当前出边为避让边
                e.success=False  
                continue

            t = e.t                                 # 当前出边末节点

            # 分支深度搜索
            _func_node_(t.id, successEs, targetNodes, avoidEdges, track, snake)
            e.ok = t.ok
            track.extend(['-',e.id])

        # 当前节点是否能到目标点
        for e in v.outs : 
            if e.ok is True : 
                successEs.append(e.id)
                v.ok = True

        v.color = True

        snake.pop()
        track.extend(['-',vid])

    # 节点、分支颜色初始化，否则连续两次搜索出错
    for eobj in self.es : 
        eobj.color = None
        eobj.ok = None

    for vobj in self.vs : 
        vobj.color = None
        vobj.ok = None

    # 函数起始点
    startNodes, targetNodes, avoidEdges = get_start_target_nodes_and_avoid_edges(
        self,
        startNodes=startNodes, 
        targetNodes=targetNodes, 
        avoidEdges=avoidEdges
    )       # 搜索起始点、目标点、避让分支
    successEs = []
    track = []      # 搜索轨迹
    snake = []      # 环形蛇
    for vid in startNodes :
        _func_node_(vid, successEs, targetNodes, avoidEdges, track, snake)

    return successEs, track


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

