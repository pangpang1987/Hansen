#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	JianGraph类 深度优先遍历搜索----全部点对最长路
#
###########################################################################################################################

import  math

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   全部源汇点对最长路
#
def get_max_path(j, starts=None, targets=None, weight=None) -> list :    # path通路，number复杂度，all全部
    vh,vpath = get_all_max_paths(j,starts=starts, targets=targets, weight=weight)
    maxH, v1, v2, path = -float(math.inf), None, None, None

    for s, hs in vh.items() :               # 源点循环
        for t, h in hs.items() :            # 汇点循环
            if h > maxH :
                maxH,v1,v2,path = h, s, t, vpath[s][t]
    print(maxH,v1,v2,path)


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   全部源汇点对最长路
#
def get_all_max_paths(j, starts=None, targets=None, weight=None) -> list :    # path通路，number复杂度，all全部
    func    =   'ALL_MAX_PATHS'
    vpath = {}
    epath = {}
    vh  =   {}
    eh  =   {} 

    if not weight :
        weight = {}
        for e in j.edges :
            weight[e['id']] = 1
                
    # 节点最大通路
    for vid in j.nodes :
        vpath[vid] = {}
        for sinkv in j.sinkNodes :
            vpath[vid].update(**{sinkv:[]})
        
    # 分支最大通路
    for e in j.edges :
        epath[e['id']] = {}
        for sinkv in j.sinkNodes :
            epath[e['id']].update(**{sinkv:[]})

    # 节点汇点最大阻力
    for vid in j.nodes :
        vh[vid] = {}
        for sinkv in j.sinkNodes :
            vh[vid][sinkv] = -float(math.inf)

    # 分支汇点最大阻力
    for e in j.edges :
        eh[e['id']] = {}
        for sinkv in j.sinkNodes :
            eh[e['id']][sinkv] = -float(math.inf)
        
    # 节点入边着色用于删除节点通路      暂不考虑删除
    vcolor = {}
    for vid,nodeInEdges in j.nodeInEdges.items() :
        vcolor[vid] = []
        vcolor[vid].extend(nodeInEdges)


    if not starts :
        starts = j.sourceNodes
    if not targets :
        targets = j.sinkNodes

    data            =   dict()
    data['vpath']   =   vpath
    data['vh']      =   vh
    data['weight']  =   weight
    data['vcolor']  =   vcolor
    data['me']      =   j.mape
  
    j.depth_first_traversal_search(
        starts=starts,
        targets=targets,
        data    =   data,
        func    =   func
    )

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',data['vh'])
    # print(data['vpath'])
    # data = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    # condition = lambda item: item[1] > 2
 
    # # 使用列表推导式过滤字典
    # filtered_data = {k: v for k, v in data.items() if condition((k, v))}
 
    # print(filtered_data)  # 输出: {'c': 3, 'd': 4}

    vh = {v:h for v,h in data['vh'].items() if h}
    # print("@@@@@@@@@@@-----------",vh)

    vpath = {v:path for v,path in data['vpath'].items() if path}

    return vh, vpath

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   目标节点处置及返回函数
#
def node_target_all_max_paths(vid=None, path=None, data=None, **kwargs) :

    # data['vpath'][vid] = []            # 节点-全部汇点正向最长通路集合


    data['vh'][vid][vid] = 0        # 节点正向通路数
    print("1--------",vid,data['vh'][vid])
    
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",vid)
    # print(data)
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#   无出边
def node_no_out_edge_all_max_paths(vid=None, path=None, data=None, **kwargs) :
    pass

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   分支搜索结束处置函数
def edge_search_traversal_all_max_paths(edge=None, vid=None, t=None, path=None, data=None, **kwargs) :
    # 末节点通路、阻力值
    for v, h in data['vh'][edge['t']].items() :                             # 末节点阻力循环
        if h + data['weight'][edge['id']] > data['vh'][edge['s']][v] :
            data['vh'][edge['s']][v] = h + data['weight'][edge['id']]
            path = [edge['id']] + data['vpath'][edge['t']][v]
            data['vpath'][edge['s']][v] = path
    # 着色,释放内存
    # data['vcolor'][edge['t']].remove(edge['id'])

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   节点搜索成功处置及返回函数      回退删除前一节点通路节省内存
def node_end_traversal_all_max_paths(vid=None, nodeOutEdges=None, data=None, **kwargs) :
    # print(nodeOutEdges)
    # input()
    for e in nodeOutEdges :
        edge = data['me'][e['id']]
        if not len(data['vcolor'][edge['t']]) :
            # data['vpath'][edge['t']] = None    
            data['vpath'][edge['t']].clear()
            data['vh'][edge['t']] = None

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€





