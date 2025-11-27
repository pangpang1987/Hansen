#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	JianGraph类 深度优先遍历搜索----遍历着色模块
#
###########################################################################################################################

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   目标节点处置及返回函数
#
def node_target_all_paths(vid=None, path=None, data=None, **kwargs) :
    if data['mode'] == 'all' :
        data['vpaths'][vid] = []
        data['vpathNumber'][vid] = 1
    # elif data['mode'] == 'path' :
    #     data['vpaths'][vid] = []
    elif data['mode'] == 'number' :
        data['vpathNumber'][vid] = 1
    
    # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",vid)
    # print(data)
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#   无出边
def node_no_out_edge_all_paths(vid=None, path=None, data=None, **kwargs) :
    if data['mode'] == 'all' :
        data['vpaths'][vid] = None
        data['vpathNumber'][vid] = 0
    # elif data['mode'] == 'path' :
    #     data['vpaths'][vid] = None
    elif data['mode'] == 'number' :
        data['vpathNumber'] = 0

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   分支搜索处置函数
def edge_search_traversal_all_paths(edge=None, vid=None, t=None, path=None, data=None, **kwargs) :
    # vid是edge始节点
    if data['mode'] == 'all' :
        data['epathNumber'][edge['id']] = data['vpathNumber'][t]
        if data['vpathNumber'][t]==1 and not len(data['vpaths'][t]) :       # 汇支
            data['epaths'][edge['id']] = [[edge['id']]]
        elif data['vpathNumber'][t]==0 :                                    # 独头汇支
            data['epathNumber'][edge['id']] = 0
            data['epaths'][edge['id']]  = []
        else :
            paths = []
            for path in data['vpaths'][t] :
                path_ = [edge['id']] + path
                paths.append(path_)
            data['epaths'][edge['id']] = paths
            print(edge['id'],vid,t,paths)
    elif data['mode'] == 'number' :
        data['epathNumber'][edge['id']] = data['vpathNumber'][t]
    # print('分支=================',edge['id'],data)
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   节点搜索成功处置及返回函数      回退删除前一节点通路节省内存
def node_end_traversal_all_paths(vid=None, nodeOutEdges=None, data=None, **kwargs) :
    if data['mode'] == 'all' :
        paths = list()
        for eid in nodeOutEdges :
            # print("eid================",eid,data['epathNumber'][eid])
            if data['epathNumber'][eid] > 0 :
                for path in data['epaths'][eid] :
                    paths.append(path)
                    #   删除前一节点
        data['vpaths'][vid] = paths
        data['vpathNumber'][vid] = len(paths)
        # print('节点结束------------------------',vid, paths)
    elif data['mode'] == 'number' :
        sum = 0
        for eid in nodeOutEdges :
            sum += data['epathNumber'][eid]
        data['vpathNumber'][vid] = sum

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


