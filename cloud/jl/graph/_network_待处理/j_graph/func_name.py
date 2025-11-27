#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	JianGraph类 深度优先遍历搜索    函数名称索引模块
#
###########################################################################################################################

#   1.      函数列表
#   1.1     节点类函数
#   1.1.1   node_target             目标节点            
#   1.1.2   node_loop               单向回路节点         
#   1.1.3   node_color              着色节点            
#   1.1.4   node_no_out_edge        无出边节点           
#   1.1.5   node_end                节点结束回退         
#   1.2     分支类函数
#   1.2.1   edge_giveway            分支避让             
#   1.2.2   edge_search             分支深度搜索          


def _func_(*args, **kwargs) -> None : return


#   2.      函数映射
#   2.1     节点类

#   2.1.1 目标节点
from    .sink_subnet         import  node_target_sink_sub_g
from    .traversal_color    import  node_target_traversal_color
from    .path.all_paths              import  node_target_all_paths           # 全部通路
from    .max_path          import  node_target_all_max_paths
func_node_target = {                # 目标节点
    'SINK_NODE_SUB_G'       :       node_target_sink_sub_g,              # 汇点子网
    'TRAVERSAL_COLOR'       :       node_target_traversal_color,        
    'ALL_PATHS'             :       node_target_all_paths,               # 全部通路
    'ALL_MAX_PATHS'         :       node_target_all_max_paths        # 全部点对最大阻力路线
}

#   2.1.2 单向回路节点
func_node_loop ={                   # 单向回路节点
    'SINK_NODE_SUB_G'       :       _func_,                          # 汇点子网
    'TRAVERSAL_COLOR'       :       _func_,
    'ALL_PATHS'             :       _func_,                              # 全部通路
    'ALL_MAX_PATHS'         :       _func_
}

#   2.1.3   着色节点
func_node_color ={
    'SINK_NODE_SUB_G'       :       _func_,                          # 汇点子网
    'TRAVERSAL_COLOR'       :       _func_,
    'ALL_PATHS'             :       _func_,
    'ALL_MAX_PATHS'         :       _func_
}

#   2.1.4     无出边节点
func_node_no_out_edge ={
    'SINK_NODE_SUB_G'       :       _func_,                          # 汇点子网
    'TRAVERSAL_COLOR'       :       _func_,
    'ALL_PATHS'             :       _func_,
    'ALL_MAX_PATHS'         :       _func_
}

#   2.1.5     节点搜索结束
from    .sink_subnet    import  node_end_sink_sub_g
from    .traversal_color    import  node_end_traversal_color
from    .path.all_paths              import  node_end_traversal_all_paths
from    .max_path          import  node_end_traversal_all_max_paths
func_node_end = {
    'SINK_NODE_SUB_G'  :     node_end_sink_sub_g,                               # 汇点子网
    'TRAVERSAL_COLOR'   :   node_end_traversal_color,
    'ALL_PATHS'         :   node_end_traversal_all_paths,
    'ALL_MAX_PATHS'     :   node_end_traversal_all_max_paths
}

#   2.2 分支

#   2.2.1   避让
func_edge_giveway ={
    'SINK_NODE_SUB_G'       :       _func_,                          # 汇点子网
    'TRAVERSAL_COLOR'       :       _func_,
    'ALL_PATHS'             :       _func_,
    'ALL_MAX_PATHS'         :       _func_
}


#   2.2.2  搜索成功
from    .sink_subnet import  edge_search_sink_sub_g
from    .traversal_color    import  edge_search_traversal_color
from    .path.all_paths          import  edge_search_traversal_all_paths
from    .max_path      import  edge_search_traversal_all_max_paths
func_edge_search = {
    'SINK_NODE_SUB_G'  :     edge_search_sink_sub_g,
    'TRAVERSAL_COLOR'   :   edge_search_traversal_color,
    'ALL_PATHS'         :   edge_search_traversal_all_paths,
    'ALL_MAX_PATHS'     :   edge_search_traversal_all_max_paths
}


#   3.  装饰器

#   3.1 目标节点
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   目标节点处置及返回装饰函数
def decorator_node_target(func):
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        return func_node_target[funcname](*args, **kwargs)
    return wrapper

        # 1.1.2 单向回路节点    node_loop
def decorator_node_loop(func):
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        return func_node_loop[funcname](*args, **kwargs)
    return wrapper

        # 1.1.3 着色节点        node_color
def decorator_node_color(func):
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        return func_node_color[funcname](*args, **kwargs)
    return wrapper
  
        # 1.1.4 无出边节点      node_no_out_edge
def decorator_node_no_out_edge(func):
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        return func_node_no_out_edge[funcname](*args, **kwargs)
    return wrapper

        # 1.1.5 节点结束回退    node_end
def decorator_node_end(func):
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        return func_node_end[funcname](*args, **kwargs)
    return wrapper


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#   节点深度优先搜索返回值处置装饰函数
def decorator_edge_giveway(func) :
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        # return func_edge_result[funcname](*args, **kwargs)
        func_edge_giveway[funcname](*args, **kwargs)
    return wrapper

def decorator_edge_search(func):
    def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
        # return func_edge_result[funcname](*args, **kwargs)
        func_edge_search[funcname](*args, **kwargs)
    return wrapper
