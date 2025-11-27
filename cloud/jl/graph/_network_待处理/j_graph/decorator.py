#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	JianGraph类 深度优先遍历搜索    装饰器模块
#
###########################################################################################################################


from    .func_name      import  (
        func_node_target,
        func_node_loop,
        func_node_color,
        func_node_no_out_edge,
        func_node_end,
        func_edge_giveway,
        func_edge_search
        # _func_
)
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

#   节点深度优先搜索返回值函数
# def decorator_node_result(func):
#     def wrapper(funcname, *args, **kwargs):  # 不定长参数*args,**kwargs
#         return func_node_result[funcname](*args, **kwargs)
#     return wrapper
