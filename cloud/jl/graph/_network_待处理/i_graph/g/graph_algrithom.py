
import os
import sys
sys.path.append(os.getcwd())

from typing import List, Dict, Union, Any
from .basic import get_no_repeat_values_set_of_dicts_in_list

def all_vertexes(edges_list):
    """
    输出网络节点列表
    
    :param edges_list: a list of dict, include the start and end vertex

    >>> all_vertexes([{'s': 1, 't': 2},{'s': 5, 't': 3},{'s':2, 't': 1}])
    [1, 2, 5, 3]
    >>> all_vertexes([{'s': 1},{'t': 3},{'s':2, 't': 1}])
    [1, 3, 2]
    >>> all_vertexes([{},{},{'s':2, 't': 1}])
    [2, 1]
    >>> all_vertexes([{},{},{}])
    []
    """
    keys = ['s', 't']
    return get_no_repeat_values_set_of_dicts_in_list(edges_list, keys=keys)

def list_to_index_map(list_: List, begin=0) -> dict:
    """
    针对列表建立起列表索引与列表值的映射字典

    :param list_: The list to be converted to a map
    :type list_: List
    :param begin: The index to start at, defaults to 0 (optional)

    >>> list_to_index_map([3,4,1,5])
    {3: 0, 4: 1, 1: 2, 5: 3}
    """
    # 创建映射字典
    return {val: i + begin for i, val in enumerate(list_)}


def edges_map(
    edges_list: List[dict],
    mapped_node: dict,
) -> Dict[Union[str, Any], Union[List[Any], List[Union[float, Any]], List[tuple]]]:
    """
    分支始节点映射

    Args:
        edges_list (List[dict]): 分支列表
        nodes_map (dict): 节点映射字典

    Returns:
        list(tuple): 
            返回映射后的分支列表

    >>> edge_map([{'s': 1, 't': 2, 'id': 1},{'s': 2, 't': 3, 'id':2},{'s':2, "t": 1, "id":3}], {1: 0, 2: 1, 3: 2})
    {
        EIDS: []
        "mapped":[(0,1), (1,2), (1,0)]
        }
    
    """
    # 初始化
    eids = []  # 分支id
    edges_map_list: List[tuple] = []  # 分支映射列表
    for edge in edges_list:
        # 获取属性值
        s, t = mapped_node[edge['s']], mapped_node[edge['t']] # 映射节点
        eid= edge['id']
        # 添加属性
        edges_map_list.append((s, t))
        eids.append(eid)

    return {'eids': eids,
            'mapped': edges_map_list}