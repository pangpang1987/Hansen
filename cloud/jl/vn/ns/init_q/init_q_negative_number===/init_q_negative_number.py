from .flow_adjustment_for_sapse_edges import adjustment
from .flow_distribution_for_sapse_edges import  distribution

# 入口函数
def init_q_negative_number(edges, minQ=-500, adjust=False, n=6):
    """
    edges: list, 拓扑数据
    fixedQs: dict, 固定风量数据
    minQ, 巷道最低风量限制
    adjust: 是否平差
    n: (int), 节点风量平衡误差控制, 相当于 10^-n
    """



    # 复制edges
    edges_top = edges[:]
    # 开始分风
    status, initQs = distribution(edges_top, minQ=minQ, n=n)
    # print(status)
    # input()
    Qs = {}
    for edge in initQs:
        Qs[edge["id"]]=edge["Q"]
    return Qs       # 负风量初始化正常，但status=False， 原因待查


    # if status:     # 初始化成功
    #     Qs = {}
    #     for edge in initQs:
    #         Qs[edge["id"]]=edge["Q"]
    #     return Qs
    # if not status and not adjust : # 初始化失败并且不需平差
    #     return {}
    # else:  # 初始化失败，平差并再次初始化
    #     initQs = adjustment(initQs)   # 平差并初始化
    #     return initQs




