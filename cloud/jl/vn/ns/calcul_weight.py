#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   计算最小树权重
#
######################################################################################

#   提示！！！！！：老虎台、红阳三矿测试表明，如果风机、构筑物不是余支，有时会出现迭代震荡
#   所以有必要实时计算权重，并重新计算回路
#   注意网络迭代计算以及进化算法中的不同情况

#   风机最大、构筑物传感器次之、普通风路按风阻大小确定排序
#   fanA    =   maxR    +   humpH
#   fanB    =   maxR    +   humpH
#   fanH    =   maxR    +   h
#   structH =   maxR    +   h

#   注意：没有考虑一条风路多个风机、构筑物

#   如果R*Q模式, road要包含初始风量initQ

#   测试简单串并联


"""
{                                   # 输入数据结构
    "roads" : [                     # list, 巷道列表, 不可缺省
        {                           # dict, 风路对象
            "id": "e6",             # str, 巷道id, 不可缺省
            "r": 0.0827            # float, 巷道风阻, 不可缺省
        },
        ......
    ],
    "fanAs": [                      # 风机fanA型列表, list, 可以缺省
        {                           # fanA型风机对象, dict, 不可缺省
            "id": "fan1",           # 风机id, str, 不可缺省
            "eid": "e10",           # 绑定id, id, 不可缺省
            "a0": 1035.92,          # 风机特性曲线0次项, float, 不可缺省
            "a1": 51.73,            # 风机特性曲线1次项, float, 不可缺省
            "a2": -0.43            # 风机特性曲线2次项, float, 不可缺省
    },
        ......
    ],
    "fanBs" [                       # 风机fanB型列表, list, 可以缺省
        {                           # fanB型风机对象, dict, 不可缺省
            "id": "fanB1",          # 风机id, str, 不可缺省
            "eid": "e10",           # 绑定id, id, 不可缺省
            "a0": 1035.92,          # 风机特性曲线0次项, float, 不可缺省
            "a1": 51.73,            # 风机特性曲线1次项, float, 不可缺省
            "a2": -0.43            # 风机特性曲线2次项, float, 不可缺省
        },
        ......
    ]
    "fanHs": [                      # 静压型通风动力列表, list, 可缺省
        {                           # fanH型通风动力对象, dict, 不可缺省
            "id": "fanH1",          # 风机id, str, 不可缺省
            "eid": "e1",            # 绑定id, id, 不可缺省
            "h": 100.00             # 风机静压, float, 不可缺省
        },
        ......
    ],
    "structRs": [                   # 等效风阻型构筑物列表, list, 可以缺省
        {                           # 等效风阻型构筑物对象, dict, 不可缺省
            "id": "structR1",       # 构筑物id, str, 不可缺省
            "eid": "e4",            # 绑定id, id, 不可缺省
            "r": 0.123             # 构筑物等效风阻, float, 不可缺省
        },
        ......
    ],
    "structHs": [                   # 压差型构筑物列表, list, 可缺省
        {
            "id": "structH1",       # 构筑物id, str, 不可缺省
            "eid": "e3",            # 绑定id, id, 不可缺省
            "h": 100.00             # 压差, float, 不可缺省     
        },
        ......
    ]
}
"""


import  math
INF     =   math.inf
INF_    =   -math.inf

# from    jl.common.dataProcessing.dataProcessing   import  (
#         GetObjsField        as  __GetObjsField
# )

from    .fanA   import  GetHump

from    pprint  import  pprint

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   最小树权重计算函数
#
def calcul_weight(
        roads,                  # list
        fanAs       =   [],
        fanBs       =   [],
        fanHs       =   [],
        structRs    =   [],
        structHs    =   [], 
        initQs      =   {},
        weightType      =   'R*Q',
        *args, 
        **kwargs
) :
    



    # config0 = {
    #     # （3）最小树权重配置参数
    #     # "weightType"    :   "EXT",          # 外部导入
    #     "weightType"    :   "R*Q"          # 考虑构筑物及风机  默认
    #     # "weightType"    :   "R",            # 考虑构筑物及风机
    #     # "weightType"    :   "DEFAULT"          # 默认风路自然列表序列

    # }

    # config = {
    #     **config0,
    #     **config
    # }


    # 1. 自然排序
    if weightType == "DEFAULT" :
        weights = {}
        i = 1
        for road in roads :
            weights[road["id"]] = i
            i += 1
        return weights       # 按roads自然排序
    
    # 2. 外部导入权重
    if weightType == "EXT" :
        # weights = __GetObjsField(roads, "weight", "id")
        weights = {}
        for road in roads :
            weights[road["id"]] = road["weight"]
            del road["weight"]
        return weights
    
    # 3. 计算风路最小树权重

    # 3.1 提取风路风阻系数字典
    # roadRs = __GetObjsField(roads, "r", "id")
    roadRs = {}
    for road in roads :
        roadRs[road['id']] = road['r']
    
    # 3.2 定义复合风路权重，复制风路风阻
    weights = dict(**roadRs)

    # 3.3 添加构筑物风阻
    for structR in structRs :
        weights[structR["eid"]]   +=  structR["r"]      # 巷道风阻 + 构筑物风阻

    # 4.3 计算最大风阻系数
    maxR1 = max(weights.values())
    # print("maxR1=",maxR)

    # 4.4 处置structH型构筑物, 数据来自传感器，所以要大一级
    for structH in structHs :
        weights[structH["eid"]] = maxR1 + abs(structH["h"])

    # 4.5 再次计算最大风阻系数, 保证风机最大
    maxR2 = max(weights.values())
    # print("maxR2=",maxR)


    # 4.6 处置fanA型动力
    for fanA in fanAs :
        a0 = fanA["a0"]
        a1 = fanA["a1"]
        a2 = fanA["a2"]
        if abs(a0) > 0 and not a1 and not a2 :
            h = abs(a0)
        else :
            q, h = GetHump(a0, a1, a2)

        # print(fanA["id"],h)
        weights[fanA["eid"]] = maxR2 + abs(h)

    # 4.7 处置fanB型动力
    for fanB in fanBs :
        a0 = fanB["a0"]
        a1 = fanB["a1"]
        a2 = fanB["a2"]
        if abs(a0) > 0 and not a1 and not a2 :
            h = abs(a0)
        else :
            q, h = GetHump(a0, a1, a2)
        # print(fanB["id"],h)
        weights[fanB["eid"]] = maxR2 + abs(h)

    # print(fanHs)

    # 4.8 处置fanH型动力
    for fanH in fanHs :
        weights[fanH["eid"]] = maxR2 + abs(fanH["h"])
 
    # 5. 计算R*Q权重, road要有initQ
    if weightType == "R*Q" :
        for eid, w in weights.items() :
            weights[eid] = w * initQs[eid]

    return weights
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

