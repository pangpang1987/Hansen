#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   通风网络解算器模块 V1.0
#
######################################################################################

#   1. 重大改进
#   （1）井口考虑标高差及自然风压, 亦即虚拟风路参与迭代
#   （2）风路增加了源汇项

#   2. 主要流程及但愿功能
#   （1）风量初始化（可以外部导入）
#   （2）计算最小树权重（可以外部导入）
#   （3）生成迭代数据（在迭代模块中, 目的是迭代模块将来设API）
#   （4）迭代计算


from    .generateData       import  GenerateData        # 生成迭代数据模块
from    .iterator           import  Iter                # 迭代计算模块

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   网络解算器函数
#
def NetworkSolver(**dataNS) :
    # dataNS 由 Mine 创建
    # print(dataNS["structHs"])
    # 1. 生成迭代数据
    data = GenerateData(dataNS)
    #     roads, 
    #     fanAs       =   fanAs,
    #     fanBs       =   fanBs,
    #     fanHs       =   fanHs,
    #     structRs    =   structRs,
    #     structHs    =   structHs, 
    #     initQs      =   initQs,
    #     fixeds      =   fixeds,
    #     weights     =   weights,
    #     config      =   config
    # )   
    # {"circuits":circuits, "comRoads":comRoadsDict, "roadQs":initQs, "config":config}
    # print(data["config"])

    # print(data["structHs"])
    print(data)
    # 2. 迭代并返回结果
    return Iter(data)
    # ret = {"roadQs" : roadQs, "iterN":iterN*n, "state" : "warning"}
    # 枚举值之间没有依赖关系用state，有依赖关系用status
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


#   3. 数据结构
#   3.1 输入数据
""" menu的
{                                   # 输入数据结构
    "roadways" : [                     # list, 巷道列表, 不可缺省
        {                           # dict, 风路对象
            "id": "e6",             # str, 巷道id, 不可缺省
            "s": "v4",              # id, 始节点id, 不可缺省
            "t": "v5",              # id, 末节点id, 不可缺省
            "r": 0.0827,            # float, 巷道风阻, 不可缺省
            "ex": 2.0,              # float, 阻力指数, 不可缺省
            "initQ": 45.0,          # float, 初始风量, 不可缺省
            "fixed": true,          # bool, 固定风路, [true, false], 可缺省, 与initQ同步
            "weight": 123.12        # float, 最小树权重, 不可缺省
        },
        ......
    ],
    "ventilators": [                      # 风机fanA型列表, list, 可以缺省
        {                           # fanA型风机对象, dict, 不可缺省
            "id": "fan1",           # 风机id, str, 不可缺省
            "eid": "e10",           # 绑定id, id, 不可缺省
            "a0": 1035.92,          # 风机特性曲线0次项, float, 不可缺省
            "a1": 51.73,            # 风机特性曲线1次项, float, 不可缺省
            "a2": -0.43,            # 风机特性曲线2次项, float, 不可缺省
            "direction": "forward", # 风机方向, enum: "forward","reverse", 不可缺省
            "pitotLocation" : "in"  # 静压管位置, enum: "in","out", 不可缺省
        },
        ......
    ],
    "fanBs" [                       # 风机fanB型列表, list, 可以缺省
        {                           # fanB型风机对象, dict, 不可缺省
            "id": "fanB1",          # 风机id, str, 不可缺省
            "eid": "e10",           # 绑定id, id, 不可缺省
            "a0": 1035.92,          # 风机特性曲线0次项, float, 不可缺省
            "a1": 51.73,            # 风机特性曲线1次项, float, 不可缺省
            "a2": -0.43,            # 风机特性曲线2次项, float, 不可缺省
            "b0": 2008.06,          # 风机反抛物线0次项, float, 不可缺省
            "b1": -12.85,           # 风机反抛物线1次项, float, 不可缺省
            "b2": 0.64,             # 风机反抛物线2次项, float, 不可缺省
            "tangentQ": 30.11,      # 正反抛物线切点风量, float, 不可缺省
            "direction": "forward", # 风机动力方向, enum, "forward","reverse", 不可缺省
            "pitotLocation": "in"   # 风机静压测点位置, enum, "in","out", 不可缺省
        },
        ......
    ]
    "fanHs": [                      # 静压型通风动力列表, list, 可缺省
        {                           # fanH型通风动力对象, dict, 不可缺省
            "id": "fanH1",          # 风机id, str, 不可缺省
            "eid": "e1",            # 绑定id, id, 不可缺省
            "h": 100.00             # 风机静压, float, 不可缺省
            "direction": "forward", # 风机动力方向, enum, "forward","reverse", 不可缺省
            "pitotLocation": "in"   # 风机静压测点位置, enum, "in","out", 不可缺省
        },
        ......
    ],
    "structs": [                   # 等效风阻型构筑物列表, list, 可以缺省
        {                           # 等效风阻型构筑物对象, dict, 不可缺省
            "id": "structR1",       # 构筑物id, str, 不可缺省
            "eid": "e4",            # 绑定id, id, 不可缺省
            "r": 0.123,             # 构筑物等效风阻, float, 不可缺省
            "ex": 2.0               # 风量指数, float, 不可缺省
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


"""     menu

{                                   # 输入数据结构, dict
    "roadways" : [                  # 巷道字典对象列表, list, 不可缺省
        {                           # 巷道字典对象, dict
            "id": "e6",             # 巷道id, uuid4, 不可缺省
            "s": "v4",              # 始节点id, uuid4, 不可缺省
            "t": "v5",              # 末节点id, uuid4, 不可缺省
            "r": 0.0827,            # 巷道风阻, float, 不可缺省
            "ex": 2.0,              # 风量指数, float, 可缺省
        },
        ......
    ],
    "ventilators": [                # 风机fanA型列表, list, 可以缺省
        {                           # fanA型风机对象, dict, 不可缺省
            "id": "fan1",           # 风机id, uuid4, 不可缺省
            "bindType": "ROADWAY",  # 绑定类型, ["ROADWAY", ], 缺省默认"ROADWAY"
            "bindId":
            "model": "fanA",        # 风机运转模式, ["fanA","fanB","fanAB","fanH"]
            "a0": 1035.92,          # 风机特性曲线0次项, float, 可缺省
            "a1": 51.73,            # 风机特性曲线1次项, float, 可缺省
            "a2": -0.43,            # 风机特性曲线2次项, float, 可缺省
            "b0": 2008.06,          # 风机反抛物线0次项, float, 可缺省
            "b1": -12.85,           # 风机反抛物线1次项, float, 可缺省
            "b2": 0.64,             # 风机反抛物线2次项, float, 可缺省
            "tangentQ": 30.11,      # 正反抛物线切点风量, float, 可缺省
            "direction": "forward", # 风机动力方向, enum, "forward","reverse", 不可缺省
            "pitotLocation": "in"   # 风机静压测点位置, enum, "in","out", 不可缺省
            "h": -3200,             # 风机静压, float, 可缺省            
            "direction": "forward", # 风机方向, enum: "forward","reverse", 不可缺省
            "pitotLocation" : "in"  # 静压管位置, enum: "in","out", 不可缺省
        },
        ......
    ],

    "structs": [                   # 等效风阻型构筑物列表, list, 可以缺省
        {                           # 等效风阻型构筑物对象, dict, 不可缺省
            "id": "structR1",       # 构筑物id, str, 不可缺省
            "bindId": "e4",            # 绑定id, id, 不可缺省
            "r": 0.123,             # 构筑物等效风阻, float, 不可缺省
            "ex":
            "h"
        },
        ......
    ]
}
"""