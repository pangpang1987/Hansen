#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#   矿井通风网络解算器模块 Ver. 2025
#
###########################################################################################################################

# “网络解算” 常见的英文翻译为 network solution 或 network calculation，具体使用需结合上下文场景：
# network solution：更侧重 “通过网络运算得出最终结果” 的过程，尤其在需要求解复杂问题（如数学模型、工程优化、数据拟合等）的场景中常用，
# 例如 “卫星定位网络解算” 可译为 “satellite positioning network solution”。
# network calculation：更偏向 “对网络相关数据进行计算” 的动作本身，强调数值运算环节，比如 “网络流量数据解算” 
# 可表述为 “network traffic data calculation”。
# 在工程、测绘、计算机网络等领域，“network solution” 是更普遍的专业表述，若涉及单纯的数值计算场景，“network calculation” 也可适用。

#   1. 重大改进
#   （1）井口考虑标高差及自然风压, 亦即虚拟风路参与迭代
#   （2）风路增加了源汇项
#   （3）初始风量、最小权重由外部导入，解耦需要，放在菜单模块

#   2. 主要流程及单元功能
#   （1）生成迭代数据（在迭代模块中, 目的是迭代模块将来设API）
#   （2）迭代计算


#
#   测试表明 :
#   （1）权重影响收敛
#   （2）收敛目标值过小可能震荡
#   （3）回路压差及风量修正值双目标可能造成震荡，宜采用 or 单目标
#   （4）fanH, structH 之 slope 返回 0 不影响收敛
#   （5）初值不合理，导致fanA, fanB 不收敛
#   （6）fanA不合理可能不收敛, 可以强制改变风机工况点，解决一部分，在驼峰右侧不进入h<0区间
#           但是不改，第二个免费版2

#   1. 前置模块
#   （1）风量初始化, 风量初始化不合理也会导致fanA反向时震荡不收敛
#   （2）复合风路权重, 权重不合理容易导致震荡不收敛
#   （3）数据检查
#   （4）网络检查

#   2. 模块构成与流程
#   （1）创建风路、动力（同时设置sword, qd）、构筑物类对象
#   （2）确定迭代回路
#   （3）迭代计算

#   3. 数据结构
#   3.1 输入数据
"""
{                                   # 输入数据结构
    "roads" : [                     # list, 巷道列表, 不可缺省
        {                           # dict, 风路对象
            "id": "e6",             # str, 巷道id, 不可缺省
            "s": "v4",              # id, 始节点id, 不可缺省
            "t": "v5",              # id, 末节点id, 不可缺省
            "r": 0.0827,            # float, 巷道风阻, 不可缺省
            "ex": 2.0,              # float, 阻力指数, 不可缺省
            "initQ": 45.0,          # float, 初始风量, 可缺省. 缺省时初始化在外部进行
            "fixedQ": 100.0,        # float, 固定风量, 可缺省. 有动力及initQ场景下可缺省
            "weight": 123.12        # float, 最小树权重, 可缺省. 外部导入模式下缺省
        },
        ......
    ],
    "fanAs": [                      # 风机fanA型列表, list, 可以缺省
        {                           # fanA型风机对象, dict, 不可缺省
            "id": "fan1",           # 风机id, str, 不可缺省
            "eid": "e10",           # 绑定id, id, 不可缺省
            "a0": 1035.92,          # 风机特性曲线0次项, float, 不可缺省
            "a1": 51.73,            # 风机特性曲线1次项, float, 不可缺省
            "a2": -0.43,            # 风机特性曲线2次项, float, 不可缺省
            "direction": "forward", # 风机动力方向, enum: "FORWARD","REVERSE", 不可缺省
                                    # 前端有两个方向，安装方向，叶轮转动方向
            "pitotLocation" : "in"  # 静压管位置, enum: "IN","OUT", 不可缺省
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
            "direction": "FORWARD", # 风机动力方向, enum, "forward","reverse", 不可缺省
            "pitotLocation": "in"   # 风机静压测点位置, enum, "in","out", 不可缺省
        },
        ......
    ],
    "structureRs": [                   # 等效风阻型构筑物列表, list, 可以缺省
        {                           # 等效风阻型构筑物对象, dict, 不可缺省
            "id": "structR1",       # 构筑物id, str, 不可缺省
            "eid": "e4",            # 绑定id, id, 不可缺省
            "r": 0.123,             # 构筑物等效风阻, float, 不可缺省
            "ex": 2.0               # 风量指数, float, 不可缺省
        },
        ......
    ],
    "structureHs": [                   # 压差型构筑物列表, list, 可缺省
        {
            "id": "structH1",       # 构筑物id, str, 不可缺省
            "eid": "e3",            # 绑定id, id, 不可缺省
            "h": 100.00             # 压差, float, 不可缺省     
        },
        ......
    ]
}
"""

#   1.2 网络解算控制参量
"""
{                                   # 控制参量对象
    "loopN"         :   20,         # 迭代计算倍增数
    "iterN"         :   30,         # 迭代计算次数    
    "minQ"          :   0.05,       # 风量修正最小值
    "minH"          :   0.1,        # 回路闭合差最小值
    # "qInitType"      :   "initQ",    # 初始化类型:
    # "minInitQ"      :   0.5,        # 初始化风路最小风量
    # "weightType"    :   "r*q",      # 最小树权重: enum, 
    # "checkNet"      :   True        # 网络检查标识
}
"""

from    .get_circuits       import  get_circuits
from    .create_com_road    import  create_com_road    # 创建复合风路类

# from    .generate_iterdata      import  generate_iterdata        # 生成迭代数据模块
from    .iterator           import  iter                # 迭代计算模块
# from    jl.error.my_error           import  MyAppValueError
# from    jl.error.error_code         import  ERROR_CODE_ITER, ERROR_MESS_ITER

from    contextlib                  import  contextmanager

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   矿井通风网络解算器  2025
#

from pprint import  pprint

def mvn_solver(
    roads,
    fanAs       =   None,
    fanBs       =   None,
    fanHs       =   None,
    structureRs =   None,
    structureHs =   None,
    configNS    =   {}
) -> dict :

    # # 数据检查  不需要，在前面已经处理
    # if roads is None:
    #     raise MyAppValueError(message='data error', code=1, data=None)

    # 1. 找回路
    circuits = get_circuits(roads, filterVirtual=True)
    # print("circuits")
    # pprint(circuits)
    # input()
    # 2. 创建复合风路
    comRoadsList, comRoadsDict = create_com_road(
        roads, 
        fanAs=fanAs,
        fanBs=fanBs,
        fanHs=fanHs,
        structureRs=structureRs,
        structureHs=structureHs
    )
    # for eid, com in comRoadsDict.items():
    #     print(eid,com.powers)
    # input()
    # 3. 生成迭代数据
    initQs = {e['id']: e['initQ'] for e in roads}
    iterdata = {
        "circuits"  :   circuits,
        "comRoadsDict"  :   comRoadsDict,
        "roadQs"    :   initQs
    }
    # pprint(iterdata)
    # input()
    # 4. 装载config
    iterdata.update(configNS)       # config装入迭代数据

    # 5. 迭代并返回结果（设置迭代异常中断）
    try :
        return iter(**iterdata) # ={"roadQs" : roadQs, "iterN":iterN*n, "state" : "warning"}
    except Exception as e:
        # 迭代计算异常，返回错误状态
        message = f'Network solution iteration error: {e}'
        return {'state': 'error', 'code': -1, 'message': message, 'data': None, 'roadQs': {}}
    # 枚举值之间没有依赖关系用state，有依赖关系用status
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   create_data_ns上下文
#
@contextmanager
def context_net_solve(*args, **kwargs):
    resultNS = mvn_solver(*args, **kwargs)
    yield resultNS
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建通风网络解算器数据装饰器
#
def decorator_net_solve(func):                                  # 创建网络解算器数据装饰器函数
    def wrapper(*args, **kwargs):                               # 包装函数
        with context_net_solve(*args, **kwargs) as resultNS :   # dict, 包括导入数据及菜单数据
            return func(resultNS=resultNS, *args, **kwargs)     # 返回被装饰函数
    return wrapper                                              # 返回包装函数
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€




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
