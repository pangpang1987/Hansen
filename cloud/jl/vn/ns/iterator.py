#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   迭代计算器
#
######################################################################################

#   1 输入数据（详见 data.json)
#   （1）迭代回路列表（不含虚拟风路、固定风路）
#   （2）复合风路字典
#   （3）初始风量字典
#   （4）迭代次数及精度控制参数

#   2 输出数据（详见 result.json）
#   （1）风路风量字典
#   （2）成功标识，满足精度成功True，否则不成功False

#   3 输入数据结构说明
"""
{
    "circuits": [               # 回路列表, 共3个独立回路
        [                       # 回路1, 构成回路的风路及方向列表
            {                   # 风路1
                "d": 1,         # 正向
                "eid": "e2"     # 风路id
            },
            {                   # 风路2
                "d": -1,        # 负向
                "eid": "e9"     # 风路id
            },
            ......
        ],
        ......
    ],
    "comRoads": {                       # 复合风路字典
        "e1": [                         # 复合风路id, 无绑定动力及构筑物
            {                           # 风路e1对象
                "id": "e1",             # 风路id
                "s": "v1",              # 始节点id
                "t": "v2",              # 末节点id
                "r": 0.054,             # 风阻系数
                "ex": 2.0,              # 风量指数
                "objName": "road"       # 对象名称
            }
        ],
        "e2": [                         # 复合风路id, 绑定机站风机
            {                           # 风路e2对象
                "id": "e2",             # 风路id
                "s": "v2",              # 始节点id
                "t": "v3",              # 末节点id
                "r": 0.0457,            # 风阻系数
                "ex": 2.0,              # 风量指数
                "objName": "road"       # 对象名称
            },
            {                           # 风机对象
                "id": "fanA1",          # 风机id
                "eid": "e2",            # 绑定风路id
                "a0": 1035.92,          # 风机特性曲线a0
                "a1": 51.73,            # 风机特性曲线a1
                "a2": -0.43,            # 风机特性曲线a2
                "sword": -1,            # 网络方向（机密）
                "qD": 1,                # 风机叶轮风流方向（机密）
                "objName": "fanA"       # 风机对象名称
            }
        ],
        ......
        "e4": [                         # 复合风路id, 绑定压差型构筑物
            {
                "id": "e4",
                "s": "v3",
                "t": "v7",
                "r": 0.0337,
                "ex": 2.0,
                "objName": "road"
            },
            {                           # 构筑物对象
                "id": "structH1",       # 构筑物id
                "eid": "e4",            # 绑定巷道id
                "h": 200.0,             # 构筑物压差
                "objName": "structH"    # 对象名称
            }
        ],
        ......
        "e7": [                         # 复合风路id
            {
                "id": "e7",
                "s": "v5",
                "t": "v6",
                "r": 0.1637,
                "ex": 2.0,
                "objName": "road"
            },
            {                           # 绑定风机对象
                "id": "fanB2",          # 风机id
                "eid": "e7",            # 绑定巷道id
                "a0": 850.0,            # 风机特性曲线a0
                "a1": 25.0,             # 风机特性曲线a1
                "a2": -0.2,             # 风机特性曲线a2
                "b0": 1225.2,           # 反抛物线b0
                "b1": 3.36,             # 反抛物线b1
                "b2": 0.112,            # 反抛物线b2
                "tangentQ": 34.68,      # 正反抛物线相切点
                "sword": -1,            # 网络方向（机密）
                "qD": 1,                # 叶轮风流方向（机密）
                "objName": "fanB"       # 对象名称
            }
        ],
        "e8": [                         # 复合风路id
            {
                "id": "e8",
                "s": "v4",
                "t": "v6",
                "r": 0.49,
                "ex": 2.0,
                "objName": "road"
            },
            {                           # 绑定动力对象（空气幕、机站）
                "id": "fanH1",          # 动力id
                "eid": "e8",            # 绑定巷道id
                "h": -1600.0,           # 相对静压
                "sword": 1,             # 网络方向（机密）
                "qD": 1,                # 叶轮风流方向（机密）
                "objName": "fanH"       # 对象名称
            },
            {                           # 绑定构筑物对象
                "id": "structR1",       # 构筑物id
                "r": 0.0123,            # 等效风阻
                "eid": "e8",            # 绑定id
                "ex": 1.5,              # 风量指数
                "objName": "structR"    # 对象名称
            }
        ......
        "e10": [                        # 复合风路id
            {                           # 风路对象
                "id": "e10",
                "s": "v7",
                "t": "v8",
                "r": 0.082,
                "ex": 2.0,
                "objName": "road"
            },
            {                           # 风机对象
                "id": "fanB1",
                "eid": "e10",
                "a0": 1035.92,
                "a1": 51.73,
                "a2": -0.43,
                "b0": 2008.06,
                "b1": -12.85,
                "b2": 0.64,
                "tangentQ": 30.11,
                "sword": -1,
                "qD": 1,
                "objName": "fanB"
            }
        ]
    },
    "roadQs": {             # 风路初始风量字典
        "e1": 46.5,         # e1风路初始风量
        "e2": 1.0,
        "e3": 45.5,
        "e4": 0.5,
        "e5": 0.5,
        "e6": 45.0,
        "e7": 45.5,
        "e8": 0.5,
        "e9": 46.0,
        "e10": 46.5
    },
    "algorithm": {          # 迭代控制参量
        "loopN": 10,        # 外循环次数
        "iterN": 30,        # 内循环次数，总迭代数 <= (loopN-n)*interN
        "minQ": 0.05,       # 风量最小修正值
        "minH": 0.1         # 阻力最小修正值
    }
}
"""

#   输出数据结构
"""
{
    "roadQs": {                         # 风路风量字典
        "e1": 100.31922266267257,       # key: 风路id, value:风路风量
        "e2": 96.24221180230083,
        ......
        "e10": 100.31922266267257
    },
    "iterN": 300                        # 迭代了次数
    "state": true                       # 解算成功标识, True迭代精度满足, 否则False
}
"""
import      math
INF_    =   -math.inf
ZERO    =   1e-6

from pprint import  pprint
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   模块2: 网络解算 2.3 迭代运算函数
#
def iter(
    circuits        =   [],
    comRoadsDict    =   {},
    roadQs          =   {},
    loopN           =   30,        # 迭代计算倍增数
    iterN           =   30,        # 迭代计算次数 , loopN*iterN 
    minQ            =   0.01,         # 风量修正最小值
    minH            =   0.1,         # 回路闭合差最小值    
    *args, 
    **kwargs
) :

    n = 0
    while loopN > 0 :
        loopN -= 1
        n += 1
        maxH, maxQ = CircuitsIter(circuits, comRoadsDict, roadQs, iterN=iterN, minH=minH, minQ=minQ)
        # print("maxQ,maxH========================",maxQ,maxH)
        if maxQ < minQ or maxH < minH :    # 回路闭合差、风量修正值小于目标值, 双目标易造成震荡
            # pprint(roadQs)
            # print(iterN*n)
            # input()
            return {'roadQs' : roadQs, "iterN":iterN*n, "state" : 'success'}
            # 返回节点风量平衡、回路风压平衡
    # pprint(roadQs)
    # print(iterN*n)
    # input()
    return {"roadQs" : roadQs, "iterN":iterN*n, "state" : "warning"}
#
#   测试表明 :
#   （1）权重影响收敛
#   （2）收敛目标值过小可能震荡
#   （3）回路压差及风量修正值双目标可能造成震荡，宜采用 or 单目标
#   （4）fanH, structH 之 slope 返回 0 不影响收敛
#   （5）初值不合理，导致fanA, fanB 不收敛
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   回路循环迭代
#
def CircuitsIter(circuits, comRoadsDict, roadQs, iterN=30, minH=10,minQ=0.1) :
    maxQ = 0                             # 回路迭代最大风量修正量
    maxH = 0                             # 回路迭代最大回路闭合差
    for i in range(iterN) :                 # 一轮迭代次数, 默认30次
        for circuit in circuits :           # 回路循环
            sigmaH,deltaQ = CorrectionCircuitQ(circuit,comRoadsDict,roadQs,minH=minH,minQ=minQ)
            if maxH < sigmaH    :   maxH = sigmaH
            if maxQ < deltaQ    :   maxQ = deltaQ
    # print(maxQ,maxH)
    return maxH, maxQ
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   修正回路风量----注意：动力是负阻力已经在动力模块解决，在此直接相加即可
#
#   测试数据：testData-20221205.xls
#
def CorrectionCircuitQ(circuit,comRoadsDict,roadQs,minH=10,minQ=0.1) :
    # 1. 回路压差代数和、压差斜率总和
    sigmaH      = 0                             # 压差代数和
    sigmaSlope  = 0                             # 压差斜率总和
    # print(circuit)
    for ce in circuit :                         # 风路循环
        eid = ce['eid']                           # 风路id
        d   = ce["d"]                             # 风路方向

        comRoad = comRoadsDict[eid]             # 复合风路
        q = roadQs[eid]                         # 风路风量
        sigmaH      +=  d * comRoad.GetH(q)     # 复合风路压差
        sigmaSlope  +=  comRoad.GetSlope(q)     # 复合风路压差斜率

    # 2. 计算回路风量修正值
    deltaQ = -0.5 * sigmaH / abs(sigmaSlope)

    # 3. 修正回路风量
    if  abs(sigmaH) >= minH or abs(deltaQ) >= minQ :    # 注意！双目标震荡
        for ce in circuit :                     # 对回路风量进行修正
            eid         =   ce["eid"]           # 风路id
            d           =   ce["d"]             # 风路方向
            roadQs[eid] +=  d * deltaQ          # 修正风路风量----d
        return abs(sigmaH), abs(deltaQ)         # 返回回路压力闭合差，风量修正值
    else :
        return 0, 0
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€