# -*- coding:utf-8 -*-
# 指定中文编码

######################################################################################
#
#   风阻系数反演遗传进化适应值模块
#
######################################################################################

#   模块简介
#   （1）适应值函数模块与GA框架无耦合关系, 是一个独立通用模块
#   （2）模块包括种群生成函数及适应值计算函数

from jl.vn.ns.calcul_weight import calcul_weight
from jl.vn.ns.ns import mvn_solver

import numpy as np
import math

import time
import json
import traceback
# from cloud.jl.callJLCloud.callJLCFuncA import CallJLCFuncA

np.seterr(divide="ignore", invalid="ignore")  # 浮点数异常处理函数

from pprint import pprint

# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   创建并行数据
#
#   说明：
#   （1）拷贝外部数据
#   （2）将染色体赋给副本数据

# 将染色体付给网络解算数据
#


# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   多进程并行适应值计算函数
#
def FuncFinness(dataPop):
    """
    :pop 种群个体
    :return误差
    """

    # 1 解析种群个体数据
    # 1.1 解析
    rhs, data = dataPop
    # print(rhs)

    dataNS = data["dataNS"]
    configNS = data["configNS"]
    targetQs = data["targetQs"]
    # 1.2 将染色体风阻、冗余动力赋给road, fanH
    i = 0
    for road in dataNS["roads"]:
        road["r"] = rhs[i]
        i += 1
    if dataNS.get("fanHs", None):
        for fanH in dataNS["fanHs"]:
            fanH["h"] = rhs[i]
            i += 1

    # 2. 计算风路权重
    initQs = {road["id"]: road["initQ"] for road in dataNS["roads"]}
    weights = calcul_weight(initQs=initQs, weightType="R*Q", **dataNS)
    for road in dataNS["roads"]:
        road["weight"] = weights[road["id"]]

    # 3. 网络解算
    try:
        result = mvn_solver(configNS=configNS, **dataNS)  # 迭代返回值
        # result = {'roadQs' : roadQs, "iterN":iterN*n, "state" : 'success'}
        roadQs = result["roadQs"]  # 返回结果的风量值

    # 3 迭代意外中断（程序继续进行）
    except Exception as e:
        traceback.print_exc()
        print(e)
        return 999999  # 意外中断返回预设误差，程序仍继续执行

    # 4 误差（适应值）计算

    q0s = []
    q1s = []
    for id, q in targetQs.items():
        q0s.append(q)
        q1s.append(roadQs[id])
    _fitness_value = math.sqrt(sum((q0 - q1) ** 2 for q0, q1 in zip(q0s, q1s)))
    print("Fitness Value:", _fitness_value)
    return _fitness_value


# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   多线程并行适应值计算函数（调运网络解算云函数）
#
def ParallelThread(pop):
    """
    并行计算函数
    args: 多进程计算数据, 数据类型为列表
    return: 返回并行计算结果, 数据类型为实数
    """
    # dataCloudNC =   pop[0]
    # targetQs    =   pop[1]
    # targetHs    =   pop[2]

    # 1 解析种群个体数据
    # (
    #     dataCloudNC,                    # 网络迭代数据
    #     targetQs,                       # 目标风量, dict
    #     targetHs,                       # 目标压差, dict
    #     errorNormalized,                # 误差归一化标识
    #     targetType,                     # 目标类型, q, qh
    # ) = __ParsingPop(pop)

    dataCloudNC = pop["dataNC"]  # 网络迭代数据
    targetQs = pop["targetQs"]  # 目标风量, dict
    targetHs = pop["targetHs"]  # 目标压差, dict
    errorNormalized = pop["errorNormalized"]  # 误差归一化标识
    targetType = pop["targetType"]  # 目标类型, q, qh

    # t = int(time.time()*100000)                        # 本地使用
    # file = str(t)+".json"
    # with open(file, "w") as f :                 # 本地使用
    #     json.dump(dataCloudNC, f, indent=4)

    # try :
    #     value = __CallJLCFuncA(dataCloudNC, "net-iter")
    #     roadQs = value["roadQs"]
    #     print("roadQs=",roadQs)

    try:
        # result = __NetIterCalculate(dataCloudNC)    # 迭代返回值
        result = __CallJLCFuncA(dataCloudNC, "net-iter")
        # roadQs = result[ROAD_QS]                    # 返回结果的风量值
        roadQs = result["data"]["roadQs"]

    except:
        return np.array(99999)

    # return __GetError(targetQs, roadQs)
    # 4 误差（适应值）计算
    return __GetError(
        targetQs,
        roadQs,
        dataCloudNC,
        errorNormalized,
        targetHs=targetHs,
        targetType=targetType,
    )


# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   串行计算，有错误，使用曹鹏最新
def serial(units):
    """
    串行计算函数！

    args: 多进程计算数据，数据类型为列表

    return: 返回并行计算结果，数据类型为实数

    """

    print(units)
    input()
    # # 数据处理
    # vars = args[0][1]         # 种群矩阵
    # data_copy = args[0][2]    # 复制网络数据，用于解算
    # target = args[0][3]       # 目标数据
    # sheet_name = args[4] # 表名
    # r_name = args[5] # 风阻名
    # Q_name = args[6] # 风量名
    # error = list() # 误差
    # for var in vars:
    #     # 处理网络数据，解算网络
    #     for id,r in enumerate(var):
    #         data_copy[sheet_name][id][r_name] = r # 更改风阻数据
    #         data_copy[sheet_name][id][Q_name] = None # 删除目标值

    #     # 网络解算
    #     Q_Vars = network_solution.main(data_copy)  # 解算新网络

    #     # 目标误差计算

    #     sum_error = 0  # 误差初始化
    #     n = len(target) # 目标值的个数
    #     # print('目标数据个数\n',n)
    #     for edge_id in Q_Vars:  # 循环网络解算数据
    #         # print(Q_Vars)
    #         for edge_id_1 in target: # 循环目标分支
    #             if str(edge_id_1) == edge_id: # 分支对等
    #                 sum_error += (np.abs(Q_Vars[edge_id])
    #                                 - target[edge_id_1]
    #                                 )**2
    #     f_error = (sum_error/n)**0.5  # 添加目标值误差
    #     error.append(f_error)
    # error = np.vstack(error)
    # return np.array(error)


# €€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   拷贝目标数据
def __Copytargets(targetQs, targetHs):
    return {**targetQs}, {**targetHs}


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   拷贝网络迭代数据
#
def __CopyDataIter(dataNC):
    # 1 定义网络迭代数据
    data = dict()

    # 2 拷贝回路
    circuits = [[{**ce} for ce in c] for c in dataNC["circuits"]]
    data["circuits"] = circuits

    # 3 过滤拷贝复合风路
    comRoads = {}
    for eid, comRoad in dataNC["comRoads"].items():  # 复合风路循环体
        comRoad_ = list()  # 复合风路

        for i, road_ in enumerate(comRoad):  # 广义风路列表
            if i == 0:
                comRoad_.append(
                    dict(
                        zip(
                            ["id", "s", "t", "ex", "objName"],
                            [
                                road_["id"],
                                road_["s"],
                                road_["t"],
                                road_["ex"],
                                road_["objName"],
                            ],
                        )
                    )
                )
            else:
                comRoad_.append({**road_})  # 复制广义风路
        comRoads[eid] = comRoad_  # 复合风路字典
    data["comRoads"] = comRoads  # 完成复合风路拷贝
    data["roadQs"] = {**dataNC["roadQs"]}  # 初始风量（亦即迭代风量）
    data[ALGO] = {**dataNC[ALGO]}  # 负值迭代控制参数

    # roadQs = {}
    # algo = {}
    # for eid, q in dataNC[ROAD_QS].items() :
    #     roadQs[eid] = q
    # for key,a in dataNC[ALGO].items() :
    #     algo[key] = a

    # data[ROAD_QS] = roadQs
    # data[ALGO] = algo

    return data  # 返回负值数据


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   解析个体数据
#
def __ParsingPop(pop):
    dataCloudNC = pop["dataNC"]  # 网络迭代数据
    targetQs = pop["targetQs"]  # 目标风量, dict
    targetHs = pop["targetHs"]  # 目标压差, dict
    errorNormalized = pop["errorNormalized"]  # 误差归一化标识
    targetType = pop["targetType"]  # 目标类型, q, qh
    return (
        dataCloudNC,  # 网络迭代数据
        targetQs,  # 目标风量, dict
        targetHs,  # 目标压差, dict
        errorNormalized,  # 误差归一化标识
        targetType,  # 目标类型, q, qh
    )


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   迭代意外中断返回误差函数
#
def __ErrorExcept(errorNormalized=False):
    if not errorNormalized:
        return np.array(99999)  # 非归一化, 返回一个较大误差
    else:
        return np.array(1)  # 归一化, 返回一个最大归一误差


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   均方差计算
#
def __GetError(
    targetQs, roadQs, dataCloudNC, errorNormalized, targetHs=None, targetType="q"
):

    # 1 仅风量误差
    if targetType == "q":
        q0 = list()
        q1 = list()
        for id, q in targetQs.items():
            q0.append(q)
            q1.append(roadQs[id])
        # return __MeanSquareError(q0, q1, errorNormalized)
        ret = __MeanSquareError(q0, q1, errorNormalized)
        print("ret=======================", ret)
        return ret

    # 2 风量风压误差
    else:  # targetType = "qh"
        q0 = list()
        q1 = list()
        for id, q in targetQs.items():
            q0.append(q)
            q1.append(roadQs[id])
        # print("q===============")
        # print(q0)
        # print(q1)
        errorq = __MeanSquareError(q0, q1, errorNormalized)

        comRoads = dataCloudNC[COM_ROADS]
        roadHs = dict()
        eids = targetHs.keys()
        for eid, comRoad in comRoads.items():
            if eid in eids:
                roadHs[eid] = comRoad[0][R] * (roadQs[eid] ** 2)

        h0 = []
        h1 = []
        for id, h in targetHs.items():
            h0.append(h)
            h1.append(roadHs[id])
        # print("hhh!!!!!!============")
        # print(h0)
        # print(h1)
        # print(roadHs)
        errorh = __MeanSquareError(h0, h1, errorNormalized)
        print("----------------", (errorq + errorh) / 2)
        return (errorq + errorh) / 2


# ¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
