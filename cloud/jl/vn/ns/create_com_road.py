#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   复合风路函数模块
#
######################################################################################

from    .road       import  Road
from    .fanSword            import  FanSword
from    .fanA       import  FanA
from    .fanB       import  FanB
from    .fanH       import  FanH
from    .structR    import  StructR
from    .structH    import  StructH

import numpy    as  np

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#	风路类
#
class ComRoad(Road) :
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    # def __init__(self, id, s, t, r=None, ex=2.0,sourceSinkQ=0, powers=[], structs=[]) :
    def __init__(self, powers=[], structs=[], **kwargs) :
        # Road.__init__(self, id, s, t, r=r, ex=ex)
        Road.__init__(self, **kwargs)
        # self.sourceSinkQ = sourceSinkQ
        self.powers     =   list()              # 一条风路安多个动力
        self.structs    =   list()              # 一条风路安多个构筑物
        self.powers.extend(powers)
        self.structs.extend(structs)

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   摩擦阻力计算函数
    #   动力不含摩擦阻力
    def GetResis(self, q) :
        resis = Road.GetResis(self, q)
        if self.structs :
            for struct in self.structs :
                resis   +=  struct.GetResis(q)
        return resis     
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   始末端点压差计算函数
    def GetH(self, q) :
        # 1. 巷道压差
        h = Road.GetH(self, q)                  # 巷道阻力

        # 2. 巷道绑定动力压差
        for power in self.powers :              # 绑定动力循环
            # h -= power.GetH(q)                # 注意 -=, sword 已有-
            h += power.GetH(q)

        # 3. 巷道绑定构筑物压差
        for struct in self.structs :
            h += struct.GetH(q)
            
        return h
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   始末端点压差斜率（注意无-，动力有）
    def GetSlope(self, q) :
        slope = abs(Road.GetSlope(self, q))
        # slope = Road.GetSlope(self, q)            # 不收敛，数据见：nc-20221216
        if self.powers :
            for power in self.powers :
                slope += abs(power.GetSlope(q))     # 待测试迭代速度
                # slope += power.GetSlope(q)        # 不收敛，数据见：nc-20221216
        if self.structs :
            for struct in self.structs :
                slope += abs(struct.GetSlope(q))
                # slope += struct.GetSlope(q)       # 不收敛，数据见：nc-20221216
        return slope
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   生成复合风路列表函数
#
def create_com_road(
    roads, 
    fanAs=None,
    fanBs=None,
    fanHs=None,
    structureRs=None,
    structureHs=None
):

    # 1. 定义复合风路列表及字典
    comRoadsList = list()
    comRoadsDict = dict()

    # 2. 创建无动力、无构筑物复合风路
    for road in roads :                     # 风路数据表循环体
        # comRoad = ComRoad(road["id"], road["s"], road["t"], road["r"], road["ex"])
        # print("---------------------",road)
        comRoad = ComRoad(**road)       # road缺省可以，不能多
        comRoadsList.append(comRoad)
        comRoadsDict[road["id"]] = comRoad

    if isinstance(fanAs, list):
    # if fanA is not None:
        for fanA in fanAs :     # fanA列表循环
            # 创建FanA对象
            fanObjA = FanA(fanA["id"], fanA["a0"], fanA["a1"], fanA["a2"])

            # 创建FanSword对象
            powerA = FanSword(fanObjA, direction=fanA["direction"], pitotLocation=fanA["pitotLocation"])
        

            # 复合风路添加动力对象
        comRoadsDict[fanA["eid"]].powers.append(powerA)

    # 4. 添加fanB型动力

    if isinstance(fanBs, list):
        for fanB in fanBs :
            fanObjB = FanB(
                fanB["id"],
                fanB["a0"], fanB["a1"], fanB["a2"],
                fanB["b0"], fanB["b1"], fanB["b2"], fanB["tangentQ"]
            )
            powerB = FanSword(fanObjB,
                direction=fanB["direction"], pitotLocation=fanB["pitotLocation"])
            comRoadsDict[fanB["eid"]].powers.append(powerB)

    # 5. 添加fanH型动力
    if isinstance(fanHs, list):            
        for fanH in fanHs :
            # print("fanH======",fanH)
            fanObjH = FanH(fanH["id"], fanH["h"])   # 定义类对象
            # print(fanObjH.id,fanObjH.h)
            
            powerH = FanSword(fanObjH)
            # direction=fanH["direction"], pitotLocation=fanH["pitotLocation"])
            comRoadsDict[fanH["eid"]].powers.append(powerH)

    if isinstance(structureRs, list):
        for structureR in structureRs :
            structureObjR = StructR(structureR["id"], eid=structureR["eid"],r=structureR["r"], ex=structureR["ex"])
            comRoadsDict[structureR["eid"]].structs.append(structureObjR)

    if isinstance(structureHs, list):
        for structureH in structureHs :
            structureObjH = StructH(structureH["id"],structureH["eid"], structureH["h"])
            comRoadsDict[structureH["eid"]].structs.append(structureObjH)
            # print(structH)
            
    return comRoadsList, comRoadsDict
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€





