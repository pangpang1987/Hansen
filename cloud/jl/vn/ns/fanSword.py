#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   通风网络动力模块
#
######################################################################################

#   通风动力如果反向安装，则风量是负值 qd=-1


#   热动力体现在回路上，单独加在某条风路上不合理，所以，2.3 删除热动力，可通过fanH进行一些特殊处理
#   解决一些特殊问题

#   （1）风流方向与风机叶轮动力方向一致为正, 反之为负
#   （2）通风网络的风流方向是风路始末节点方向, 一致为正, 反之为负
#   （3）通风网络的风流方向在传入之前要改成以风机动力方向为基准

#   传感器即可以安装在风机入口，也可以安装在风机出口
#   传感器位于风机入口或出口，对风机特性曲线计算结果没有，但是对传感器有影响
#   传感器是多少就是多少，但要绑定传感器位置


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#	通风网络动力类
#
class FanSword :
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数

    def __init__(self, power, direction="FORWARD", pitotLocation="IN") :
        # print('------------direction',direction)
        # input()
        self.power      =   power           # 通风动力 fanA, fanB, fanH 对象
        # self.eid        =   eid             # 绑定id
        self.sword, self.qd = _SetSword(
            power.__class__.__name__, 
            direction, 
            pitotLocation
        )
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   始末端点压差计算函数
    def GetH(self, q) :
        q   =   self.qd * q                 # 流经动轮风流方向，与动轮一致为正
        # print('------------q',q)
        return  self.sword * self.power.GetH(q)     # 网络动力（抽出式为负值）
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   始末端点压差斜率（注意无-，动力有）
    def GetSlope(self, q) :
        q   =   self.qd * q
        return  self.sword * self.power.GetSlope(q)     # *-1 .  ???
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
#
#   设置动力方向        风机反转要将direction设置为REVERSE
#
def _SetSword(className, direction, pitotLocation) :
    """
    direction - 风机安装方向, 与风机运转方向不同。正常运转动力方向与巷道始末节点方向一致为正，否则为负
    pitotLocation - 皮托管安装位置, 也是传感器感压位置
    """

    # # 1. 默认
    # if not direction or not pitotLocation :
    #     return 1, 1

    # 2. 通风机安装方向，正常运转时叶轮动力方向
    if direction == "FORWARD" :         # 叶轮与风路一致
        d1 = 1
    if direction == "REVERSE" :         # 叶轮与风路反向
        d1 = -1
    
    # 传感器位置（只影响fanH型）
    # if pitotLocation == "IN" :      # 风机入风口
    #     d2 = 1
    # if pitotLocation == "OUT" :     # 风机出风口
    #     d2 = -1

    # # 特性曲线
    # if className == "fanH" :        # 读数为正，已经前面变负
    #     d3 = 1
    # if className == "FanA" or className == "FanB" :           # fanA, fanB
    #     d3 = -1


    # 3. 风压计算：特性曲线、传感器
    # 3.1 传感器

    if className == "FanH" :        # fanH  实测
        d2 = 1                      # 特性曲线始终为 1
        if pitotLocation == "IN" :      # 风机入风口
            d3 = 1
        if pitotLocation == "OUT" :     # 风机出风口
            d3 = -1

    # 3.2 特性曲线
    if className == "FanA" or className == "FanB" :           # fanA, fanB
        d2 = -1         # 特性曲线始终为负
        d3 = 1          # 皮托管始终为正

    # 4. 计算sword
    sword = d1 * d2 * d3

    # 5. 确定叶轮风流方向
    if direction == "FORWARD" :
        qd = 1
    if direction == "REVERSE" :
        qd = -1

    # 6. 返回结果
    return sword, qd
#¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥





# #¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥
# #
# #   设置动力方向
# #
# def _SetSword(className, direction, pitotLocation) :
#     # # 1. 默认
#     # if not direction or not pitotLocation :
#     #     return 1, 1

#     # 2. 叶轮方向
#     if direction == "FORWARD" :         # 叶轮与风路一致
#         d1 = 1
#     if direction == "REVERSE" :         # 叶轮与风路反向
#         d1 = -1
    
#     # 3. 风压计算：特性曲线、传感器
#     # 3.1 传感器
#     if className == "FanH" :        # fanH  实测
#         d2 = 1
#         if pitotLocation == "IN" :      # 风机入风口
#             d3 = 1
#         if pitotLocation == "OUT" :     # 风机出风口
#             d3 = -1
#     # 3.2 特性曲线
#     if className == "FanA" or className == "FanB" :           # fanA, fanB
#         d2 = -1
#         d3 = 1 

#     # 4. 计算sword
#     sword = d1 * d2 * d3

#     # 5. 确定叶轮风流方向
#     if direction == "FORWARD" :
#         qd = 1
#     if direction == "REVERSE" :
#         qd = -1

#     # 6. 返回结果
#     return sword, qd
# #¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥¥


