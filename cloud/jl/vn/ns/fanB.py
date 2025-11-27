#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   通风网络动力模块    fanB    vnet
#
######################################################################################

#####################################################################################
#
#   通风网络动力模块    vnet    fanA
#
######################################################################################

#   网络中的风机动力是 -
#   计算回路阻力代数和时，直接代数相加，不要按习惯 -

from    math                import  sqrt
from    .fanA   import  FanA

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#	风路类
#
class FanB(FanA) :
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   构造函数
    def __init__(self, 
        id,                     # id
        a0,                     # 正抛物线0次幂系数
        a1,                     # 正抛物线1次幂系数
        a2,                     # 正抛物线2次幂系数
        b0          =   None,   # 反抛物线0次幂系数
        b1          =   None,   # 反抛物线1次幂系数
        b2          =   None,   # 反抛物线2次幂系数
        tangentQ    =   None,   # 正反抛物线切点风量
        qK          =   0.65,   # 反抛物线波谷风量与正抛物线波峰风量之比系数
        hK          =   0.8,    # 反抛物线波谷风压与正抛物线波峰风压之比系数
    ) :
        # 构造父对象
        FanA.__init__(self, id, a0, a1, a2)

        # 计算反抛物线及切点
        if not b0 and not b1 and not b2 :
            self.b0, self.b1, self.b2, self.tangentQ, self.tangentH = \
                GetParameterB(a0, a1, a2, qK, hK)
            # print("tangentQ=",self.tangentQ)

        # 反抛物线参数
        else :
            self.b0         =   b0
            self.b1         =   b1
            self.b2         =   b2
            self.tangentQ   =   tangentQ
        
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   动力计算函数
    def GetH(self, q):
        # return self.sword * (self.a0 + self.a1*q + self.a2*q**2)
        if q >= self.tangentQ :
            return FanA.GetH(self, q)
        else :
            return self.b0 + self.b1*q + self.b2*q**2
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   动力计算函数斜率，修正误差+abs
    def GetSlope(self, q):
        if q>= self.tangentQ :
            return FanA.GetSlope(self, q)
        else :
            return self.b1 + 2*self.b2*q
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   波谷
    def GetTrough(self) :
        # a = self.b2
        # b = self.b1
        # c = self.b0
        # troughQ = - b / (2 * a)
        # troughH = (4* a * c - b**2) / (4 *a)
        troughQ = -self.b1 / (2*self.b2)
        troughH = self.b0 + self.b1*troughQ + self.b2*troughQ**2
        
        return troughQ, troughH
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   横轴交点
    def HorizontalAxisIntersectionB(self, h) :
        a = self.b2
        b = self.b1
        c = self.b0
        # y = 0
        q1 = (-b + pow((b**2 - 4*a*c + 4*a*h),0.5))/(2*a)
        q2 = (-b - pow((b**2 - 4*a*c + 4*a*h),0.5))/(2*a)

        if q1 < q2 :
            return q1, q2       # 返回大风量
        else :
            return q2, q1
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




def get_down_parabol_intervals(x, x_max, x_1, x_2, is_positive):
    """
    验证自变量是否在抛物线的某一取值范围内
    若不在则修改自变量
    抛物线为开口向下（存在非零解）

    Args:
        x (numeric): 自变量
        x_max (numeric): x自变量最大值
        is_positive (bool): 控制因变量是否为正
        x_1 (_type_): 抛物线的解1
        x_2 (_type_): 抛物线的解2, x_2 > x_1

    return:
        返回区间内自变量
    """
    # 返回的满足区间要求的自变量
    x_return = x

    # 因变量为正，抛物线开口向下
    if is_positive is True:
        if x <= x_1 and x>= x_2:
            x_return = (x_1 + x_2)/2

    # 因变量为负，抛物线开口向下
    if is_positive is False:
        if x_1 <= x <= x_2:
            if abs(x - x_1) <= abs(x - x_2):
                x_return = x_1 - 0.1
            if abs(x - x_1) >= abs(x - x_2):
                x_return = x_2 + 0.1

    return x_return

def GetParameterB(a0,a1,a2,qk,hk):
    """
    根据正抛物线求解相切反抛物线
    两条抛物线平滑相切应满足两个个条件: 1) 相交; 2) 交点处一阶导数相同.
    正抛物线方程: H = a0 + a1*Q + a2*Q^2
    h_min, q_min, 二者单位需匹配统一

    Args:
        h_min (float): 反抛物线最低点风压
        q_min (float): 反抛物线最低点风压对应的风量

    Returns:
        1、H = b0 + b1*Q + b2*Q^2
        2、b0, b1, b2, contact_q, contact_h
    """
    # a2不能为0，必须小于0
    if a2 == 0 or a2 > 0:
        raise ValueError(
            'a2 cannot equal to zero and cannot greater than zero')

    # 正抛物线驼峰值
    q_max = -a1 / (2 * a2)
    h_max = a0 + a1 * q_max + a2 * q_max**2

    # 初始反抛物线波谷值
    h_min = h_max * hk
    if q_max <= 0:
        q_min = q_max / qk
    else:
        q_min = q_max * qk

    # 确定h_min
    # 参数1为0时的解，h_min不能为此数
    h0 = (4 * a0 * a2 - a1**2) / (4 * a2)
    if h_min == h0:
        h_min = h0-1

    # 参数1
    parameter1 = a1**2 - 4 * a0 * a2 + 4 * a2 * h_min

    # 以h_min为基准确定q_min
    # q1、q2为parameter2为0的解，
    q1 = (-a1 + sqrt(-4 * a0 * a2 + a1**2 + 4 * a2 * h_min)) / (2 * a2)
    q2 = -(a1 + sqrt(-4 * a0 * a2 + a1**2 + 4 * a2 * h_min)) / (2 * a2)

    # 确保b2 > 0，确定q_min区间
    if parameter1 < 0:
        is_positive = True
    if parameter1 > 0:
        is_positive = False
    q_min = get_down_parabol_intervals(q_min, q_max, q1, q2, is_positive)

    # 参数2
    parameter2 = a2 * q_min**2 + a1 * q_min + a0 - h_min

    # 求解参数
    b0 = h_min - parameter1 * q_min**2 / (4 * parameter2)  # 求解b0
    b1 = parameter1 * q_min / (2 * parameter2)  # 求解b1
    b2 = -parameter1 / (4 * parameter2)  # 求解b2

    # 找切点
    # 切点风量值
    contact_q = (b1 - a1) / (2 * a2 - 2 * b2)
    # 切点风压值
    contact_h = a0 + a1 * contact_q + a2 * pow(contact_q, 2)
    
    return b0, b1, b2, contact_q, contact_h