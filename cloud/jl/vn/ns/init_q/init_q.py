#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#   通风网络解算风量初始化模块
#
###########################################################################################################################

#   1. 2015年10月版本的变化主要是解耦合
#   （1）将风量初始化以及权重计算从网络解算函数中剥离出来，这样风量初始化也就没有外部导入模式了
#   （2）将测试数据、传感器数据统归为测试数据，以后考虑实时数据
#   （3）将设计项目与孪生项目分离

#   2. 在菜单函数中确定初始化模式，菜单是根据场景设计的
#   （1）按需分风: 守恒模式+无风机
#   （2）自然分风: 平差模式+fanA+fanB, 由于有风机，风机初始风量可能不平衡，所以平差
#   （3）混合模式: 实时网络解算: 守衡模式+排序模式+平差模式, 用户选择     由于传感器+特性曲线驱动，所以可以有多种模式，任选
#   （4）仿真: 任意模式模式，用户通过冻结数据等设置
#   （5）反风: 负风量，任意模式，负风量正风量混合或特性曲线 ？？？
#   （6）如果仅有fanH，没有固定风量，两种办法初始化，一是设定最小风量值，二是调用风机工况风量等有效数值

#   3. 仿真
#   （1）仿真：各种驱动数据均可使用，谈不上固定风量，只是驱动数据，负风量、曲线、正风量
#   （2）自然风压仅有fanH，无固定风量可以初始化, 设置风路最小风量
#   （3）如果没有固定风量、没有fanAs、fanBs，仅有fanHs, 则使用无固定最小巷道风量模式，最小风量设置为0
#   （4）仿真究竟采用何种模式，取决于用户

#   4. 单一用风点，应当采用混合模式，注意掘进头、回采面调风智能控制，仅给出掘进头、工作面临时需风量

#   5. 注意
#   （1）一条风路安装多台风机
#   （2）为防止特性曲线错误收敛，风机初值风量要合理

from    .flow_distribution_for_sapse_edges  import  Dinic
from    .flow_adjustment_for_sapse_edges    import  adjustment
# from    jl.math.fan                         import  find_vertex         # 正抛物线驼峰点
# from    jl.math.fan                         import  find_intersections  # 正抛物线与水平轴交点交点
from    jl.error.my_error                   import  MyAppValueError
import  operator

import  math

'''
        road = {
            'id'    :   str,          # 风路id
            's'     :   str,          # 始点id
            't'     :   str,          # 末点id
            'fixedQ':   float,        # 固定风量, 缺省None
            'initQ' :   float,        # 初始化风量, 缺省None
            'fanA'  :   float,        # fanA曲线
            'fanB'  :   float,        # fanB曲线
            'fanH'  :   float,        # fanH曲线
        }
'''

from    pprint  import  pprint

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#	风量初始化类
#
class InitQ:
    
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   初始化函数
    def __init__(
        self,   
        roads,                 # list
        fanAs       =   None,                 # list
        fanBs       =   None,                 # 无需考虑fanHs
        fanHs       =   None,
        qInitType   =   None,               # 初始化模式
        minInitQ    =   0.1,                # 最小初始风量(注意负风量, 如果存在负风量，取最大负风量)
        n           =   6,                  # 风量误差控制, 相当于 10^-n
        **kwargs
    ) :

        self.roads          =   roads                           # 风路列表
        if fanAs is None    :   self.fanAs  =   []
        else                :   self.fanAs  =   fanAs                           # A型风机列表    
        if fanBs is None    :   self.fanBs  =   []
        else                :   self.fanBs  =   fanBs                           # B型风机列表
        if fanHs is None    :   self.fanHs  =   []
        else                :   self.fanHs  =   fanHs                           # H型风机列表
        self.qInitType      =   qInitType                       # 初始化模式, 外部导入模式、守恒模式、排序模式、平差模式
        self.minInitQ       =   minInitQ                        # 最小初始风量(注意负风量, 如果存在负风量，取最大负风量)
        self.n              =   n                               # 风量误差控制, 相当于 10^-n
        self.edges          =   [{**road} for road in roads]    # 复制独立的edges
        self.fixedQs        =   {**{e['id']:e['fixedQ'] for e in self.edges if e.get('fixedQ', None)}}
                                # 提取fixedQs的目的在于有时需要排序删除
        self.sumH           =   len(self.fanHs)                         # H型风机数量
        self.sumAB          =   len(self.fanAs) + len(self.fanBs)       # A,B型风机数量
        self.sumABH         =   self.sumH + self.sumAB                  # A,B,H型风机数量
        self.sumFixedQ      =   len(self.fixedQs)                       # 固定风量数量
        self.sortedFanQs    =   self.__calculate_sort_AB_fan_initq__()     # list({eid:q},...) 由大到小
        self.negativeNumber =   len([id for id, q in self.fixedQs.items() if q < 0])
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   风量初始化函数
    #   要点: 函数内部结构次序不能变
    def initialization(self) :
        # print('sumABC=========',self.sumABH)
        # 1. 无驱动数据（无固定风量、fanA, fanB, fanH）异常中断
        if all([self.sumFixedQ==0, self.sumABH==0]) :     # 有固定风量或动力
            message = 'No airflow rate initialization drive data!'
            raise MyAppValueError(message=message, code=0, data=None)

        # 2. 纯fanH驱动（无固定风量，无fanA,无fanB,通过设置最小风量 low_l=0.5, 一般不会有错误, 无需异常中断）
        elif all([self.sumFixedQ==0, self.sumAB==0, self.sumH>0]):
            self.__reset_fixed_q__(self.fixedQs)    # 重设固定风量
            flag, initQs = Dinic(edges_dict=self.edges, low_l=0.5, n=self.n)
            return initQs, self.fixedQs

        # 3. 正、负风量混合模式（大概率出现回路分母=0, 误差大不能进行误差判别）
        elif self.negativeNumber > 0 :          # 负风量风路数>0
            self.__reset_fixed_q__(self.fixedQs)    # 重置固定风量、补充漏风源汇字段
            minQ    =   min(self.fixedQs.values())  # 计算最小负风量
            flag, initQs = Dinic(edges_dict=self.edges, low_l=minQ, n=self.n)
            return initQs, self.fixedQs

        # 4. 无论有无固定风量, 先按守恒初始化（此时驱动数据为最小风量）, 如果成功, 则转入添加风机
        else :
            # print('======================',self.fixedQs)
            # input()

            self.__reset_fixed_q__(self.fixedQs)
            flag, initQs = Dinic(edges_dict=self.edges,low_l=self.minInitQ,n=self.n)
            if flag :   # 固定风量初始化成功
                success_initQs = initQs
                if self.sumAB > 0 :                                             # 有AB型风机
                    success_initQs = self.__add_fan__(success_initQs)           # 添加风机, 注意固定不变
                    # 将来如果固定不合理，在初始阶段就进行测试数据及传感器数据平差
                return success_initQs, self.fixedQs
            else :      # 固定风量初始化失败
                if self.qInitType == 'CONSERVATION' :                           # 如果守恒模式
                    message = 'Airflow rate initialization error!'              # 错误提示
                    raise MyAppValueError(message=message, code=0, data=None)   # 异常中断
                elif self.qInitType == 'SORT' :                                 # 如果排序模式
                    self.fixedQs = self.__func_sort__()                         # 固定风量排序
                    self.__reset_fixed_q__(self.fixedQs)                        # 重置固定风量、补充漏风源汇字段
                    flag, initQs = Dinic(edges_dict=self.edges,low_l=self.minInitQ,n=self.n)
                    success_initQs = initQs
                    if self.sumAB > 0 :                                         # 有AB型风机
                        success_initQs = self.__add_fan__(success_initQs)       # 添加风机
                    return success_initQs, self.fixedQs
                elif  self.qInitType == 'ADJUSTMENT' :                          # 如果平差模式
                    # 此法属于全风网平差，将来淘汰，改成通路法，
                    # 固定风量加必须连接风路构成网络，然后平差
                    self.__reset_fixed_q__(self.fixedQs)                        # 重置固定风量、补充漏风源汇字段
                    flag, initQs = Dinic(edges_dict=self.edges,low_l=self.minInitQ,n=self.n)
                    for e in self.edges:
                        e['Q'] = initQs[e['id']]
                    initQs = adjustment(self.edges)
                    for id,q in self.fixedQs.items():
                        self.fixedQs[id] = initQs[id]
                        # return initQs, self.fixedQs
                    success_initQs = initQs
                    if self.sumAB > 0 :                                             # 有AB型风机
                        success_initQs = self.__add_fan__(success_initQs)           # 添加风机
                    return success_initQs, self.fixedQs
                else :
                    message = 'Airflow rate initialization error!'
                    raise MyAppValueError(message=message, code=0, data=None)
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   添加风机初始风量
    def __reset_fixed_q__(self,fixedQs) :
        for e in self.edges :
            e['fixedQ'] = fixedQs.get(e['id'], None)    # 不能使用self.fixedQs, 因为添加风机导致fixedQs变化
            e['sourceSinkQ'] = None

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   添加风机初始风量
    def __add_fan__(self,success_initQs) :
        fixedQs = {eid:q for eid,q in self.fixedQs.items()}
        # self.fixedQs                                      # self.fixedQs在此模块不变
        for f_q in self.sortedFanQs :                       # 循环添加风机
            fixedQs.update({f_q['eid']:f_q['q']})           # 添加风机
            self.__reset_fixed_q__(fixedQs)                 # 重设固定风量
            flag, initQs = Dinic(                           # 添加风机后重新初始化    
                edges_dict=self.edges,
                low_l=self.minInitQ,
                n=self.n
            )
            if not flag :                                   # 添加风机不成功   
                last_key, last_value = fixedQs.popitem()    # 弹出添加的风机
            else :                                          # 添加风机成功
                success_initQs = initQs                     # 保存添加风机后初始化风量
        return success_initQs                               # 返回添加风机后初始化风量
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    #   计算AB型风机初始风量并排序
    def __calculate_sort_AB_fan_initq__(self) :
        def get_fan_init_q(a0, a1, a2) :                                    # 计算个体风机初始风量内置函数
            hk = 0.85                                                       # 0.85倍驼峰高度右侧风量
            qk = 1.35                                                       # 1.35倍驼峰风量
            humpQ, humpH = find_vertex(a0, a1, a2)                          # 计算驼峰
            q = find_intersections(a0, a1, a2, y0=humpH*hk)[0]              # 0.75*humpH高度处右侧值
            return int((humpQ*qk + q)/2)                                    # 取两种算法的平均风量, 小数位数太多不利于初始化

        fanQs = []
        fans = self.fanAs + self.fanBs                                      # 合并曲线类型
        # print(fans)
        for fan in fans :                                                   # A、B型风机循环     
            q = get_fan_init_q(fan['a0'], fan['a1'], fan['a2'])             # 风机初始风量
            fanQs.append({'eid': fan['eid'], 'q': q})                       # 转成对应巷道初始风量
        sorted_roadQs = sorted(fanQs, key=lambda x: x['q'], reverse=True)   # 由大到小排序
        return sorted_roadQs
    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

    #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
    
    # def __func_sort__(edges, fixedQs={}, minInitQ=0.1, n=6, fans=[]) :
    def __func_sort__(self) :
        
        newFixedQs = {}             # 固定风量可能发生变化
        # 5.1 固定风量由大到小排序
        sortFixed = list(sorted(self.fixedQs.items(),key=operator.itemgetter(1),reverse=True))  # 由大到小，生成元组列表
        
    #     d = {'e1': 22, 'e2': 33, 'e3': 11}
    # # 按照值从大到小排序
    #     sorted_dict = dict(sorted(d.items(), key=lambda item: item[1], reverse=True))
    #     print(sorted_dict)
    #     input()
        # 5.2 由大到小依次尝试添加
        
        while len(sortFixed) :              # 循环体
            eid, q = sortFixed.pop(0)       # 弹出列表最大的固定风量
            newFixedQs.update({eid : q})    # 添加到合理固定风量
            self.__reset_fixed_q__(newFixedQs)
            flag,initQs = Dinic(self.edges, low_l=self.minInitQ, n=self.n)    # 初始化
            if not flag :                  # 初始化失败
                # del newFixedQs[eid]         # 删除
                newFixedQs.pop(eid)
        
        # 5.3 按合理的固定重新初始化
        # self.__reset_fixed_q__(newFixedQs)
        # initQs = Dinic(self.edges, low_l=self.minInitQ,n=self.n)

        # return initQs, list(newFixedQs.keys())
        return newFixedQs
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@




#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   正抛物线驼峰
#
def find_vertex(a, b, c):
    """
    a : 抛物线的常数项, 对应 y = ax^2 + bx + c
    b : 抛物线的一次项系数
    c : 抛物线的二次项系数（必须为负数，确保开口向下）, 用户保证数据正确性，函数不进行数据检查
    """
    # 计算顶点的 x 坐标
    x_vertex = -b / (2 * c)
    # 代入方程计算 y 坐标
    y_vertex = a + b * x_vertex + c * (x_vertex ** 2)
    return (x_vertex, y_vertex)
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   正抛物线与水平线交点
#
def find_intersections(a, b, c, y0=0):
    # 计算二次方程的判别式
    discriminant = b**2 - 4 * c * (a - y0)
    # print('discriminant==========',discriminant)
    
    # 根据判别式的值返回不同结果
    if discriminant < 0:
        return []
    elif discriminant == 0:
        x = (-b) / (2 * c)
        return [x]
    else:
        sqrt_d = math.sqrt(discriminant)
        x1 = (-b + sqrt_d) / (2 * c)
        x2 = (-b - sqrt_d) / (2 * c)
        return sorted([x1, x2],reverse=True)  # 由大到小排序

# # 获取用户输入
# a = float(input("请输入抛物线的系数 a: "))
# b = float(input("请输入抛物线的系数 b: "))
# c = float(input("请输入抛物线的系数 c (需为负数): "))
# y0 = float(input("请输入水平线 y0 的值: "))

# # 检查抛物线是否开口向下
# if c >= 0:
#     print("警告：系数 c 应为负数以确保抛物线开口向下。")

# # 计算交点
# solutions = find_intersections(a, b, c, y0)

# # 输出结果
# if len(solutions) == 0:
#     print(f"抛物线与水平线 y={y0} 无交点。")
# elif len(solutions) == 1:
#     print(f"抛物线与水平线 y={y0} 相切，交点为 x = {solutions[0]:.4f}。")
# else:
#     print(f"抛物线与水平线 y={y0} 有两个交点：")
#     print(f"x₁ = {solutions[0]:.4f}, x₂ = {solutions[1]:.4f}")
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


