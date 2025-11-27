#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#   通风阻力系数、机站、局部风机冗余风压反演-2025
#
###########################################################################################################################

#   1. 反演-染色体
#   （1）全部巷道风阻, 如果某巷道风阻固定不变，可通过调整上下界相等来控制
#   （2）井下基站、局部通风机作用在系统上的剩余风压fanH, 如果主通风机风压值可靠，可通过调整上下界相等来控制

#   2. 目标值、适应值
#   2.1 单目标
#   （1）目标值：巷道测定风量、构筑物巷道设定的微风量
#   （2）适应值：目标风量与解算风量欧氏距离，归一与否均可
#   2.2 多目标（下一版本）
#   （1）同单目标的风量目标
#   （2）将构筑物微风量单独作为一个目标维度
#   注：构筑物压差值直接进入网络解算

#   3. 数据
#   （1）网络解算数据dataNS, 不含r、fanH, r、fanH作为染色体
#   （2）染色体上下界
#   （3）targetQ

#   4. 调试经验
#   （4）经常出现的构筑物风量为负，说明构筑物压差值测试值偏大，需减小构筑物压差，前提是巷道阻力基本正确
#       巷道阻力大也会出现风量负

#   5. 程序改进
#   1. 为了减少IO型数据传输量和计算量
#   （1）初始风量外部输入
#   （2）权重计算、找回路放在并行计算，每个个体计算一次，利用初始风量
#   （3）本地机多线程拓扑数据在并行中复制，多进程并行也无需复制
#   （4）云端，
#   数据分为拓扑+初始风量+构筑物、R、剩余风压



#   输入数据
"""
dataInversionRRedunH数据结构 = {
    "roads": [                      # 风路列表, list, 必有
        {                           # 风路对象
            "id"        :   str,    # 井巷id,   str, 必有
            "s"         :   str,    # 始节点id, id, 必有
            "t"         :   "str",  # 末节点id, id, 必有
            "minR"      :   float,  # 风阻下限, float, 必有
            "maxR",     :   float,  # 风阻上限, float, 必有
            "targetQ"   :   float,  # 测试风量, float, 必有
            "initQ"     :   float   # 初始风量, float, 必有
        },
        ......
    ],
    "fanAs": [
    
    ],         # 同dataNS
    "fanBs": [
    
    ],         # 同dataNS
    "fanHs": [                              # 静压型风机列表, list, ==必有==
        {                                   # 风机对象
            "id"            :   str,        # 风机id, str, 必有
            "eid"           :   str,        # 绑定巷道id, id, 必有
            "minH"          :   float,
            "maxH"          :   float
            "direction",    :   "forward",  # 动力方向, enum, ["forward", "reverse"], 必有
            "pitotLocation" :   "in"        # 皮托管位置, enum, ["in", "out"], 必有
        },
        ......
    ],
    "structureHs": [            # 井巷构筑物列表, list, 可缺省
        {                       # 构筑物对象
            "id"    :   str,    # 构筑物id, str, 必有
            "eid"   :   str,    # 绑定对象id, id, 必有
            "h"     :   float   # 构筑物压差, float, 必有
        },
        ......
    ],
    "structureRs": [
        {
            "id"    :   str,    # 构筑物id, str, 必有
            "eid"   :   str,    # 绑定对象id, id, 必有
            "r"     :   float   # 构筑物压差, float, 必有            
        }
    ]
}
"""
# 输出结果
"""
{
    "eid: float,     # eid : r
    ...
    "fanHId": float,  # fanHId: h
    ...
}
"""

#   ====== 注意 ======  要保持风路列表次序全周期的一致性

import  geatpy  as  ea
import  numpy   as  np
np.seterr(divide='ignore', invalid='ignore')    # 设置numpy，防止出现0/0的情况


from    jl.vn.ns.ns     import  mvn_solver

from    jl.ga.problem   import  Myproblem
from    .customAimFuct    import  CustomAimFuct

from    jl.ga.algorithm     import  CreateAlgorithm
from    jl.ga.population    import  create_population
import  numpy   as  np
import  json

from    pprint  import  pprint

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   风阻系数反演-风量风压GA反演法---XXX
#
def inversion_r_redun_h(dataInversionRRedunH, configNS, configGA) :
    """
    dataInversionRRedunH    :   通风阻力系数、冗余动力反演数据
                        阻力目标值）、通风动力静压数据、构筑物压差数据
    configNC        :   解网配置参数
    configGA        :   GA配置参数
    """

    # 1. 解析dataInversionRRedunH
    # 1.1 提取dataNS
    dataNS = {}
    dataNS["roads"] = []
    for road in dataInversionRRedunH['roads']:
        dataNS["roads"].append(dict(zip(
            ['id','s','t','initQ'],
            [road['id'],road['s'],road['t'],road['initQ']]
        )))
    if dataInversionRRedunH.get("fanAs", None)          :    dataNS["fanAs"]        =   dataInversionRRedunH["fanAs"]
    if dataInversionRRedunH.get("fanBs", None)          :    dataNS["fanBs"]        =   dataInversionRRedunH["fanBs"]
    if dataInversionRRedunH.get('structureHs', None)    :   dataNS["structureHs"]   =   dataInversionRRedunH["structureHs"]
    if dataInversionRRedunH.get('structureRs', None)    :   dataNS["structureRs"]   =   dataInversionRRedunH["structureRs"]

    # 1.2 提取染色体（风阻、冗余动力, 一个链）上下限
    lb = []        # 染色体下界 minRs
    ub = []        # 染色体上界 maxRs
    for road in dataInversionRRedunH['roads']:         # 风路循环
        lb.append(road["minR"])          # 下限
        ub.append(road["maxR"])          # 上限
    for fanH in dataInversionRRedunH['fanHs']:
        lb.append(fanH["minH"])          # 下限
        ub.append(fanH["maxH"])          # 上限

    # 1.3 提取目标风量
    targetQs = {road['id']:road['targetQ'] for road in dataInversionRRedunH['roads'] if road.get('targetQ', None)}

    # 1.4 多目标决策算法设置参数更新（用户设置参数刷新默认参数）
    configGA0 = {                               # 默认参数设置
        "algorithmId"       :   20,             # 多目标差分进化DE
        "name"              :   "muti_target",  # 多目标决策
        "poolType"          :   "Thread",       # 'Thread'适合io密集型任务, 
                                                # ‘Process’适合计算型任务, 
                                                # 串行'Serial' #有错误
        "M"                 :   1,              # 目标维数, 风量决策1维, 风量压差决策2维
        "MAXGEN"            :   800,            # 最大进化代数
        "maxormin"          :   1,              # 适应值类型, 1为最小值, -1最大值
        "varTypes"          :   [0],            # 变量类型为连续性
        "lbin"              :   [1],            # 下边界，1表示包含，0表示不包含  
        "ubin"              :   [1],            # 上边界，1表示包含，0表示不包含
        "Encoding"          :   "RI",           # 编码类型, 默认实数编码    ===pop
        "NIND"              :   100,            # 初始种群数
        "F"                 :   0.5,            # 变异率，最大1
        "X"                 :   0.5,            # 交叉率，最大1
        "trappedValue"      :   1e-3,           # 停止条件1e-6
        "maxTrappedCount"   :   10,             # 进化停滞步数
        "logTras"           :   1,              # int, 日志步长Tras即周期的意思，该参数用于设置在进化过程中每多少代记录一次日志信息。
                                                # 设置为0表示不记录日志信息。
                                                # 注：此时假如设置了“每10代记录一次日志”而导致最后一代没有被记录，
                                                # 则会补充记录最后一代的信息，除非找不到可行解。
        "printOrnot"        :   False,          # 打印日志
        "cloudComputing"    :   True           # 云计算标识
        # "threadPoolNum"     :   3000           # 云计算线程数
    }
    configGATemp = configGA0.copy()
    configGATemp.update(configGA)
    configGA = configGATemp

    # 2. 构造问题对象
    # 2.1 加载外部数据
    chromosome_ids = [road['id'] for road in dataInversionRRedunH['roads']]
    if dataInversionRRedunH.get('fanHs'):
        chromosome_ids.extend([fanH['id'] for fanH in dataInversionRRedunH['fanHs']])

    data = {
        "dataNS"    :   dataNS,       # 网络解算迭代,回路等放在适应函数
        "configNS"  :   configNS,
        "targetQs"  :   targetQs,       # 目标风量
        "chromosomeIds": chromosome_ids,
        "iterationResultsDir": configGA.get("iterationResultsDir"),
        "bestSaveInterval": configGA.get("bestSaveInterval"),
        "adaptiveMode": configGA.get("adaptiveMode", False)
    }

    # 2.2 创建问题对象
    myProblem = Myproblem(          # 构造问题对象
        name            =   configGA["name"],        # 多目标
        M               =   configGA["M"],                        # 目标函数维数, 风量1维, 风量压差2维
        maxormins       =   [configGA["maxormin"]]*configGA["M"],      # 目标函数最大或最小值，-1最大，1最小
        Dim             =   len(lb),                  # 函数变量维数，若与data数据中的变量维数不同，会报错
        varTypes        =   configGA["varTypes"] * len(lb),        # 变量类型, 默认 0 连续型
        lb              =   lb,                             # 决策变量下界
        ub              =   ub,                             # 决策变量上界
        lbin            =   configGA["lbin"] * len(lb),     # 下边界，1表示包含，0表示不包含
        ubin            =   configGA["ubin"] * len(lb) ,    # 上边界，1表示包含，0表示不包含
        # aimFunc = custom_aimFuction
        aimFunc         =   None,
        CustomAimFunc   =   CustomAimFuct,
        data            = data,                             # 外部导入数据
        fitnesss        = list(),                           # 全部个体适应值
        poolType        =   configGA["poolType"]  
    )

    # 3. 创建染色体种群对象
    population = create_population(
        myProblem.varTypes,     # problem.varTypes = np.array(varTypes)
        myProblem.ranges,       # problem.ranges = np.array([lb, ub])  # 初始化ranges（决策变量范围矩阵）
        myProblem.borders,      # problem.borders = np.array([lbin, ubin])  # 初始化borders（决策变量范围边界矩阵）
        configGA["NIND"],       # NIND,      # 种群规模，亦即个体数
        configGA["Encoding"]    # Encoding   染色体编码类型，实型
    )

    # 4. 创建进化算子对象（实例化种群对象）
    # 如果启用自适应模式，设置trappedValue和maxTrappedCount来实现提前终止
    adaptive_mode = configGA.get("adaptiveMode", False)
    if adaptive_mode:
        # 使用自适应配置的收敛阈值和停滞计数
        trapped_value = configGA.get("convergenceThreshold", 1e-5)
        max_trapped_count = configGA.get("stagnationCount", 30)
    else:
        # 使用默认配置
        trapped_value = configGA.get("trappedValue", 1e-3)
        max_trapped_count = configGA.get("maxTrappedCount", 10)
    
    algorithm = CreateAlgorithm(
        problem         =   myProblem,                      # class <Problem> 问题类的对象
        population      =   population,                     # class <Population> 染色体种群对象
        id              =   configGA["algorithmId"],        # int - 算法id，默认多目标差分进化 - 20
        F               =   configGA["F"],                  # float - 变异率, 最大1
        X               =   configGA["X"],                  # float - 交叉率，最大1
        MAXGEN          =   configGA["MAXGEN"],             # 最大进化代数，generation
        trappedValue    =   trapped_value,                 # “进化停滞”判断阈值
        maxTrappedCount =   max_trapped_count,              # 进化停滞计数最大上限值
        # logTras         =   configGA["logTras"],            # 日志步长
        logTras         =   1,            # 日志步长
        verbose         =   configGA["printOrnot"],         # 是否打印日志
        drawing         =   0 # 设置绘图方式(0：不绘图，1：绘制结果图，2：绘制目标空间过程动画，3：绘制决策空间过程动画)
    ) # 算法选择

    # 5. 进化迭代计算（支持自适应终止）
    roadRs = Run(algorithm, myProblem, configGA)

    # print('ok======',myProblem.ReferObjV)
    # print("2735498-----",algorithm.logTras)

    # 6. 返回反演风阻
    return roadRs, myProblem.fitnesss
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€

# {
#     'circuits': [
#         [{'d': 1, 'eid': 'e6'}, {'d': 1, 'eid': 'e5'}, {'d': -1, 'eid': 'e2'}, {'d': 1, 'eid': 'e3'}], 
#         [{'d': 1, 'eid': 'e8'}, {'d': -1, 'eid': 'e7'}, {'d': 1, 'eid': 'e5'}, {'d': -1, 'eid': 'e2'}, {'d': 1, 'eid': 'e3'}], 
#         [{'d': 1, 'eid': 'e9'}, {'d': -1, 'eid': 'e4'}, {'d': -1, 'eid': 'e5'}, {'d': 1, 'eid': 'e7'}], 
#         [{'d': 1, 'eid': 'e10'}, {'d': 1, 'eid': 'e1'}, {'d': 1, 'eid': 'e2'}, {'d': 1, 'eid': 'e4'}]
#     ], 
#     'comRoads': {
#         'e1': [{'id': 'e1', 's': 'v1', 't': 'v2', 'ex': 2.0, 'minR': 0.00054, 'maxR': 5.4, 'targetH': 610.9026565530529, 'targetQ': 106.3626445627875, 'objName': 'road'}], 
#         'e2': [{'id': 'e2', 's': 'v2', 't': 'v3', 'ex': 2.0, 'minR': 0.000457, 'maxR': 4.569999999999999, 'targetH': 163.55267926855885, 'targetQ': 59.8233503273479, 'objName': 'road'}], 'e3': [{'id': 'e3', 's': 'v2', 't': 'v4', 'ex': 2.0, 'minR': 0.000341, 'maxR': 3.4099999999999997, 'targetH': 73.82565507184059, 'targetQ': 46.52929423543958, 'objName': 'road'}], 'e4': [{'id': 'e4', 's': 'v3', 't': 'v7', 'ex': 2.0, 'minR': 0.000337, 'maxR': 3.37, 'targetH': 197.8776671904187, 'targetQ': 76.62728538963475, 'objName': 'road'}], 'e5': [{'id': 'e5', 's': 'v5', 't': 'v3', 'ex': 2.0, 'minR': 0.000286, 'maxR': 2.86, 'targetH': 9.06563096588081, 'targetQ': 17.803935062286826, 'objName': 'road'}], 'e6': [{'id': 'e6', 's': 'v4', 't': 'v5', 'ex': 2.0, 'minR': 0.0008269999999999999, 'maxR': 8.27, 'targetH': 80.66139323083748, 'targetQ': 31.230583952474298, 'objName': 'road'}], 'e7': [{'id': 'e7', 's': 'v5', 't': 'v6', 'ex': 2.0, 'minR': 0.0016370000000000002, 'maxR': 16.37, 'targetH': 34.02336956699572, 'targetQ': 14.416648890187437, 'objName': 'road'}], 'e8': [{'id': 'e8', 's': 'v4', 't': 'v6', 'ex': 2.0, 'minR': 0.0049, 'maxR': 49.0, 'targetH': 114.68476279783314, 'targetQ': 15.298710282965308, 'objName': 'road'}], 'e9': [{'id': 'e9', 's': 'v6', 't': 'v7', 'ex': 2.0, 'minR': 0.001957, 'maxR': 19.57, 'targetH': 172.91992858930368, 'targetQ': 29.725359173152754, 'objName': 'road'}], 
#         'e10': [
#             {'id': 'e10', 's': 'v7', 't': 'v8', 'ex': 2.0, 'minR': 0.00082, 'maxR': 8.200000000000001, 'targetH': 927.6669969879691, 'targetQ': 106.3626445627875, 'objName': 'road'}, 
#             {'id': 'fan1', 'eid': 'e10', 'h': -1900.0, 'sword': 1, 'qD': 1, 'objName': 'fanH'}
#         ]
#     }, 
#     'roadQs': {'e1': 106.3626445627875, 'e2': 59.8233503273479, 'e3': 46.52929423543958, 'e4': 76.62728538963475, 'e5': 17.803935062286826, 'e6': 31.230583952474298, 'e7': 14.416648890187437, 'e8': 15.298710282965308, 'e9': 29.725359173152754, 'e10': 106.3626445627875}, 'algorithm': {'loopN': 100, 'iterN': 30, 'minQ': 0.001, 'minH': 0.01}}












# from    jl.common.errorCodes        import  *


from    jl.ga.algorithm     import  CreateAlgorithm
# from    jl.ga.population    import  CreatePopulation

from    .customAimFuct      import  CustomAimFuct
from    .adaptive_de       import  should_terminate_early

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#
#
def GetAlgo(configGA) :
    """
    遗传算法主函数，输出运算结果！
    data: 网络数据 数据类型为字典  {sheet:[{key: value}],...}
    target: 目标值，数据类型为字典 {key: value,...}
    poolType: 运行方式设置，默认为Serial串行计算，Thread为多线程，Process为多进程
    iter_num: 迭代步数，若不设置默认为10
    M: 适应值函数维数，默认为1
    Dim: 函数变量维数，若与data数据中的变量维数不同，会报错
    lb: 变量下界
    ub: 变量上界
    Encoding: 变量编码形式，默认为'RI'实数编码
    NIND: 种群个体数，默认为10
    trappedValue: 停止误差阈值, 默认为1e-6
    maxTrappedCount: 进化停滞步数，默认为10
    log_step: 日志步长, 默认为1
    p_verbose: 是否打印日志，默认为True
    """
    configGA_0 = {        # 默认参数设置
        "algorithmId"       :   20,         # 多目标差分进化DE
        "name"              :   "muti_target",  # 多目标决策
        "poolType"          :   "Thread",   # 'Thread'适合io密集型任务, 
                                        # ‘Process’适合计算型任务, 
                                        # 串行'Serial' #有错误
        "M"                 :   1,          # 目标维数, 风量决策1维, 风量压差决策2维
        "MAXGEN"           :   800,         # 最大进化代数
        "maxormin"          :   1,          # 适应值类型, 1为最小值, -1最大值
        "varTypes"          :   [0],        # 变量类型为连续性
        "lbin"              :   [1],        # 下边界，1表示包含，0表示不包含  
        "ubin"              :   [1],        # 上边界，1表示包含，0表示不包含
        "Encoding"          :   "RI",       # 编码类型, 默认实数编码    ===pop
        "NIND"              :   100,        # 初始种群数
        "F"                 :   0.5,       # 变异率，最大1
        "X"                 :   0.5,        # 交叉率，最大1
        "trappedValue"      :   1e-3,       # 停止条件1e-6
        "maxTrappedCount"   :   10,         # 进化停滞步数
        "logTras"           :   1,          # int, 日志步长Tras即周期的意思，该参数用于设置在进化过程中每多少代记录一次日志信息。
                                            # 设置为0表示不记录日志信息。
                                            # 注：此时假如设置了“每10代记录一次日志”而导致最后一代没有被记录，
                                            # 则会补充记录最后一代的信息，除非找不到可行解。
        "printOrnot"        :   False,      # 打印日志
        "cloudComputing"    :   True,       # 云计算标识
        "threadPoolNum"     :   1000,       # 云计算线程数
        "errorNormalized"   :   False,      # 不归一
        "targetType"        :   "q"         # q, qh
    }
    algo = configGA_0.copy()
    algo.update(configGA)

    return algo
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
'''
 MAXGEN          : int      - 最大进化代数。
    
    currentGen      : int      - 当前进化的代数。
    
    MAXTIME         : float    - 时间限制（单位：秒）。
    
    timeSlot        : float    - 时间戳（单位：秒）。
    
    passTime        : float    - 已用时间（单位：秒）。
    
    MAXEVALS        : int      - 最大评价次数。
    
    evalsNum        : int      - 当前评价次数。
    
    MAXSIZE         : int      - 最优个体的最大数目。
    
    logTras         : int      - Tras即周期的意思，该参数用于设置在进化过程中每多少代记录一次日志信息。
                                 设置为0表示不记录日志信息。
                                 注：此时假如设置了“每10代记录一次日志”而导致最后一代没有被记录，
                                     则会补充记录最后一代的信息，除非找不到可行解。

    log             : Dict     - 日志记录。其中包含2个基本的键：'gen'和'eval'，其他键的定义由该算法类的子类实现。
                                 'gen'的键值为一个list列表，用于存储日志记录中的每一条记录对应第几代种群。
                                 'eval'的键值为一个list列表，用于存储进化算法的评价次数。
                                 注：若设置了logTras为0，则不会记录日志，此时log会被设置为None。
    
    verbose         : bool     - 表示是否在输入输出流中打印输出日志信息。
'''
   
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   进化运行函数（支持自适应终止）
#
# def Run(myAlgorithm,eids) :
def Run(myAlgorithm, myProblem=None, configGA=None) :
    """
    执行算法模板，支持自适应终止
    
    参数:
        myAlgorithm: 算法对象
        myProblem: 问题对象（用于访问fitnesss）
        configGA: 配置参数（用于自适应判断）
    """
    # 如果启用自适应模式，利用trappedValue和maxTrappedCount实现提前终止
    # 注意：由于geatpy的run()是一次性执行的，我们通过设置合适的trappedValue
    # 和maxTrappedCount来实现提前终止，但这需要算法内部支持
    # 这里我们采用更实用的方式：正常运行，但在运行后检查是否应该提前终止
    # 并在日志中记录相关信息
    
    adaptive_mode = configGA and configGA.get("adaptiveMode", False)
    original_maxgen = myAlgorithm.MAXGEN
    
    # 运行算法
    [NDSet, population] = myAlgorithm.run() # 执行算法模板
    
    # 如果启用自适应模式，检查是否应该提前终止（用于日志记录）
    if adaptive_mode and myProblem is not None:
        actual_gen = myAlgorithm.currentGen if hasattr(myAlgorithm, 'currentGen') else original_maxgen
        if actual_gen < original_maxgen:
            print(f"自适应终止：实际运行 {actual_gen} 代（最大 {original_maxgen} 代）")
        else:
            # 检查是否收敛（用于信息输出）
            should_term, reason, _ = should_terminate_early(
                myProblem.fitnesss, 
                configGA
            )
            if should_term:
                print(f"注意：算法已运行到最大代数，但可能已收敛：{reason}")
    
    # NDSet.save()                          # 把非支配种群的信息保存到文件中

    # 输出-------次序
    # eids = [obj[ID] for obj in data["roads"]]   # 不是列表
    print("@@@@@@@@@@@@===========",population.ObjV)
    if NDSet.sizes != 0:
        # return dict(zip(eids,NDSet.Phen[0]))  # 解算结果[{id:r}]
        return NDSet.Phen[0]
    else:
        MyAppValueError(
            message= 'Evolutionary iteration error !',
            code= 1, # 网络解算错误，无合适解
            data=None
        )
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€



