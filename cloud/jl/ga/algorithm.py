#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   遗传和进化算法框架
#
######################################################################################

# 遗传算法库
import  geatpy  as  ea
import  numpy   as  np
np.seterr(divide='ignore', invalid='ignore')    # 设置numpy，防止出现0/0的情况

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#
#   遗传与进化算法问题类
#
# class Myproblem(ea.Problem) :
#     """遗传与进化算法问题类
#     :问题类是用来存储与待求解问题相关信息的一个类，继承遗传算法geatpy库中的问题类
#     :ea.Problem 父类    
#     """
#     #€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#     #   构造函数
#     def __init__(
#         self,
"""
Algorithm : class - 算法模板顶级父类

描述:
    算法设置类是用来存储与算法运行参数设置相关信息的一个类。

属性:
    name            : str      - 算法名称（可以自由设置名称）。
    
    problem         : class <Problem> - 问题类的对象。

    population      : class <Population> - 种群对象。
    
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

函数:
    __init__()       : 构造函数，定义一些属性，并初始化一些静态参数。

    initialization() : 在进化前对算法模板的一些动态参数进行初始化操作，具体功能需要在继承类中实现。
    
    run()            : 执行函数，具体功能需要在继承类中实现。

    logging()        : 用于在进化过程中记录日志，具体功能需要在继承类中实现。

    stat()           : 用于分析当代种群的信息，具体功能需要在继承类中实现。

    terminated()     : 计算是否需要终止进化，具体功能需要在继承类中实现。

    finishing ()     : 进化完成后调用的函数，具体功能需要在继承类中实现。

    check()          : 用于检查种群对象的ObjV和CV的数据是否有误。

    call_aimFunc()   : 用于调用问题类中的aimFunc()进行计算ObjV和CV(若有约束)。

    display()        : 用于在进化过程中进行一些输出，需要依赖属性verbose和log属性。
    
""" 

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   进化算法设置函数    
#
def CreateAlgorithm(
        problem,                    # class <Problem> - 问题类的对象
        population,                 # class <Population> - 种群对象
        id              =   20,     # int - 算法id，默认多目标差分进化
        F               =   0.5,    # float - 变异率, 最大1
        X               =   0.5,    # float - 交叉率，最大1
        MAXGEN          =   None,   # 进化迭代数
        trappedValue    =   None,   # “进化停滞”判断阈值
        maxTrappedCount =   None,   # 进化停滞计数最大上限值
        logTras         =   None,   # 日志步长
        verbose         =   None,   # 是否打印日志
        drawing         =   0       # 设置绘图方式(0：不绘图，1：绘制结果图，2：绘制目标空间过程动画，
                                    # 3：绘制决策空间过程动画)

    ):

    """
    进化算法选择函数
    其中, 以“soea”开头的是单目标进化算法模板, 以“moea”开头的是多目标进化算法模板。    
    """

    is_Pm = True        # 变异概率，在[0,1]区间
    is_Xovr = True      # 差分变异缩放因子，无区间限制
    is_F = False        # 交叉概率，在[0,1]区间

    if id == 1:
         """
        soea_SGA_templet : class - Simple GA templet(最简单、最经典的遗传算法模板)
        算法描述:
            本模板实现的是最经典的单目标遗传算法。算法流程如下：
            1) 根据编码规则初始化N个个体的种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 独立地从当前种群中选取N个母体。
            5) 独立地对这N个母体进行交叉操作。
            6) 独立地对这N个交叉后的个体进行变异，得到下一代种群。
            7) 回到第2步。
        """
         algorithm = ea.soea_SGA_templet(problem,population)
        

    if id == 2:
        '''
        精英保留的遗传算法，适用于'BG''、''RI''或''P'编码
        算法描述:
            本模板实现的是基于杰出保留的单目标遗传算法。算法流程如下：
            1) 根据编码规则初始化N个个体的种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 独立地从当前种群中选取N-1个母体。
            5) 独立地对这N-1个母体进行交叉操作。
            6) 独立地对这N-1个交叉后的个体进行变异。
            7) 计算当代种群的最优个体，并把它插入到这N-1个交叉后的个体的第一位，得到新一代种群。
            8) 回到第2步。
        '''
        algorithm = ea.soea_EGA_templet(problem,population)
    
    if id == 3:
        '''
        增强精英保留的遗传算法
        算法描述:
            本模板实现的是增强精英保留的遗传算法。算法流程如下：
            1) 根据编码规则初始化N个个体的种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 独立地从当前种群中选取N个母体。
            5) 独立地对这N个母体进行交叉操作。
            6) 独立地对这N个交叉后的个体进行变异。
            7) 将父代种群和交叉变异得到的种群进行合并，得到规模为2N的种群。
            8) 从合并的种群中根据选择算法选择出N个个体，得到新一代种群。
            9) 回到第2步。

            该算法宜设置较大的交叉和变异概率，否则生成的新一代种群中会有越来越多的重复个体。
        '''
        algorithm = ea.soea_SEGA_templet(problem,population)
    if id == 4:
        """
        soea_GGAP_SGA_templet : class - Generational Gap Simple GA templet(带代沟的简单遗传算法模板)

        算法描述:
            本模板实现的是带代沟的简单遗传算法，
            它在SGA算法模板的基础上增加“代沟”，用于控制使用多少个子代替换父代来形成新一代种群，算法流程如下：
            1) 根据编码规则初始化N个个体的种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 独立地从当前种群中选取N个母体。
            5) 独立地对这N个母体进行交叉操作。
            6) 独立地对这N个交叉后的个体进行变异，并根据代沟从中选择N个个体替换父代最差的N个个体，得到下一代种群。
            7) 回到第2步。
        """ 
        algorithm = ea.soea_GGAP_SGA_templet(problem,population)
    if id == 5:
        """
        soea_steadyGA_templet : class - Steady State GA templet(稳态遗传算法)

        算法描述:
            本模板实现的是稳态遗传算法，算法流程如下：
            1) 根据编码规则初始化N个个体的种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 独立地从当前种群中选取2个母体。
            5) 独立地对这2个母体进行交叉操作。
            6) 独立地对这2个交叉后的个体进行变异。
            7) 将这2个母体和由交叉变异得到的个体进行一对一生存者竞争选择。
            8) 将第7步得到的个体替换父代中原母体所在位置的个体，形成新一代种群。
            9) 回到第2步。
            
        """
        algorithm = ea.soea_steadyGA_templet(problem,population)

    if id == 6:
        """
        soea_studGA_templet.py - Stud GA templet(种马遗传算法模板)

        算法描述:
            本模板实现的是种马遗传算法。算法流程详见参考文献[1]。

        参考文献:
            [1] Khatib W , Fleming P J . The stud GA: A mini revolution?[C]// International 
            Conference on Parallel Problem Solving from Nature. Springer, Berlin, Heidelberg, 1998.
            
        """
        algorithm = ea.soea_studGA_templet(problem,population)
    if id == 7:
        """
        soea_ES_1_plus_1_templet : class - (1+1)进化策略模板

        算法描述:
            本模板实现的是(1+1)进化策略。算法流程如下：
            1) 根据编码规则初始化N个个体的种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 初始化控制高斯变异中的标准差Sigma(Geatpy中的高斯变异算子传入的是3倍的标准差即Sigma3)。
            5) 独立地对这种群个体进行高斯变异，得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群，
            同时统计新一代种群中有多少个个体继承自实验种群（即变异成功率）。
            7) 根据变异成功率修改Sigma3。
            8) 回到第2步。

        参考文献:
            [1] Beyer H G , Schwefel H P . Evolution strategies – A comprehensive 
            introduction[J]. Natural Computing, 2002, 1(1):3-52.

        """
        algorithm = ea.soea_ES_1_plus_1_templet(problem,population)
        is_Pm = False # 无变异算子
        is_Xovr = False # 无交叉算子

    if id == 8:
        """
        soea_ES_miu_plus_lambda_templet : class - (μ+λ)进化策略模板

        算法描述:
            本模板实现的是(μ+λ)进化策略[1]。

        参考文献:
            [1] Beyer H G , Schwefel H P . Evolution strategies – A comprehensive 
            introduction[J]. Natural Computing, 2002, 1(1):3-52.

        """
        algorithm = ea.soea_ES_miu_plus_lambda_templet(problem,population)
        is_Pm = False # 无变异算子

    if id == 9:
        '''
        soea_DE_best_1_bin_templet : class - 差分进化DE/best/1/bin算法模板
        '编码方式必须为实数'RI'
        算法描述:
            本模板实现的是DE/best/1/bin单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 选择变异的基向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用二项式分布交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
    '''
        algorithm = ea.soea_DE_best_1_bin_templet(problem,population)
        is_Pm = False # 无变异算子
        is_F = True # # 差分变异缩放因子
    if id == 10:
        """
        soea_DE_best_1_L_templet : class - 差分进化DE/best/1/L算法模板
        '编码方式必须为实数'RI'
        算法描述:
            本模板实现的是经典的DE/best/1/L单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 选择变异的基向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_best_1_bin_templet(problem,population)
        is_Pm = False # 无变异算子
        is_F = True # # 差分变异缩放因子
    if id == 11:
        """
        soea_DE_currentToBest_1_bin_templet : class - 差分进化DE/current-to-best/1/bin算法模板
        '编码方式必须为实数'RI'
        算法描述:
            本模板实现的是经典的DE/current-to-best/1/bin单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 采用current-to-best的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用二项式分布交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_currentToBest_1_bin_templet(problem,population)
        is_Pm = False # 无变异算子
        is_F = True # # 差分变异缩放因子
    if id == 12:
        """
        soea_DE_currentToBest_1_L_templet : class - 差分进化DE/current-to-best/1/bin算法模板
        '编码方式必须为实数'RI'
        算法描述:
            为了实现矩阵化计算，本模板采用打乱个体顺序来代替随机选择差分向量。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 采用current-to-best的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_currentToBest_1_bin_templet(problem,population)
        is_Pm = False # 无变异算子
        is_F = True # # 差分变异缩放因子

    if id == 13:
        """
        soea_DE_currentToBest_1_L_templet : class - 差分进化DE/current-to-best/1/bin算法模板
        '编码方式必须为实数'RI'
        算法描述:
            为了实现矩阵化计算，本模板采用打乱个体顺序来代替随机选择差分向量。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 采用current-to-best的方法选择差分变异的各个向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_currentToBest_1_L_templet(problem,population)
        is_Pm = False # 无变异算子
        is_F = True # # 差分变异缩放因子
    if id == 14:
        """
        soea_DE_currentToRand_1_templet : class - 差分进化DE/current-to-rand/1算法模板
        '编码方式必须为实数'RI'
        算法描述:
            本模板实现的是经典的DE/current-to-rand/1单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 采用current-to-rand/1的重组方法对当前种群进行差分变异和重组（它等价于将DE/rand/1的变异算法和旋转不变的线性重组算法的结合），得到实验个体。
            5) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            6) 回到第2步。
        """
        algorithm = ea.soea_DE_currentToRand_1_templet(problem,population)
        is_Xovr= False # 无变异算子
        is_F = True # # 差分变异缩放因子
    if id == 15:
        """
        soea_DE_rand_1_bin_templet : class - 差分进化DE/rand/1/bin算法模板       
        '编码方式必须为实数'RI'
        算法描述:
            本模板实现的是经典的DE/rand/1/bin单目标差分进化算法。算法流程如下：
            1) 初始化候选解 种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 选择差分变异的基向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用二项式分布交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_rand_1_bin_templet(problem,population)
        is_Pm = False # 无变异算子
        is_F = True # # 差分变异缩放因子
    if id == 16:
        """ 
        '编码方式必须为实数'RI'
        soea_DE_rand_1_L_templet : class - 差分进化DE/rand/1/L算法模板

        算法描述:
            本模板实现的是经典的DE/rand/1/L单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 选择差分变异的基向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_rand_1_L_templet(problem,population)
        is_Pm = False # 无变异算子 
        is_F = True # # 存在差分变异缩放因子
    if id == 17:
        """ 
        '编码方式必须为实数'RI'
        soea_DE_targetToBest_1_bin_templet : class - 差分进化DE/target-to-best/1/bin算法模板
        算法描述:
            本模板实现的是经典的DE/target-to-best/1/bin单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 采用target-to-best的方法选择差分变异的基向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用二项式分布交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_targetToBest_1_bin_templet(problem,population)
        is_Pm = False # 无变异算子 
        is_F = True # # 存在差分变异缩放因子
    if id == 18:
        """ 
        '编码方式必须为实数'RI'
        soea_DE_targetToBest_1_L_templet : class - 差分进化DE/target-to-best/1/L算法模板

        算法描述:
            本模板实现的是经典的DE/target-to-best/1/L单目标差分进化算法。算法流程如下：
            1) 初始化候选解种群。
            2) 若满足停止条件则停止，否则继续执行。
            3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
            4) 采用target-to-best的方法选择差分变异的基向量，对当前种群进行差分变异，得到变异个体。
            5) 将当前种群和变异个体合并，采用指数交叉方法得到试验种群。
            6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
            7) 回到第2步。
        """
        algorithm = ea.soea_DE_targetToBest_1_L_templet(problem,population)
        is_Pm = False # 无变异算子 
        is_F = True # # 存在差分变异缩放因子
    if id == 19:
        """
        moea_awGA_templet : class - 多目标进化优化awGA算法模板
            
        算法描述:
            采用awGA进行多目标优化。
            
        参考文献:
            [1] Gen M,CHeng R. Genetic Algorithms and Engineering Optimization[M]. 
            New York: John Wiley & Sons,2000
                
        """
        algorithm = ea.moea_awGA_templet(problem,population) 
    if id == 20:        # 默认多目标差分进化
        """
        moea_NSGA2_DE_templet : class - 基于NSGA-II-DE算法的多目标进化算法模板
            
        算法描述:
            采用NSGA-II-DE进行多目标优化，
            与NSGA-II不同的是，该算法把NSGA-II中的子代生成部分替换成DE/rand/1/bin。
        """
        # print("====================",id,F,X)
        # input()

        # ea.moea_NSGA2_DE_templet(problem, population)
        algorithm = ea.moea_NSGA2_DE_templet(problem,population) 
        is_Pm = True       # False 无变异算子 
        is_F = True         # 存在差分变异缩放因子
    if id == 21:
        """
        moea_NSGA2_templet : class - 多目标进化NSGA-II算法模板
            
        算法描述:
            采用NSGA-II进行多目标优化
            算法详见参考文献[1]。
        参考文献:
            [1] Deb K , Pratap A , Agarwal S , et al. A fast and elitist multiobjective 
            genetic algorithm: NSGA-II[J]. IEEE Transactions on Evolutionary 
            Computation, 2002, 6(2):0-197.
        """
        
        algorithm = ea.moea_NSGA2_templet(problem,population) 
    if id == 22:
        """
        moea_NSGA3_DE_templet : class - 多目标进化优化NSGA-III-DE算法模板
        算法描述:
            采用NSGA-III-DE进行多目标优化，
            与NSGA-III不同的是，该算法把NSGA-III中的子代生成部分替换成DE/rand/1/bin。
            注意：在初始化染色体时，种群规模会被修正为NSGA-III所用的参考点集的大小。
        """
        algorithm = ea. moea_NSGA3_DE_templet(problem,population) 
        is_Pm = False # 无变异算子 
        is_F = True # # 存在差分变异缩放因子
    if id == 23:
        """
        moea_NSGA3_templet : class - 多目标进化优化NSGA-III算法模板
        算法描述:
            采用NSGA-III进行多目标优化。
            注意：在初始化染色体时，种群规模会被修正为NSGA-III所用的参考点集的大小。
        """
        algorithm = ea. moea_NSGA3_DE_templet(problem,population) 
    if id == 24:
        """
        moea_RVEA_RES_templet : class - 带参考点再生策略的多目标进化优化RVEA算法模板(RVEA With the Reference Vector Regeneration Strategy)
            
        算法描述:
            采用带参考点再生策略的RVEA进行多目标优化，即参考文献[1]中的RVEA*算法。
            该算法与RVEA算法类似，不过可以更好地解决具有复杂帕累托前沿面的多目标优化问题。
            """
        algorithm = ea.moea_RVEA_RES_templet(problem,population) 
    if id == 25:
        """
        moea_RVEA_templet : class - 多目标进化优化RVEA算法模板
            
        算法描述:
            采用RVEA进行多目标优化。
        """
        algorithm = ea.moea_RVEA_templet(problem,population) 


    # 参数        
    if is_Pm == True:
        algorithm.mutOper.Pm = F  # 变异概率，在[0,1]区间
    if is_F == True:
        algorithm.mutOper.F = F # 差分变异缩放因子，无区间限制
    if is_Xovr == True:
        algorithm.recOper.XOVR = X # 交叉概率，在[0,1]区间
    
    algorithm.MAXGEN          = MAXGEN           # 进化迭代数
    # myAlgorithm.mutOper = ea.Mutde(F=F)  # 生成差分变异算子对象
    # myAlgorithm.recOper = ea.Xovbd(XOVR=X)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr

    algorithm.trappedValue    = trappedValue      # “进化停滞”判断阈值
    algorithm.maxTrappedCount = maxTrappedCount   # 进化停滞计数最大上限值
    algorithm.logTras         = logTras           # 日志步长
    algorithm.verbose         = verbose        # 是否打印日志
    algorithm.drawing         = drawing # 设置绘图方式(0：不绘图，1：绘制结果图，2：绘制目标空间过程动画，3：绘制决策空间过程动画)

    return algorithm
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€






#                                       文档

# # -*- coding: utf-8 -*-
# from aem import con
# import numpy as np
# import geatpy as ea
# import warnings
# import time

# class Algorithm:
#     """
# Algorithm : class - 算法模板顶级父类

# 描述:
#     算法设置类是用来存储与算法运行参数设置相关信息的一个类。

# 属性:
#     name            : str      - 算法名称（可以自由设置名称）。
    
#     problem         : class <Problem> - 问题类的对象。

#     population      : class <Population> - 种群对象。
    
#     MAXGEN          : int      - 最大进化代数。
    
#     currentGen      : int      - 当前进化的代数。
    
#     MAXTIME         : float    - 时间限制（单位：秒）。
    
#     timeSlot        : float    - 时间戳（单位：秒）。
    
#     passTime        : float    - 已用时间（单位：秒）。
    
#     MAXEVALS        : int      - 最大评价次数。
    
#     evalsNum        : int      - 当前评价次数。
    
#     MAXSIZE         : int      - 最优个体的最大数目。
    
#     logTras         : int      - Tras即周期的意思，该参数用于设置在进化过程中每多少代记录一次日志信息。
#                                  设置为0表示不记录日志信息。
#                                  注：此时假如设置了“每10代记录一次日志”而导致最后一代没有被记录，
#                                      则会补充记录最后一代的信息，除非找不到可行解。

#     log             : Dict     - 日志记录。其中包含2个基本的键：'gen'和'eval'，其他键的定义由该算法类的子类实现。
#                                  'gen'的键值为一个list列表，用于存储日志记录中的每一条记录对应第几代种群。
#                                  'eval'的键值为一个list列表，用于存储进化算法的评价次数。
#                                  注：若设置了logTras为0，则不会记录日志，此时log会被设置为None。
    
#     verbose         : bool     - 表示是否在输入输出流中打印输出日志信息。

# 函数:
#     __init__()       : 构造函数，定义一些属性，并初始化一些静态参数。

#     initialization() : 在进化前对算法模板的一些动态参数进行初始化操作，具体功能需要在继承类中实现。
    
#     run()            : 执行函数，具体功能需要在继承类中实现。

#     logging()        : 用于在进化过程中记录日志，具体功能需要在继承类中实现。

#     stat()           : 用于分析当代种群的信息，具体功能需要在继承类中实现。

#     terminated()     : 计算是否需要终止进化，具体功能需要在继承类中实现。

#     finishing ()     : 进化完成后调用的函数，具体功能需要在继承类中实现。

#     check()          : 用于检查种群对象的ObjV和CV的数据是否有误。

#     call_aimFunc()   : 用于调用问题类中的aimFunc()进行计算ObjV和CV(若有约束)。

#     display()        : 用于在进化过程中进行一些输出，需要依赖属性verbose和log属性。
    
# """

#     def __init__(self):

#         """
#         描述: 
#             构造函数。
        
#         """
#         self.name = 'Algorithm'
#         self.problem = None
#         self.population = None
#         self.MAXGEN = None
#         self.currentGen = None
#         self.MAXTIME = None
#         self.timeSlot = None
#         self.passTime = None
#         self.MAXEVALS = None
#         self.evalsNum = None
#         self.MAXSIZE = None
#         self.logTras = None
#         self.log = None
#         self.verbose = None

#     def initialization(self):
#         pass

#     def run(self, pop):
#         pass

#     def logging(self, pop):
#         pass

#     def stat(self, pop):
#         pass

#     def terminated(self, pop):
#         pass

#     def finishing(self, pop):
#         pass

#     def check(self, pop):

#         """
#         描述:
#             用于检查种群对象的ObjV和CV的数据是否有误。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             无输出参数。

#         """

#         # 检测数据非法值
#         if np.any(np.isnan(pop.ObjV)):
#             warnings.warn(
#                 "Warning: Some elements of ObjV are NAN, please check the calculation of ObjV.(ObjV的部分元素为NAN，请检查目标函数的计算。)",
#                 RuntimeWarning)
#         elif np.any(np.isinf(pop.ObjV)):
#             warnings.warn(
#                 "Warning: Some elements of ObjV are Inf, please check the calculation of ObjV.(ObjV的部分元素为Inf，请检查目标函数的计算。)",
#                 RuntimeWarning)
#         if pop.CV is not None:
#             if np.any(np.isnan(pop.CV)):
#                 warnings.warn(
#                     "Warning: Some elements of CV are NAN, please check the calculation of CV.(CV的部分元素为NAN，请检查CV的计算。)",
#                     RuntimeWarning)
#             elif np.any(np.isinf(pop.CV)):
#                 warnings.warn(
#                     "Warning: Some elements of CV are Inf, please check the calculation of CV.(CV的部分元素为Inf，请检查CV的计算。)",
#                     RuntimeWarning)

#     def call_aimFunc(self, pop):

#         """
#         使用注意:
#             本函数调用的目标函数形如：aimFunc(pop), (在自定义问题类中实现)。
#             其中pop为种群类的对象，代表一个种群，
#             pop对象的Phen属性（即种群染色体的表现型）等价于种群所有个体的决策变量组成的矩阵。
#             若不符合上述规范，则请修改算法模板或自定义新算法模板。

#         描述:
#             该函数调用自定义问题类中自定义的目标函数aimFunc()得到种群所有个体的目标函数值组成的矩阵，
#             以及种群个体违反约束程度矩阵（假如在aimFunc()中构造了该矩阵的话）。
#             该函数不返回任何的返回值，求得的目标函数值矩阵保存在种群对象的ObjV属性中，
#             违反约束程度矩阵保存在种群对象的CV属性中。
#         例如：population为一个种群对象，则调用call_aimFunc(population)即可完成目标函数值的计算。
#              之后可通过population.ObjV得到求得的目标函数值，population.CV得到违反约束程度矩阵。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             无输出参数。

#         """

#         pop.Phen = pop.decoding()  # 染色体解码
#         if self.problem is None:
#             raise RuntimeError('error: problem has not been initialized. (算法模板中的问题对象未被初始化。)')
#         self.problem.aimFunc(pop)  # 调用问题类的aimFunc()
#         self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数
#         # 格式检查
#         if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
#                 pop.ObjV.shape[1] != self.problem.M:
#             raise RuntimeError('error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
#         if pop.CV is not None:
#             if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
#                 raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')

#     def display(self):

#         """
#         描述:
#             该函数打印日志log中每个键值的最后一条数据。假如log中只有一条数据或没有数据，则会打印表头。
#             该函数将会在子类中被覆盖，以便进行更多其他的输出展示。

#         """

#         self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算display()的耗时
#         headers = []
#         widths = []
#         values = []
#         for key in self.log.keys():
#             # 设置单元格宽度
#             if key == 'gen':
#                 width = max(3, len(str(self.MAXGEN - 1)))  # 因为字符串'gen'长度为3，所以最小要设置长度为3
#             elif key == 'eval':
#                 width = 8  # 因为字符串'eval'长度为4，所以最小要设置长度为4
#             else:
#                 width = 13  # 预留13位显示长度，若数值过大，表格将无法对齐，此时若要让表格对齐，需要自定义算法模板重写该函数
#             headers.append(key)
#             widths.append(width)
#             value = self.log[key][-1] if len(self.log[key]) != 0 else "-"
#             if isinstance(value, float):
#                 values.append("%.5E" % value)  # 格式化浮点数，输出时只保留至小数点后5位
#             else:
#                 values.append(value)
#         if len(self.log['gen']) == 1:  # 打印表头
#             header_regex = '|'.join(['{}'] * len(headers))
#             header_str = header_regex.format(*[str(key).center(width) for key, width in zip(headers, widths)])
#             print("=" * len(header_str))
#             print(header_str)
#             print("-" * len(header_str))
#         if len(self.log['gen']) != 0:  # 打印表格最后一行
#             value_regex = '|'.join(['{}'] * len(values))
#             value_str = value_regex.format(*[str(value).center(width) for value, width in zip(values, widths)])
#             print(value_str)
#         self.timeSlot = time.time()  # 更新时间戳


# class MoeaAlgorithm(Algorithm):  # 多目标优化算法模板父类

#     """
#     描述:
#         此为多目标进化优化算法模板的父类，所有多目标优化算法模板均继承自该父类。

#     对比于父类该类新增的变量和函数:

#         drawing        : int - 绘图方式的参数，
#                                0表示不绘图；
#                                1表示绘制最终结果图；
#                                2表示实时绘制目标空间动态图；
#                                3表示实时绘制决策空间动态图。

#         draw()         : 绘图函数。

#     """

#     def __init__(self, problem, population):

#         """
#         描述: 
#             在该构造函数里只初始化静态参数以及对动态参数进行定义。
        
#         """

#         super().__init__()  # 先调用父类构造函数
#         self.problem = problem
#         self.population = population
#         self.logTras = 1  # 默认设置logTras的值为1
#         self.verbose = True  # 默认设置verbose的值为True
#         self.drawing = 1  # 默认设置drawing的值为1
#         self.ax = None  # 存储动态图像

#     def initialization(self):

#         """
#         描述: 
#             该函数用于在进化前对算法模板的一些动态参数进行初始化操作。
#             该函数需要在执行算法模板的run()函数的一开始被调用，同时开始计时，
#             以确保所有这些参数能够被正确初始化。
        
#         """

#         self.ax = None  # 初始化ax
#         self.passTime = 0  # 初始化passTime
#         self.log = None  # 初始化log
#         self.currentGen = 0  # 初始为第0代
#         self.evalsNum = 0  # 初始化评价次数为0
#         self.log = {'gen': [], 'eval': []} if self.logTras != 0 else None  # 初始化log
#         self.timeSlot = time.time()  # 开始计时

#     def logging(self, pop):

#         """
#         描述:
#             用于在进化过程中记录日志。该函数在stat()函数里面被调用。
#             如果需要在日志中记录其他数据，需要在自定义算法模板类中重写该函数。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             无输出参数。

#         """

#         self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
#         if len(self.log['gen']) == 0:  # 初始化log的各个键值
#             if self.problem.ReferObjV is not None:
#                 self.log['gd'] = []
#                 self.log['igd'] = []
#             self.log['hv'] = []
#             self.log['spacing'] = []
#         self.log['gen'].append(self.currentGen)
#         self.log['eval'].append(self.evalsNum)  # 记录评价次数
#         [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # 非支配分层
#         NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
#         if self.problem.ReferObjV is not None:
#             self.log['gd'].append(ea.indicator.GD(NDSet.ObjV, self.problem.ReferObjV))  # 计算GD指标
#             self.log['igd'].append(ea.indicator.IGD(NDSet.ObjV, self.problem.ReferObjV))  # 计算IGD指标
#             self.log['hv'].append(ea.indicator.HV(NDSet.ObjV, self.problem.ReferObjV))  # 计算HV指标
#         else:
#             self.log['hv'].append(ea.indicator.HV(NDSet.ObjV))  # 计算HV指标
#         self.log['spacing'].append(ea.indicator.Spacing(NDSet.ObjV))  # 计算Spacing指标
#         self.timeSlot = time.time()  # 更新时间戳

#     def draw(self, pop, EndFlag=False):

#         """
#         描述:
#             该函数用于在进化过程中进行绘图。该函数在stat()以及finishing函数里面被调用。

#         输入参数:
#             pop     : class <Population> - 种群对象。
            
#             EndFlag : bool - 表示是否是最后一次调用该函数。

#         输出参数:
#             无输出参数。

#         """

#         if not EndFlag:
#             self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算画图的耗时
#             # 绘制动画
#             if self.drawing == 2:
#                 # 绘制目标空间动态图
#                 self.ax = ea.moeaplot(pop.ObjV, 'objective values', False, self.ax, self.currentGen, gridFlag=True)
#             elif self.drawing == 3:
#                 # 绘制决策空间动态图
#                 self.ax = ea.varplot(pop.Phen, 'decision variables', False, self.ax, self.currentGen, gridFlag=False)
#             self.timeSlot = time.time()  # 更新时间戳
#         else:
#             # 绘制最终结果图
#             if self.drawing != 0:
#                 if pop.ObjV.shape[1] == 2 or pop.ObjV.shape[1] == 3:
#                     ea.moeaplot(pop.ObjV, 'Pareto Front', saveFlag=True, gridFlag=True)
#                 else:
#                     ea.moeaplot(pop.ObjV, 'Value Path', saveFlag=True, gridFlag=False)

#     def stat(self, pop):

#         """
#         描述:
#             该函数用于分析当代种群的信息。
#             该函数会在terminated()函数里被调用。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             无输出参数。

#         """

#         feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
#         if len(feasible) > 0:
#             feasiblePop = pop[feasible]  # 获取满足约束条件的个体
#             if self.logTras != 0 and self.currentGen % self.logTras == 0:
#                 self.logging(feasiblePop)  # 记录日志
#                 if self.verbose:
#                     self.display()  # 打印日志
#             self.draw(feasiblePop)  # 展示输出

#     def terminated(self, pop):

#         """
#         描述:
#             该函数用于判断是否应该终止进化，population为传入的种群对象。
#             该函数会在各个具体的算法模板类的run()函数中被调用。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             True / False。
            
#         """

#         self.check(pop)  # 检查种群对象的关键属性是否有误
#         self.stat(pop)  # 进行统计分析，更新进化记录器
#         self.passTime += time.time() - self.timeSlot  # 更新耗时
#         self.timeSlot = time.time()  # 更新时间戳
#         # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
#         if (self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN:
#             return True

#         # ---------终止的条件
#         if not self.currentGen % 2:
#             print(pop.decoding())
#             pop.save()
#             keybord_input = input("敲击回车")

#         else:
#             self.currentGen += 1  # 进化代数+1
#             return False

#     def finishing(self, pop, globalNDSet=None):

#         """
#         描述:
#             进化完成后调用的函数。

#         输入参数:
#             pop : class <Population> - 种群对象。
            
#             globalNDSet : class <Population> - (可选参数)全局存档。

#         输出参数:
#             [NDSet, pop]，其中pop为种群类型；NDSet的类型与pop的一致。

#         """

#         if globalNDSet is None:
#             # 得到非支配种群
#             [levels, _] = ea.ndsortDED(pop.ObjV, needLevel=1, CV=pop.CV, maxormins=self.problem.maxormins)  # 非支配分层
#             NDSet = pop[np.where(levels == 1)[0]]  # 只保留种群中的非支配个体，形成一个非支配种群
#             if NDSet.CV is not None:  # CV不为None说明有设置约束条件
#                 NDSet = NDSet[np.where(np.all(NDSet.CV <= 0, 1))[0]]  # 最后要彻底排除非可行解
#         else:
#             NDSet = globalNDSet
#         if self.logTras != 0 and NDSet.sizes != 0 and (
#                 len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  # 补充记录日志和输出
#             self.logging(NDSet)
#             if self.verbose:
#                 self.display()
#         self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
#         self.draw(NDSet, EndFlag=True)  # 显示最终结果图
#         # 返回帕累托最优个体以及最后一代种群
#         return [NDSet, pop]


# class SoeaAlgorithm(Algorithm):  # 单目标优化算法模板父类

#     """
#     描述:
#         此为单目标进化优化算法模板的父类，所有单目标优化算法模板均继承自该父类。

#     对比于父类该类新增的变量和函数:

#         trappedValue    : int  - 进化算法陷入停滞的判断阈值。

#         maxTrappedCount : int  - “进化停滞”计数器最大上限值。

#         drawing         : int  - 绘图方式的参数，
#                                  0表示不绘图；
#                                  1表示绘制进化过程中种群的平均及最优目标函数值变化图；
#                                  2表示实时绘制目标空间过程动画；
#                                  3表示实时绘制决策空间动态图。
            
#         ----------------- 以下为用户不需要设置的属性 -----------------

#         BestIndi        : class <Population> - 存储算法所找到的最优的个体。

#         trace           : dict - 进化记录器，可以看作是一个内部日志，用于记录每一代种群的一些信息。
#                                  它与算法类的log类似，它有两个键：'f_best'以及'f_avg'。
#                                  'f_best'的键值为一个list列表，存储着每一代种群最优个体的目标函数值；
#                                  'f_avg'的键值为一个list列表，存储着每一代种群所有个体的平均目标函数值。

#         trappedCount    : int  - “进化停滞”计数器。

#         draw()          : 绘图函数。
        
#     """

#     def __init__(self, problem, population):

#         """
#         描述: 
#             在该构造函数里只初始化静态参数以及对动态参数进行定义。
        
#         """

#         super().__init__()  # 先调用父类构造函数
#         self.problem = problem
#         self.population = population
#         self.trappedValue = 0  # 默认设置trappedValue的值为0
#         self.maxTrappedCount = 1000  # 默认设置maxTrappedCount的值为1000
#         self.logTras = 1  # 默认设置logTras的值为1
#         self.verbose = True  # 默认设置verbose的值为True
#         self.drawing = 1  # 默认设置drawing的值为1
#         # 以下为用户不需要设置的属性
#         self.BestIndi = None  # 存储算法所找到的最优的个体
#         self.trace = None  # 进化记录器
#         self.trappedCount = None  # 定义trappedCount，在initialization()才对其进行初始化为0
#         self.ax = None  # 存储动态图像

#     def initialization(self):

#         """
#         描述: 
#             该函数用于在进化前对算法模板的一些动态参数进行初始化操作。
#             该函数需要在执行算法模板的run()函数的一开始被调用，同时开始计时，
#             以确保所有这些参数能够被正确初始化。
        
#         """

#         self.ax = None  # 初始化ax
#         self.passTime = 0  # 初始化passTime
#         self.trappedCount = 0  # 初始化“进化停滞”计数器
#         self.currentGen = 0  # 初始为第0代
#         self.evalsNum = 0  # 初始化评价次数为0
#         self.BestIndi = ea.Population(None, None, 0)  # 初始化BestIndi为空的种群对象
#         self.log = {'gen': [], 'eval': []} if self.logTras != 0 else None  # 初始化log
#         self.trace = {'f_best': [], 'f_avg': []}  # 重置trace
#         # 开始计时
#         self.timeSlot = time.time()

#     def logging(self, pop):

#         """
#         描述:
#             用于在进化过程中记录日志。该函数在stat()函数里面被调用。
#             如果需要在日志中记录其他数据，需要在自定义算法模板类中重写该函数。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             无输出参数。

#         """

#         self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
#         if len(self.log['gen']) == 0:  # 初始化log的各个键值
#             self.log['f_opt'] = []
#             self.log['f_max'] = []
#             self.log['f_avg'] = []
#             self.log['f_min'] = []
#             self.log['f_std'] = []
#         self.log['gen'].append(self.currentGen)
#         self.log['eval'].append(self.evalsNum)  # 记录评价次数
#         self.log['f_opt'].append(self.BestIndi.ObjV[0][0])  # 记录算法所找到的最优个体的目标函数值
#         self.log['f_max'].append(np.max(pop.ObjV))
#         self.log['f_avg'].append(np.mean(pop.ObjV))
#         self.log['f_min'].append(np.min(pop.ObjV))
#         self.log['f_std'].append(np.std(pop.ObjV))
#         self.timeSlot = time.time()  # 更新时间戳

#     def draw(self, pop, EndFlag=False):

#         """
#         描述:
#             该函数用于在进化过程中进行绘图。该函数在stat()以及finishing函数里面被调用。

#         输入参数:
#             pop     : class <Population> - 种群对象。
            
#             EndFlag : bool - 表示是否是最后一次调用该函数。

#         输出参数:
#             无输出参数。

#         """

#         if not EndFlag:
#             self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算画图的耗时
#             # 绘制动画
#             if self.drawing == 2:
#                 metric = np.array(self.trace['f_best']).reshape(-1, 1)
#                 self.ax = ea.soeaplot(metric, Label='Objective Value', saveFlag=False, ax=self.ax, gen=self.currentGen,
#                                       gridFlag=False)  # 绘制动态图
#             elif self.drawing == 3:
#                 self.ax = ea.varplot(pop.Phen, Label='decision variables', saveFlag=False, ax=self.ax,
#                                      gen=self.currentGen, gridFlag=False)
#             self.timeSlot = time.time()  # 更新时间戳
#         else:
#             # 绘制最终结果图
#             if self.drawing != 0:
#                 metric = np.vstack(
#                     [self.trace['f_avg'], self.trace['f_best']]).T
#                 ea.trcplot(metric, [['种群个体平均目标函数值', '种群最优个体目标函数值']], xlabels=[['Number of Generation']],
#                            ylabels=[['Value']], gridFlags=[[False]])

#     def stat(self, pop):

#         """
#         描述:
#             该函数用于分析、记录和打印当代种群的信息。
#             该函数会在terminated()函数里被调用。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             无输出参数。

#         """

#         # 进行进化记录
#         feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
#         if len(feasible) > 0:
#             feasiblePop = pop[feasible]
#             bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]  # 获取最优个体
#             if self.BestIndi.sizes == 0:
#                 self.BestIndi = bestIndi  # 初始化global best individual
#             else:
#                 delta = (
#                                     self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV
#                 # 更新“进化停滞”计数器
#                 self.trappedCount += 1 if np.abs(delta) < self.trappedValue else 0
#                 # 更新global best individual
#                 if delta > 0:
#                     self.BestIndi = bestIndi
#             # 更新trace
#             self.trace['f_best'].append(bestIndi.ObjV[0][0])
#             self.trace['f_avg'].append(np.mean(feasiblePop.ObjV))
#             if self.logTras != 0 and self.currentGen % self.logTras == 0:
#                 self.logging(feasiblePop)  # 记录日志
#                 if self.verbose:
#                     self.display()  # 打印日志
#             self.draw(feasiblePop)  # 展示输出

#     def terminated(self, pop):

#         """
#         描述:
#             该函数用于判断是否应该终止进化，population为传入的种群对象。
#             该函数会在各个具体的算法模板类的run()函数中被调用。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             True / False。

#         """

#         self.check(pop)  # 检查种群对象的关键属性是否有误
#         self.stat(pop)  # 分析记录当代种群的数据
#         self.passTime += time.time() - self.timeSlot  # 更新耗时
#         self.timeSlot = time.time()  # 更新时间戳
#         # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
#         if (
#                 self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN or self.trappedCount >= self.maxTrappedCount:
#             return True
#         else:
#             self.currentGen += 1  # 进化代数+1
#             return False

#     def finishing(self, pop):

#         """
#         描述:
#             进化完成后调用的函数。

#         输入参数:
#             pop : class <Population> - 种群对象。

#         输出参数:
#             [self.BestIndi, pop]，其中pop为种群类型；BestIndi的类型与pop的一致。
        
#         注意:
#             若没有找到可行解，则返回的self.BestIndi为None。
        
#         """

#         feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
#         if len(feasible) > 0:
#             feasiblePop = pop[feasible]
#             if self.logTras != 0 and (len(self.log['gen']) == 0 or self.log['gen'][-1] != self.currentGen):  # 补充记录日志和输出
#                 self.logging(feasiblePop)
#                 if self.verbose:
#                     self.display()
#         self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
#         self.draw(pop, EndFlag=True)  # 显示最终结果图
#         # 返回最优个体以及最后一代种群
#         return [self.BestIndi, pop]