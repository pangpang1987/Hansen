#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#   通风阻力系数反演V1.0模块 
#
###########################################################################################################################

#   要点：
#   1. 运行程序
#   （1）在MyProblem定义中导入GA
#   （2）GA运行适应值函数AimFunc调用此函数, 并把染色体种群及有关数据传入此函数
#   2. 适应函数功能
#   （1）创建种群数据, 为了减少IO数据量, 加快迭代速度, 只读数据进行拷贝
#   （2）权重计算, 找回路等创建网络解算迭代数据的任务在个体适应值计算中完成, 这是本次的最大修改。


#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   适应值函数
#
import  numpy as np
import  multiprocessing as mp
from    multiprocessing import Pool as ProcessPool
from    multiprocessing.dummy import Pool as ThreadPool
import  os
import  json
from    .FuncFitness                import  FuncFinness
# poolType = "Process"
num         = 0

from    pprint  import  pprint


def _persist_iteration_outputs(fitness_matrix, pop, data, iteration_index):
    """
    将单代误差和区间最佳个例写入results文件夹。
    无论自适应模式还是非自适应模式，error文件都写入results文件夹。
    """
    if fitness_matrix is None or pop is None:
        return

    # 获取迭代结果目录，如果未设置则使用默认路径
    iteration_dir = data.get("iterationResultsDir")
    if not iteration_dir:
        # 如果没有指定目录，尝试使用 results 文件夹
        results_base = os.path.join(os.getcwd(), "results")
        # 尝试查找最新的 Round 目录
        if os.path.exists(results_base):
            round_dirs = []
            for item in os.listdir(results_base):
                round_path = os.path.join(results_base, item)
                if os.path.isdir(round_path) and item.startswith('Round-'):
                    try:
                        round_num = int(item.split('-')[1])
                        round_dirs.append((round_num, round_path))
                    except (ValueError, IndexError):
                        continue
            if round_dirs:
                # 使用最新的 Round 目录
                round_dirs.sort(key=lambda x: x[0], reverse=True)
                iteration_dir = os.path.join(round_dirs[0][1], 'iterations')
            else:
                # 如果没有 Round 目录，使用默认路径
                iteration_dir = os.path.join(results_base, 'iterations')
        else:
            # 如果 results 文件夹不存在，创建默认路径
            iteration_dir = os.path.join(results_base, 'iterations')
    
    # 确保目录存在
    os.makedirs(iteration_dir, exist_ok=True)

    # 写入 error 文件到 results 文件夹
    error_path = os.path.join(iteration_dir, f"error{iteration_index}.txt")
    np.savetxt(error_path, fitness_matrix)

    save_interval = data.get("bestSaveInterval") or 0
    if save_interval <= 0 or iteration_index % save_interval != 0:
        return

    chromosome_ids = data.get("chromosomeIds") or []
    flattened = fitness_matrix.reshape(-1)
    if flattened.size == 0:
        return
    best_idx = int(np.argmin(flattened))
    if best_idx >= pop.Phen.shape[0]:
        return
    best_vector = pop.Phen[best_idx]
    variables = {}
    if chromosome_ids and len(chromosome_ids) == len(best_vector):
        variables = {cid: float(val) for cid, val in zip(chromosome_ids, best_vector)}
    else:
        variables = [float(val) for val in best_vector]

    best_payload = {
        "iteration": iteration_index,
        "fitness": float(flattened[best_idx]),
        "variables": variables
    }

    best_path = os.path.join(iteration_dir, f"best_iter_{iteration_index}.json")
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, ensure_ascii=False, indent=2)

def CustomAimFuct(
    pop,            # 系统生成的染色体种群, AimFunc(self,pop)函数导入
    # pop.Phen  # 个体矩阵，每个个体有Dim个决策变量, 每一代的群不种群
    # 10个风路, 种群数=2, Vars=[[r1,r2,...r10],[r1,r2,...,r10]]
    fitnesss,       # 种群适应值,以后从系统导出
    data,
    poolType
):

    # 1. 染色体pop.Phen解码为风阻系数, 创建种群数据
    dataPop = [(rhs,data) for rhs in pop.Phen]   # 种群数据

    # ----------本地显示
    global num
    num += 1    
    print("进化第   %d  代 : " % num) 

    # 本地，多进程、多线程
    num_cores = int(mp.cpu_count())  # 获得计算机的核心数，用于并行计算
    if poolType == 'Process':  # 多进程
        pool = ProcessPool(num_cores)  # 设置计算核数
    if poolType == 'Thread':  # 多线程
        pool = ThreadPool(len(pop.Phen))

    # 2.1 多进程
    if poolType == "Process" :                             # 多进程
        error = pool.map_async(FuncFinness, dataPop)     # 一代迭代迭代误差计算
                                            # error----适应值对象, 对象不能打印
        error.wait()                        # 等待多进程结果
        fitness = np.vstack(error.get())    # 一代适应值数组, 与多线程返回数据格式不一样
        pool.close()
        pool.join()
        pop.ObjV = np.hstack([fitness])     # 适应值加入种群对象
        #   数据格式
        # error.get()   ----    [array(53.5938609), array(52.26123295), array(50.05459613)]
        # fitness       ----    [[53.5938609 ]
        #                           [52.26123295]
        #                           [50.05459613]]
        #   np.hstack   ----    [[53.5938609 ]
        #                           [52.26123295]
        #                           [50.05459613]]
    
        fitnesss.append(fitness)
        _persist_iteration_outputs(fitness, pop, data, num)


    # 2.2 多线程
    if poolType == "Thread" :                      # 多线程
        error = pool.map(FuncFinness, dataPop)    # 一代迭代误差计算，error与多进程不同
        fitness = np.vstack(np.array(error))
        pool.close()
        pool.join()
        pop.ObjV = fitness
        #   数据格式
        # error     ----    [array(54.26147918), array(53.75947329), array(55.13111155)]
        # fitness   ----    [[53.27571877]
        #                       [53.66187221]
        #                       [53.50847852]]
        # np.vstack ----    [[54.77536941]
        #                       [55.0899665 ]
        #                       [51.88840207]]
        fitnesss.append(fitness)
        _persist_iteration_outputs(fitness, pop, data, num)

    # 2.3 串行
    # if problem.poolType == "Serial":   # 串行计算
    #     f = problem.Serial(data)       # 适应值
    #     pop.ObjV = f
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€


