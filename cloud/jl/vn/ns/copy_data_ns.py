#-*- coding:utf-8 -*-
#指定中文编码

######################################################################################
#
#   网络解算数据拷贝模块
#
######################################################################################

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   copy_data
#
def copy_data_ns(dataNS) :

    roads = [{**road} for road in dataNS['roads']]
    fanAs = [{**fanA} for fanA in dataNS['fanAs']]
    fanBs = [{**fanB} for fanB in dataNS['fanBs']]
    fanHs = [{**fanH} for fanH in dataNS['fanHs']]
    structRs = [{**structR} for structR in dataNS['structRs']]
    structHs = [{**structH} for structH in dataNS['structHs']]

    return {
        'roads'     :   roads,
        'fanAs'     :   fanAs,
        'fanBs'     :   fanBs,
        'fanHs'     :   fanHs,
        'structRs'  :   structRs,
        'structHs'  :   structHs
    }
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
