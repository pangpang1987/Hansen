#-*- coding:utf-8 -*-
#指定中文编码

###########################################################################################################################
#
#	igraph 模块
#
###########################################################################################################################

#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
#
#   拓扑检查
#
def topology_check(self) :
    return {
        'sum_edges'             :   len(self.dGraph.es),                        # int, 分支总数
        'sum_virtices'          :   len(self.dGraph.vs),                        # int, 节点总数
        'is_connected'          :   self.is_connected(),
        'sum_connected_graph'   :   len(self.get_connected_graph()),
        'is_dag'                :   self.is_dag(),
        'self_loops'            :   self.get_self_loops(),
        'source_nodes'          :   self.get_virtices_source(),
        'sink_nodes'            :   self.get_virtices_sink(),
        'source_edges'          :   self.get_edges_source(),
        'sink_edges'            :   self.get_edges_sink(),
        'unidirectional_loops'  :   self.get_unidirectional_loops()
    }        
#€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
