# import numpy as np
from harl.envs.WS_simulator.node import *
import copy

class Jobdag(object):
    def __init__(self, graph):
        # 自身信息
        # self.idx:自身索引
        self.idx = None
        self.nodes = []
        self.graph = graph

        #the state of the work package
        self.random_node=self.graph[4]
        self.lag_time=self.graph[3]

        self.state=[]

        #the adaject matrix of the tasks (unchanging
        self.original_task_adj_mat=copy.deepcopy(graph[0])
        self.original_adj_mat = copy.deepcopy(graph[0])
        self.original_feature_mat = copy.deepcopy(graph[1])


        temp = np.delete(self.original_adj_mat, [0, -1], axis=0)
        temp = np.delete(temp, [0, -1], axis=1)




        #this is used for the calcualtion for the work package resource and working time;

        #this is the work package adj matrix(changing when work packaging)
        self.adj_mat = copy.deepcopy(temp)
        self.reversed_adj_mat = copy.deepcopy(self.adj_mat.T)

        # adj_mat有多少列，输入父代和子代节点时需要用到
        self.adj_mat_line_shape = self.adj_mat.shape[0]

        #this is the work package feature matrix(changing when work packging)
        self.feature_mat = copy.deepcopy(np.delete(self.original_feature_mat, [0, -1], axis=0))

        #this is the resource variant of the environment (changing)
        self.resource_exec= copy.deepcopy(graph[2])


        # 状态信息
        self.nodes_all_done = False
        self.num_nodes = None
        self.done_nodes = []

        self.tasks = []
        self.original_resource_exec = copy.deepcopy(graph[2])
        self.adj_mat_node = copy.deepcopy(temp)


        # task is the work tasks, and the node is the work package.
        # with the initial settings, every node contains the corresponding task.

        for idx, info in zip(range(self.feature_mat.shape[0]), self.feature_mat):
            task_duration, content, resource1, resource2, resource3, resource4, _, _ = info
            new_task = Task(idx, content, task_duration, resource1, resource2, resource3, resource4)

            new_task.task_child_nodes = [idx_temp for idx_temp, var in zip(range(self.adj_mat_line_shape), self.adj_mat[new_task.task_idx]) if var == 1 and idx_temp != idx]
            new_task.task_parent_nodes = [idx_temp for idx_temp, var in zip(range(self.adj_mat_line_shape), self.reversed_adj_mat[new_task.task_idx]) if var == 1 and idx_temp != idx]

            new_task.task_parent_set= list(set(self.graph_predecessor(idx))-set([idx]))
            new_task.task_child_set = list(set(self.graph_successor(idx))-set([idx]))

            self.tasks.append(new_task)

            new_node = Node(idx, new_task)
            self.nodes.append(new_node)

        # 统计一下节点总数
        self.num_nodes = len(self.nodes)


    def graph_predecessor(self, node):
        parent_nodes = []
        for i in range(len(self.adj_mat)):
            if self.adj_mat[i][node] == 1:
                parent_nodes.append(i)
                parent_nodes.extend(self.graph_predecessor(i))
        return parent_nodes


    def graph_successor(self, node):
        child_nodes = []
        for i in range(len(self.adj_mat)):
            if self.adj_mat[node][i] == 1:
                child_nodes.append(i)
                child_nodes.extend(self.graph_successor(i))
        return child_nodes

    def node_merge(self, i, j, runable_nodes_idx):
        j=int(j)
        a=copy.deepcopy(self.adj_mat)
        self.nodes[i].parent_nodes=[]
        self.nodes[i].child_nodes=[]
        self.nodes[i].work_parent_set=[]
        self.nodes[i].work_child_set=[]
        for k in range(len(self.adj_mat)):
            if a[j][k]==1:
                self.adj_mat[i][k]=1
            if a[k][j]==1:
                self.adj_mat[k][i]=1
            self.adj_mat[i][i]=0
        self.nodes[i].node_add_node(self.nodes[j],self.lag_time, self.adj_mat_node, runable_nodes_idx,self.tasks,self.original_resource_exec)

        for k in range(len(self.adj_mat[0])):
            self.adj_mat[j][k]=0
            self.adj_mat[k][j]=0


        for node in self.nodes:
            node.parent_nodes=[]
            node.child_nodes=[]
            node.work_parent_set=[]
            node.work_child_set=[]

            for v in range(len(self.adj_mat)):
                if self.adj_mat[node.idx][v] == 1:
                    node.child_nodes.append(v)
                if self.adj_mat[v][node.idx] == 1:
                    node.parent_nodes.append(v)
            node.work_parent_set=list(set(self.graph_predecessor(node.idx))-set([node.idx]))
            node.work_child_set=list(set(self.graph_successor(node.idx))-set([node.idx]))

        self.nodes[j].node_clear()


        self.feature_mat[i][0]=self.nodes[i].work_duration
        self.feature_mat[i][1]=self.nodes[i].work_content
        self.feature_mat[i][2]=self.nodes[i].resource1
        self.feature_mat[i][3]=self.nodes[i].resource2
        self.feature_mat[i][4]=self.nodes[i].resource3
        self.feature_mat[i][5]=self.nodes[i].resource4
        self.feature_mat[i][6]=0
        self.feature_mat[i][7]=0


    def get_state(self):
        state=[]
        state.append(self.adj_mat)
        #task and the work package relationship
        task_distribution=np.zeros((self.adj_mat_line_shape,self.adj_mat_line_shape))
        for node in self.nodes:
            for task in node.task_list:
                task_distribution[node.idx][task.task_idx]=1
        #work content, resource and duration
        task_information=np.zeros((self.adj_mat_line_shape,self.adj_mat_line_shape))
        for node in self.nodes:
                task_information[node.idx][node.idx]=node.work_content/25         #normalization for the state for the convergence of the network
                task_information[node.idx][(node.idx+1)%self.adj_mat_line_shape]=node.resource1/40
                task_information[node.idx][(node.idx+2)%self.adj_mat_line_shape]=node.resource2/40
                task_information[node.idx][(node.idx+3)%self.adj_mat_line_shape]=node.resource3/40
                task_information[node.idx][(node.idx+4)%self.adj_mat_line_shape]=node.resource4/40
                task_information[node.idx][(node.idx+5)%self.adj_mat_line_shape]=node.work_duration/40
        state.append(task_distribution)
        state.append(task_information)


        #work package constraint
        active_nodes=list(set(list(range(0,self.adj_mat_line_shape)))-set(self.random_node))
        work_packaging_constrain=np.zeros((self.adj_mat_line_shape,self.adj_mat_line_shape))

        # print("active_nodes:", active_nodes)

        for node in self.nodes:
            # print("node:",node.idx,"node_condition:",node.condition)
            # # print(node.idx)
            # print(node.parent_nodes)
            # print(node.child_nodes)
            # print(node.work_parent_set)
            # print(node.work_child_set)
            if node.idx not in self.random_node:
                if node.condition=='not_start':
                    pre_temp=[]
                    suc_temp=[]
                    for i in node.work_parent_set:
                        pre_temp=list(set(pre_temp).union(set(self.graph_predecessor(i))-set([i])))
                    # pre_temp=node.work_parent_set

                    for i in node.work_child_set:
                        suc_temp=list(set(suc_temp).union(set(self.graph_successor(i))-set([i])))

                    # print("node.work_child",node.work_child_set)
                    # print("node.work_parent",node.work_parent_set)
                    # print("pre_temp:",pre_temp)
                    # print("suc_temp:",suc_temp)

                    total_node=list(set(pre_temp).union(set(suc_temp)))
                    total_node=list(set(active_nodes)-set(total_node)-set([node.idx]))


                    for i in total_node:
                        if self.nodes[i].condition=='not_start' and i not in self.random_node:
                            work_packaging_constrain[node.idx][i]=1
                        # print("work_packaging_constrain:",work_packaging_constrain[i])
            else:
                pass

        state.append(work_packaging_constrain)
        #the shape of the state is (5,30,30), when the adj_mat is modified, then the state can use this function to update the
        return state
    def get_obs(self):
        self.obs = np.zeros((self.adj_mat_line_shape,len(self.adj_mat)+8)).tolist()

        for i in range(len(self.adj_mat)):
            if self.nodes[i].condition=='not_start':
                for task in self.nodes[i].task_list:
                    self.obs[i][task.task_idx]=1
                self.obs[i][(len(self.adj_mat)+1)]=self.nodes[i].work_content/25
                self.obs[i][(len(self.adj_mat)+2)]=self.nodes[i].work_duration/40
                self.obs[i][(len(self.adj_mat)+3)]=self.nodes[i].resource1/40
                self.obs[i][(len(self.adj_mat)+4)]=self.nodes[i].resource2/40
                self.obs[i][(len(self.adj_mat)+5)]=self.nodes[i].resource3/40
                self.obs[i][(len(self.adj_mat)+6)]=self.nodes[i].resource4/40
                if self.nodes[i].condition=='not_start' or self.nodes[i].condition=='merged' or self.nodes[i].condition=='done':
                    self.obs[i][(len(self.adj_mat)+7)]=0
                elif self.nodes[i].condition=='running':
                    self.obs[i][(len(self.adj_mat)+7)]=1
        return self.obs



