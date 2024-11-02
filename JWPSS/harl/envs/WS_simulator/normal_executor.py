from harl.envs.WS_simulator.task import Task
import copy
import numpy as np
from harl.envs.WS_simulator.Task_Ongoing import Node_Ongoing


class Normal_executor(object):
    def __init__(self, jobdag):
        self.jobdag = jobdag

        # 原始资源矩阵，不会变
        self.original_resource_exec = copy.deepcopy(jobdag.resource_exec)

        # 资源矩阵, 和jobdag的资源矩阵是同一地址，所以不用deepcopy
        self.resource_exec = jobdag.resource_exec

        # 时间信息
        self.walltime = 0
        # all the tasks that can be executed at a time step t
        self.runable_nodes_idx = self.ini_runable_nodes_idx()
        self.runable_nodes_idx_ws = self.ini_runable_nodes_idx()

        #work packaging is located before all the tasks
        self.node_work_packaging=self.ini_runable_nodes_idx()

        # the nodes that need to be scheduled
        self.nodes_to_schedule = []

        # the work packages that have been scheduled
        self.running_tasks = []
        self.running_work_packages=[]

        #the noded that have been merged
        self.merged_nodes=[]

        # 已经执行完成的work_package的列表
        self.done_tasks = []

        # 节点的特征矩阵，同样是“一维数组”
        self.feature_mat = jobdag.feature_mat

        # 已经完成的节点的数量
        self.complete_node = 0

        self.action_sequence = []

        # 需要先初始化这两个
        mask = np.zeros(self.jobdag.num_nodes)
        mask[self.runable_nodes_idx] = 1
        self.now_action_mask = mask


    def ini_runable_nodes_idx(self):
        index=[0 for i in range(len(self.jobdag.adj_mat))]
        for i in range(len(self.jobdag.adj_mat)):
            for j in range(len(self.jobdag.adj_mat[i])):
                if self.jobdag.adj_mat[j][i] == 1:
                    index[i] = 1

        idx = np.where(np.array(index) == 0)[0].tolist()
        return idx

    def assign_task(self, node_idx):
        assert node_idx in self.runable_nodes_idx, 'can not assign input node idx'
        self.action_sequence.append(node_idx)

        # 先把这个节点从可执行节点移除，注意
        self.runable_nodes_idx.remove(node_idx)

        node = self.jobdag.nodes[node_idx]

        node_going = Node_Ongoing(node_idx, node, self.walltime)
        self.running_tasks.append(node_going)
        # 改变节点状态
        node.condition = 'running'
        # 改变特征矩阵，把矩阵对应数设为1（代表正在执行）
        self.feature_mat[node_idx][6] = 1
        # 减去资源
        self.resource_exec[0] -= node.resource1
        self.resource_exec[1] -= node.resource2
        self.resource_exec[2] -= node.resource3
        self.resource_exec[3] -= node.resource4
    # 推进时间函数
    def advance_time(self):
        # 推进时间
        self.walltime += 1

        # 不能直接移除！！！会导致Bug，还不会报错。为了找这个bug我用了一晚上！！！ 加油兄弟，这个几把东西写了两周

        wait_remove_task = []
        ####
        for task in self.running_tasks:
            # 同时减去特征矩阵中的对应的时间 ###
            self.feature_mat[task.node_idx][0] -= 1
            # task时间记录加一
            # task.have_run_time += 1
            # # 如果该task已完成，进行下面操作

            if self.feature_mat[task.node_idx][0] == 0:
                wait_remove_task.append(task)

        for task in wait_remove_task:
            # 解锁新节点的操作应该在assign task时就操作，而不是等到有节点完成时操作
            # 后来多了个新的列表储存已经完成的父节点
            for child_node in self.jobdag.nodes[task.node_idx].child_nodes:
                self.jobdag.nodes[child_node].completed_parent_nodes.append(task.node_idx)

            # 移除操作

            task.finish_time = self.walltime
            task.during_time = task.finish_time - task.start_time

            self.running_tasks.remove(task)
            self.done_tasks.append(task)
            # 设置对应节点的condition
            # task.node.condition = 'done'
            # 返还资源
            self.resource_exec[0] += task.node.resource1
            self.resource_exec[1] += task.node.resource2
            self.resource_exec[2] += task.node.resource3
            self.resource_exec[3] += task.node.resource4
            # 将特征矩阵one hot编码设置为01(代表已完成)
            self.feature_mat[task.node_idx][6] = 0
            self.feature_mat[task.node_idx][7] = 1
            # 已完成的节点数+1
            self.complete_node += 1

    # the action for our joint work package and scheduling problem
    def assign_task_ws(self, node_idx):
            assert node_idx in self.runable_nodes_idx_ws, 'can not assign input node idx'
            self.action_sequence.append(node_idx)
            self.running_work_packages.append(node_idx)

            # 先把这个节点从可执行节点移除，注意
            # self.runable_nodes_idx_ws.remove(node_idx)
            if node_idx in self.node_work_packaging:
                self.node_work_packaging.remove(node_idx)
            else:
                self.nodes_to_schedule.remove(node_idx)

            node = self.jobdag.nodes[node_idx]
            node.start_time = self.walltime
            # 生成node_going
            node_going = Node_Ongoing(node_idx, node, self.walltime)
            self.running_tasks.append(node_going)
            # 改变节点状态
            node.condition = 'running'
            # 改变特征矩阵，把矩阵对应数设为1（代表正在执行）
            self.feature_mat[node_idx][6] = 1
            # 减去资源
            self.resource_exec[0] -= node.resource1
            self.resource_exec[1] -= node.resource2
            self.resource_exec[2] -= node.resource3
            self.resource_exec[3] -= node.resource4

    def work_packaging(self, i, j):
        self.merged_nodes.append(j)
        self.jobdag.node_merge(i, j, self.runable_nodes_idx)

        if j in self.runable_nodes_idx_ws:
            self.runable_nodes_idx_ws.remove(j)
        if j in self.node_work_packaging:
            self.node_work_packaging.remove(j)
        if j in self.nodes_to_schedule:
            self.nodes_to_schedule.remove(j)

        # print("agentid",i,"node",j,"work packaging is done")
        # print("agentid",i,self.jobdag.nodes[i].parent_nodes,self.jobdag.nodes[i].completed_parent_nodes)

        # temp_adj_mat = copy.deepcopy(self.jobdag.adj_mat)
        self.jobdag.nodes[i].completed_parent_nodes=[]
        for node_id in self.jobdag.nodes[i].parent_nodes:
            if self.jobdag.nodes[node_id].condition == 'done' or self.jobdag.nodes[node_id].condition == 'merged':
                self.jobdag.nodes[i].completed_parent_nodes.append(node_id)

        # self.jobdag.nodes[i].completed_parent_nodes = list(set(self.jobdag.nodes[i].completed_parent_nodes))

        if (i in self.runable_nodes_idx_ws and self.jobdag.nodes[i].parent_nodes != self.jobdag.nodes[
            i].completed_parent_nodes):
            self.runable_nodes_idx_ws.remove(i)
            self.node_work_packaging.remove(i)

    def advance_time_ws(self):

        # print("推进时间步")
        # 推进时间
        self.walltime += 1

        # 不能直接移除！！！这个几把东西写了两周
        wait_remove_task = []
        ####
        for task in self.running_work_packages:
            self.feature_mat[task][0] -= 1
            if self.feature_mat[task][0] == 0:
                wait_remove_task.append(task)

        for task in wait_remove_task:
            for child_node in self.jobdag.nodes[task].child_nodes:
                self.jobdag.nodes[child_node].completed_parent_nodes.append(task)

            # 移除操作
            self.runable_nodes_idx_ws.remove(task)
            self.running_work_packages.remove(task)

            for i in self.jobdag.nodes[task].child_nodes:
                if set(self.jobdag.nodes[i].parent_nodes)==set(self.jobdag.nodes[i].completed_parent_nodes):
                    self.node_work_packaging.append(i)
                    self.runable_nodes_idx_ws.append(i)


            self.jobdag.nodes[task].end_time = self.walltime

            self.done_tasks.append(task)
            self.jobdag.nodes[task].condition = 'done'
            # 返还资源
            self.resource_exec[0] += self.jobdag.nodes[task].resource1
            self.resource_exec[1] += self.jobdag.nodes[task].resource2
            self.resource_exec[2] += self.jobdag.nodes[task].resource3
            self.resource_exec[3] += self.jobdag.nodes[task].resource4
            # 将特征矩阵one hot编码设置为01(代表已完成)
            self.feature_mat[self.jobdag.nodes[task].idx][6] = 0
            self.feature_mat[self.jobdag.nodes[task].idx][7] = 1
            # 已完成的节点数+1
            self.complete_node = len(self.merged_nodes)+len(self.done_tasks)

    def get_state(self):
        state=self.jobdag.get_state()
        # print("resource_exec:",self.resource_exec)

        resource_constrain=np.zeros((self.jobdag.adj_mat_line_shape,self.jobdag.adj_mat_line_shape))
        for node in self.jobdag.nodes:
            if node.condition=='not_start':
                if node.resource1 <= self.resource_exec[0] \
                    and node.resource2 <= self.resource_exec[1] \
                    and node.resource3 <= self.resource_exec[2] \
                    and node.resource4 <= self.resource_exec[3]:
                    resource_constrain[node.idx][node.idx]=1
        state.append(resource_constrain)
        return state

    def get_obs(self):
        obs=self.jobdag.get_obs()
        for i in obs:
            i.append(self.resource_exec[0])
            i.append(self.resource_exec[1])
            i.append(self.resource_exec[2])
            i.append(self.resource_exec[3])
        return obs
    # work packaging

    def Doing_nothing(self,agent_id):
        if agent_id in self.node_work_packaging:
            self.node_work_packaging.remove(agent_id)
            self.nodes_to_schedule.append(agent_id)
        else:
            pass

    def get_avail_action_agent(self,agent_id):
        action_legal_list=np.zeros(self.jobdag.adj_mat_line_shape+2).tolist()

        if agent_id in self.runable_nodes_idx_ws:
            # print("agentid",agent_id,"in runable_nodes_idx_ws")
            # print("node_parent",self.jobdag.nodes[agent_id].work_parent_set)
            # print("node_child",self.jobdag.nodes[agent_id].work_child_set)
            if agent_id in self.node_work_packaging:
                state = self.get_state()[3][agent_id]
                # print("agentid", agent_id, "constraint",state)
                for i in range(len(state)):
                    if (state[i] == 1 \
                            and self.jobdag.nodes[i].resource1 + self.jobdag.nodes[agent_id].resource1 <=
                            self.original_resource_exec[0] \
                            and self.jobdag.nodes[i].resource2 + self.jobdag.nodes[agent_id].resource2 <=
                            self.original_resource_exec[1] \
                            and self.jobdag.nodes[i].resource3 + self.jobdag.nodes[agent_id].resource3 <=
                            self.original_resource_exec[2] \
                            and self.jobdag.nodes[i].resource4 + self.jobdag.nodes[agent_id].resource4 <=
                            self.original_resource_exec[3]\
                            and i not in self.running_work_packages):
                            action_legal_list[i] = 1

                flag = 0
                for i in range(len(self.jobdag.nodes)):
                    if self.jobdag.adj_mat[i][agent_id]==1:
                        if self.jobdag.nodes[i].condition!='done':
                            print("work_packaging发生错误，节点",agent_id,"的前置节点",i,"未完成")
                            flag=1
                            break

                if (self.jobdag.nodes[agent_id].resource1 >
                        self.resource_exec[0] \
                        or self.jobdag.nodes[agent_id].resource2 >
                        self.resource_exec[1] \
                        or self.jobdag.nodes[agent_id].resource3 >
                        self.resource_exec[2] \
                        or self.jobdag.nodes[agent_id].resource4 >
                        self.resource_exec[3]):
                    flag = 1

                if flag==1:
                    action_legal_list[-2]=0
                    action_legal_list[-1]=1
                else:
                    action_legal_list[-2]=1
                    action_legal_list[-1]=0

            elif agent_id in self.nodes_to_schedule:
                flag = 0
                for i in range(len(self.jobdag.nodes)):
                    if self.jobdag.adj_mat[i][agent_id] == 1:
                        if self.jobdag.nodes[i].condition != 'done':
                            print("to_schedule发生错误，节点",agent_id,"的前置节点",i,"未完成")
                            flag = 1
                            break

                if (self.jobdag.nodes[agent_id].resource1 >
                        self.resource_exec[0] \
                        or self.jobdag.nodes[agent_id].resource2 >
                        self.resource_exec[1] \
                        or self.jobdag.nodes[agent_id].resource3 >
                        self.resource_exec[2] \
                        or self.jobdag.nodes[agent_id].resource4 >
                        self.resource_exec[3]):
                    flag = 1

                if flag == 1:
                    action_legal_list[-2] = 0
                    action_legal_list[-1] = 1
                else:
                    action_legal_list[-2] = 1
                    action_legal_list[-1] = 0
            else:
                action_legal_list[-1] = 1
        else:
            action_legal_list[-1] = 1

        return action_legal_list





