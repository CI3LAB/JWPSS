import copy

import numpy as np
from harl.envs.WS_simulator.task import *
# 定义work pacakge的类
# 节点的属性有 1. work content 2.resource  3.duration  4. 前任节点列表  5. 后继节点列表 6. 是否在调度 7. 是否已经完成
class Node(object):
    def __init__(self, idx, task):
        # 自身属性
        self.idx = idx
        self.task_list = []

        self.work_content = 0
        self.work_duration = 0

        # work package scheduling related
        self.resource1 = 0
        self.resource2 = 0
        self.resource3 = 0
        self.resource4 = 0

        # work package related
        self.parent_nodes = []
        self.child_nodes = []

        # the successor and predecessor set of the work package
        self.work_parent_set = []
        self.work_child_set = []

        #This is the set of descendant nodes of the work package
        self.descendant_nodes = []



        # the initial setting of the work package
        if len(self.task_list) == 0:
            self.work_duration = task.task_duration

        self.task_list.append(task)
        self.work_content += task.task_content
        self.resource1 += task.resource1
        self.resource2 += task.resource2
        self.resource3 += task.resource3
        self.resource4 += task.resource4

        self.parent_nodes = list(set(self.parent_nodes).union(set(task.task_parent_nodes)))
        self.child_nodes = list(set(self.child_nodes).union(set(task.task_child_nodes)))


        self.work_child_set = list(set(self.work_child_set).union(set(task.task_child_set)))
        self.work_parent_set = list(set(self.work_parent_set).union(set(task.task_parent_set)))


        # 已经加工完成的父节点，在节点完成后加入
        self.completed_parent_nodes = []


        # 意思是正在加工和已经加工完成的父节点，在assign task之后就加入
        self.unlocked_parent_nodes = []

        # 自身状态 有四种：'running' 'done' 'not_start' 'merged'
        self.condition = 'not_start'

        # 这个工序是否能执行，能为1，不能为0
        self.if_could_run = None

        # 这个参数是为了在检查可执行节点时节省步骤，None为未知，False为不需要检查，True为需要检查
        # 称为需要检查标志
        self.if_need_check = None

        #记录节点相关信息
        self.start_time = None
        self.end_time = None

#this is the sum policy of the resource

    # def node_add_node(self, node,lag_time):
    #     # for the considderation of the lag time,the work package should be reconstructed and the working time and the resource should be updated
    #    for task in node.task_list:
    #        duration_pre = []
    #        duration_succ = []
    #
    #        if len(self.task_list) == 0:
    #            self.work_duration = task.task_duration
    #
    #            self.task_list.append(task)
    #            self.work_content += task.task_content
    #            self.resource1 += task.resource1
    #            self.resource2 += task.resource2
    #            self.resource3 += task.resource3
    #            self.resource4 += task.resource4
    #
    #        else:
    #            for task1 in self.task_list:
    #                if task1.task_idx in task.task_parent_set:
    #                    duration_pre.append(task1.task_duration)
    #                elif task1.task_idx in task.task_child_set:
    #                    duration_succ.append(task1.task_duration)
    #            if sum(duration_pre) + task.task_duration + sum(duration_succ) > self.work_duration:
    #                self.work_duration = sum(duration_pre) + task.task_duration + sum(duration_succ)
    #
    #            self.task_list.append(task)
    #            self.work_content += task.task_content
    #
    #            # Actually, the real resource requirement has a lower-bound and the upper bound, this is the sum policy of the resource; and we also try the peak resource policy.
    #            self.resource1 += task.resource1
    #            self.resource2 += task.resource2
    #            self.resource3 += task.resource3
    #            self.resource4 += task.resource4

# this is the peak policy of the resource, we need a simulator for the time calculation and the resource calculation

    def search_runable_task(self, task_id, task_list_idx, done_task_list,tasks):
        runnable_task = []
        task_temp = None
        for task in tasks:
            if task.task_idx == task_id:
                task_temp = task

        if task_id in task_list_idx and all(item in done_task_list for item in task_temp.task_parent_nodes):
            # print("the first task_id is:",task_id)
            runnable_task.append(task_id)
            return list(set(runnable_task))
        elif task_id in task_list_idx and not all(item in done_task_list for item in task_temp.task_parent_nodes):
            # print("the second task_id is:", task_id)
            return list(set(runnable_task))
        elif task_id not in task_list_idx:
            # print("the third task_id is:", task_id)
            if task_temp and task_temp.task_child_nodes:
                for child in task_temp.task_child_nodes:
                    runnable_task.extend(self.search_runable_task(child, task_list_idx, done_task_list,tasks))
            return list(set(runnable_task))

    def node_add_node(self, node, lag_time, adj_mat_node, initial_runnable_task, tasks, resource_exec):
        temp_task_list = copy.deepcopy(self.task_list)
        self.task_list=temp_task_list+node.task_list

        task_list_idx=[]
        for task in self.task_list:
            task_list_idx.append(task.task_idx)
        done_task_list=[item for item in np.arange(0,len(adj_mat_node),1).tolist() if item not in task_list_idx]

        runnable_task=[]
        for i in initial_runnable_task:
            runnable_task.extend(self.search_runable_task(i, task_list_idx, done_task_list,tasks))

        runnable_task=list(set(runnable_task))

        running_task_dict={}

        resource1_dict={}
        resource2_dict={}
        resource3_dict={}
        resource4_dict={}

        clock=0

        temp_resource1=0
        temp_resource2=0
        temp_resource3=0
        temp_resource4=0

        temp=copy.deepcopy(runnable_task)
        for i in runnable_task:
            for task in self.task_list:
                if task.task_idx == i:
                    running_task_dict.update({i: task.task_duration})
                    temp.remove(i)
                    temp_resource1+=task.resource1
                    temp_resource2+=task.resource2
                    temp_resource3+=task.resource3
                    temp_resource4+=task.resource4

        runnable_task=temp

        resource1_dict.update({clock:temp_resource1})
        resource2_dict.update({clock:temp_resource2})
        resource3_dict.update({clock:temp_resource3})
        resource4_dict.update({clock:temp_resource4})

        while len(done_task_list)!=len(lag_time)-2:
            clock+=1
            # print("clock:",clock)
            # print(list(running_task_dict.keys()))

            for i in list(running_task_dict.keys()):
                running_task_dict[i]-=1
                if running_task_dict[i]==0:
                    done_task_list.append(i)
                    running_task_dict.pop(i)
                    for task in tasks:
                        if task.task_idx==i:
                            for child in task.task_child_nodes:
                                runnable_task.extend(self.search_runable_task(child, task_list_idx, done_task_list,tasks))
                            # print("the child runnable task",runnable_task)
                            temp=copy.deepcopy(runnable_task)
                            for j in runnable_task:
                                for task_child in tasks:
                                    if (task_child.task_idx==j and temp_resource1+task_child.resource1<=resource_exec[0]
                                            and temp_resource2+task_child.resource2<=resource_exec[1]
                                            and temp_resource3+task_child.resource3<=resource_exec[2]
                                            and temp_resource4+task_child.resource4<=resource_exec[3]):
                                        running_task_dict.update({j:task_child.task_duration-lag_time[i+1][j+1]})
                                        temp.remove(j)
                                        # print("the lag_time is ",lag_time[i+1][j+1])

                                        temp_resource1 += task_child.resource1
                                        temp_resource2 += task_child.resource2
                                        temp_resource3 += task_child.resource3
                                        temp_resource4 += task_child.resource4
                                        for t in range(int(clock-lag_time[i+1][j+1]), clock):
                                            resource1_dict.update({t:temp_resource1})
                                            resource2_dict.update({t:temp_resource2})
                                            resource3_dict.update({t:temp_resource3})
                                            resource4_dict.update({t:temp_resource4})
                                    elif task_child.task_idx==j:
                                        running_task_dict.update({j:task_child.task_duration})
                                        temp.remove(j)
                                        temp_resource1 += task_child.resource1
                                        temp_resource2 += task_child.resource2
                                        temp_resource3 += task_child.resource3
                                        temp_resource4 += task_child.resource4

                            runnable_task=temp
                            temp_resource1-=task.resource1
                            temp_resource2-=task.resource2
                            temp_resource3-=task.resource3
                            temp_resource4-=task.resource4

                            resource1_dict.update({clock: temp_resource1})
                            resource2_dict.update({clock: temp_resource2})
                            resource3_dict.update({clock: temp_resource3})
                            resource4_dict.update({clock: temp_resource4})

        self.resource1=max(resource1_dict.values())
        self.resource2=max(resource2_dict.values())
        self.resource3=max(resource3_dict.values())
        self.resource4=max(resource4_dict.values())
        self.work_duration=clock


        for task in node.task_list:
            self.work_content += task.task_content









    def node_clear(self):
        # All the states should be resette and the work package calss is empty
        self.task_list = []
        self.work_content = 0
        self.work_duration = 0
        self.resource1 = 0
        self.resource2 = 0
        self.resource3 = 0
        self.resource4 = 0
        self.parent_nodes = []
        self.child_nodes = []
        self.work_parent_set = []
        self.work_child_set = []
        self.descendant_nodes = []
        self.completed_parent_nodes = []
        self.unlocked_parent_nodes = []
        self.condition = 'merged'
        self.if_could_run = None
        self.if_need_check = None
        self.start_time = None
        self.end_time = None










