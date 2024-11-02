import copy
from harl.envs.WS_simulator.jobdag import Jobdag
from harl.envs.WS_simulator.normal_executor import Normal_executor
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Normal_environment(object):
    def __init__(self,env_args,data_project):
        # the parameters of the environment

        self.step=0
        self.args = env_args
        # self.path = env_args["path"]
        # self.lag_time = env_args["lag_time"]

        # the parameeters for our objective function
        self.episode_limit = 200
        self.work_package_cost = 50
        self.a1=0.8
        self.a2=0.8
        self.a3=1.2
        self.cash=0.001
        self.cash1=100

        self.global_reward=0
        self.episode_reward=0
        self.global_objection=0

        # the data structure of the project used for the environment initialization

        # self.graph=Graph(self.path,env_args["project_index"])
        self.graph=data_project


        self.executor = None

        # 邻接矩阵的储存格式：idx0:jobdag编号 idx1:邻接矩阵
        self.adj_mat = None
        self.last_decision_time = 0

        # this time, we do not consieder the baseline schedule and gives the env information with the experiments scale
        self.state_shape=(5,len(data_project[0])-2,len(data_project[0])-2)
        self.obs_shape=len(data_project[0])+10
        self.n_agent=len(data_project[0])-2



        self.scheduling_reward=[]
        self.work_packaging_reward=[]




    def reset(self):
        # 1.重置env的状态, 也就是重新装载所有数据， 所以需要输入任务信息
        jobdag = Jobdag(self.graph)
        executor = Normal_executor(jobdag)
        self.executor = executor
        self.last_decision_time = 0
        self.global_objection=0
        self.global_reward=0
        self.episode_reward=0
        self.adj_mat = self.executor.jobdag.adj_mat
        # 2.返回初始state
        state = self.executor.get_state()
        observation = self.executor.get_obs()
        available_actions=None
        return observation,state,available_actions

    def get_env_info(self):
        env_info = {
            "state_shape": self.state_shape,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_agent+2,
            "n_agents": self.n_agent,
            "episode_limit": self.episode_limit,
        }
        return env_info


    def env_close(self):
        self.executor = None
        self.adj_mat = None
        self.last_decision_time = 0
        self.step=0

    def seed(self, seed):
        pass

    def render(self):
        pass

    def step_agent(self,AgentID,action):
        r=0
        if action < len(self.executor.jobdag.nodes):
            # print("the reward of work packaging is: ",self.work_package_cost)
            self.executor.work_packaging(AgentID, action)
            # self.global_reward += self.work_package_cost/50
            self.global_reward += 0
            # print("the reward of work packaging is: ",self.work_package_cost/50)


        elif action == self.executor.jobdag.num_nodes:
            self.executor.assign_task_ws(AgentID)
            r_temp = 0

            r_temp += ((self.executor.jobdag.nodes[AgentID].work_content) ** self.a1 + \
                       (self.executor.jobdag.nodes[AgentID].work_content) ** self.a2 + \
                       (self.executor.jobdag.nodes[AgentID].work_content) ** self.a3)

            r_temp += self.cash1 * self.executor.jobdag.nodes[AgentID].work_content * (
                    1 - math.exp(-self.cash * (self.executor.walltime+self.executor.jobdag.nodes[AgentID].work_duration)))

            r_temp += self.work_package_cost


            self.global_reward -= r_temp/50
        else:
            if AgentID in self.executor.runable_nodes_idx_ws:
                self.executor.Doing_nothing(AgentID)
            else:
                pass

    def step1(self,actions):
        # print("执行动作")
        actions = np.array(actions)
        self.step+=1
        reward=copy.deepcopy(self.global_reward)
        self.global_reward=0
        done=False

        left_node_notstart = 0
        left_node_running = 0

        for node in self.executor.jobdag.nodes:
            if node.condition == 'not_start':
                left_node_notstart += 1
            if node.condition == 'running':
                left_node_running += 1

        if left_node_notstart == 0 and left_node_running !=0:
            # print("第一种情况推进")
            self.last_decision_time = self.executor.walltime
            while (left_node_running>0):
                self.executor.advance_time_ws()
                left_node_running = 0
                for node in self.executor.jobdag.nodes:
                    if node.condition == 'running':
                        left_node_running += 1
            done=True
            reward +=(self.last_decision_time - self.executor.walltime)/20

            self.episode_reward+=reward

            state=self.executor.get_state()
            observation=self.executor.get_obs()
            available_actions=None
            info=None
            obj=self.episode_obj()
            individual=self.episode_individual()
            return reward,done,state,observation,available_actions,info,obj,individual
        else:
            flag=0
            while (len(self.executor.running_work_packages)+len(self.executor.nodes_to_schedule)==len(self.executor.runable_nodes_idx_ws) and flag==0):
                if len(self.executor.nodes_to_schedule) == 0:
                    self.last_decision_time = self.executor.walltime
                    self.executor.advance_time_ws()
                    reward +=(self.last_decision_time - self.executor.walltime)/20
                else:
                    for i in range(len(self.executor.nodes_to_schedule)):
                        if (self.executor.jobdag.nodes[self.executor.nodes_to_schedule[i]].resource1 <= self.executor.resource_exec[0] and
                           self.executor.jobdag.nodes[self.executor.nodes_to_schedule[i]].resource2 <= self.executor.resource_exec[1] and
                            self.executor.jobdag.nodes[self.executor.nodes_to_schedule[i]].resource3 <= self.executor.resource_exec[2] and
                            self.executor.jobdag.nodes[self.executor.nodes_to_schedule[i]].resource4 <= self.executor.resource_exec[3]):
                            flag = 1

                    self.last_decision_time = self.executor.walltime



                    if(flag==0):
                        self.executor.advance_time_ws()

                    reward +=(self.last_decision_time - self.executor.walltime)/20

            self.episode_reward+=reward
            state=self.executor.get_state()
            observation=self.executor.get_obs()
            available_actions=None
            info=None
            obj=self.episode_obj()
            individual=self.episode_individual()
            return reward,done,state,observation,available_actions,info,obj,individual

    def episode_obj(self):
        obj=0
        if len(self.executor.done_tasks)==0:
            return 0
        else:
            obj+=len(self.executor.done_tasks)*self.work_package_cost
            for i in self.executor.done_tasks:
                obj+=((self.executor.jobdag.nodes[i].work_content) ** self.a1 + \
                      (self.executor.jobdag.nodes[i].work_content) ** self.a2 + \
                      (self.executor.jobdag.nodes[i].work_content) ** self.a3)

                obj+=self.cash1 * self.executor.jobdag.nodes[i].work_content * (
                        1 - math.exp(-self.cash * self.executor.jobdag.nodes[i].end_time))

            obj=0.5*obj
            # print("the makespan of the individual is:",self.executor.jobdag.nodes[self.executor.done_tasks[-1]].end_time)
            obj+=0.5*self.executor.jobdag.nodes[self.executor.done_tasks[-1]].end_time
            return obj


    def episode_individual(self):
        individual=[]
        wp=[]
        task_temp=[]
        resource1=[]
        resource2=[]
        resource3=[]
        resource4=[]
        start_time=[]
        end_time=[]
        if len(self.executor.done_tasks)==0:
            return individual
        else:
            for i in self.executor.done_tasks:
                task=[]
                wp.append(i)
                for j in range(len(self.executor.jobdag.nodes[i].task_list)):
                    task.append(self.executor.jobdag.nodes[i].task_list[j].task_idx)
                task_temp.append(task)
                resource1.append(self.executor.jobdag.nodes[i].resource1)
                resource2.append(self.executor.jobdag.nodes[i].resource2)
                resource3.append(self.executor.jobdag.nodes[i].resource3)
                resource4.append(self.executor.jobdag.nodes[i].resource4)
                start_time.append(self.executor.jobdag.nodes[i].start_time)
                end_time.append(self.executor.jobdag.nodes[i].end_time)
            individual.append(wp)
            # print("wp",wp)
            individual.append(task_temp)
            # print("task_temp",task_temp)
            individual.append(resource1)
            # print("resource1",resource1)
            individual.append(resource2)
            individual.append(resource3)
            individual.append(resource4)
            individual.append(start_time)
            # print("start_time",start_time)
            individual.append(end_time)
            # print("end_time",end_time)
            return individual


    # def get_available_actions(self,AgentID):
    #     return self.executor.get_avail_action_agent(AgentID)

    def get_map(self):
        self.G = nx.DiGraph()
        self.node = []
        self.edge = []
        self.random_node=self.executor.jobdag.random_node



        for i in self.executor.jobdag.nodes:
            self.node.append('' + str((i.idx + 1)))

        for i in range(len(self.executor.jobdag.adj_mat)):
            for j in range(len(self.executor.jobdag.adj_mat[i])):
                if self.executor.jobdag.adj_mat[i][j] == 1:
                    self.edge.append(('' + str(i+1), '' + str(j+1)))

        self.node_map=['yellow' for i in range(len(self.executor.jobdag.adj_mat))]

        for i in self.random_node:
            self.node_map[i]='red'

        for i in self.executor.runable_nodes_idx_ws:
            self.node_map[i]='green'

        for i in self.executor.done_tasks:
            self.node_map[i]='blue'

        self.G.add_nodes_from(self.node)
        self.G.add_edges_from(self.edge)


        pos1 = nx.spectral_layout(self.G)
        pos2 = nx.spring_layout(self.G)
        nx.draw(self.G, pos2, node_color=self.node_map, with_labels=True, node_size=1000, width=3,  edge_color='b')
        plt.title('有向图')
        plt.xticks([])
        plt.yticks([])
        plt.show()




























