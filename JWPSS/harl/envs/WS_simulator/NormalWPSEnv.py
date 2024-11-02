import copy
import numpy as np
from harl.envs.WS_simulator.normal_env import Normal_environment
class WPSEnv:
    def __init__(self, env_args,index):
        self.args = copy.deepcopy(env_args)
        self.path= self.args["path"]
        self.data = np.load(self.path, allow_pickle=True)
        data_project = self.data[index]

        self.envs = Normal_environment(self.args, data_project)

        self.n_agents = self.envs.get_env_info()["n_agents"]
        self.share_observation_space = self.repeat(np.array(self.envs.get_env_info()["state_shape"]).tolist())
        self.observation_space = self.repeat([self.envs.get_env_info()["obs_shape"]])
        self.action_space = self.repeat([self.envs.get_env_info()["n_actions"]])

    def step(self, actions):
        rewards,done,state,obs,available_actions,info,obj,individual=self.envs.step1(actions)
        state = self.repeat(state)
        rewards = [[rewards]] * self.n_agents
        dones = [done] * self.n_agents

        info={}

        if done:
            if self.envs.step > self.envs.episode_limit:
                info["bad_transition"] = True
            else:
                info["bad_transition"] = False
        else:
            info["bad_transition"] = False
        infos = [info] * self.n_agents
        return obs, state, rewards, dones, infos, available_actions,obj,individual

    def step_agent(self, agent_id, actions):
        self.envs.step_agent(agent_id, actions)

    def get_available_actions(self,AgentID):
        return self.envs.get_available_actions(AgentID)


    def episode_obj(self):
        return self.envs.episode_obj()

    def reset(self):
        obs, state, available_actions = self.envs.reset()
        state=self.repeat(state)
        return obs, state, available_actions

    def seed(self, seed):
        pass

    def render(self):
        pass

    def close(self):
        self.envs.env_close()

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]

    def split(self, a):
        return [a[i] for i in range(self.n_agents)]

    # def get_available_actions(self,AgentID):
    #     available_actions = self.envs.get_available_actions(AgentID)
    #     return available_actions


