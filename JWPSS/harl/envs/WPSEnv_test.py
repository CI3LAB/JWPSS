import time

from harl.envs.WS_simulator.normal_env import Normal_environment
from harl.PSPLIB_dataset.Data_process.Data_process import Graph
import argparse
import numpy as np
import copy

import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


# path= '../PSPLIB_dataset/problems_30_complete.npy'
# graph=Graph(path,0)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="wps",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "wps"
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict


    print("args",args)

    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])


    # print("algo_args",algo_args)
    # print("env_args",env_args)

    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    path="../PSPLIB_dataset/problems_15.npy"
    data=np.load(path,allow_pickle=True)

    process=[]
    for i in range(1,2):
        j=0
        project=[]
        while j<1:
            j+=1
            print(j)
            performance=[]
            print("************************************project:",i,"**********************************************")
            data_project=copy.deepcopy(data[i])
            # print(data_project[1])
            print(data_project[3])
            for ii in range(len(data_project[3])):
                for jj in range(len(data_project[3])):
                    if data_project[3][ii][jj]!=0:
                        print(data_project[3][ii][jj])
                        print("the i is:",ii+1,"the j is:",jj+1)
                        print("the lag time is:",data_project[3][ii][jj])
            # data_project[4]=np.array([0,1,2,3,4,5,6,10,11,12,13,14])
            env = Normal_environment(env_args, data_project)
            a,b,c=env.reset()

            print("env.info",env.get_env_info(), "project:",i)

            flag= True
            step=0
            while flag:
                actions=[]
                for ii in range(env.get_env_info()["n_agents"]):
                    avail_actions = np.array(env.get_available_actions(ii))
                    avail_actions_agent=np.where(avail_actions==1)
                    action_agent=np.random.choice(avail_actions_agent[0])
                    # if action_agent<15:
                    #     print("the work packaging action is:",action_agent,"the agent is:",i,"the step is:",step)
                    env.step_agent(ii,action_agent)
                    actions.append(action_agent)
                reward,done,state,observation,available_actions,info,obj,individual=env.step1(actions)
                if done:
                    flag=False
                    print("the reward is:", reward)
                    print(env.executor.done_tasks)
                    objective = env.episode_obj()
                    print("the objective is:", objective, "the project is done!")
                    print("the episode reward is:", env.episode_reward)


                    print("resource1:", individual[2])
                    print("resource2:", individual[3])
                    print("resource3:", individual[4])
                    print("resource4:", individual[5])

                    print(individual[1])
                    print(len(individual[-1]))
                    print(len(individual[-2]))
                    print(individual[-1])
                    print(individual[-2])
                    performance.append(env.episode_reward)
                    objective=env.episode_obj()
                    performance.append(objective)
                    performance.append(individual[-1][-1])
                    performance.append(len(individual[-2]))
                step+=1
            project.append(performance)
        process.append(project)
    process=np.array(process)

    # path = '../process_case_gurobi.npy'
    # np.save(path, process)






if __name__ == "__main__":
    main()