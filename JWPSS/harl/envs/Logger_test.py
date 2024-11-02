#测试SummaryWriter
import copy
import os
import json
import argparse
from tensorboardX import SummaryWriter
import numpy as np

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--logdir', type=str, default='test_log', help='logdir')
#     args = parser.parse_args()
#     writer = SummaryWriter(args.logdir)
#
#     r = 5
#     for i in range(100):
#         writer.add_scalars('run_14h', {'xsinx': i * np.sin(i / r),
#                                        'xcosx': i * np.cos(i / r),
#                                        'tanx': np.tan(i / r)}, 10)
#     writer.close()


#test the gpu
# import torch
#
# def main():
#     print(torch.cuda.is_available())
#     print(torch.cuda.device_count())
#     print(torch.cuda.current_device())
#     print(torch.cuda.get_device_name(0))
#
# if __name__ == '__main__':
#     main()
runnable_task = [7,9]
temp=copy.deepcopy(runnable_task)
task_list=[7,9,10]
running_task_dict = {}

for i in runnable_task:
    print(i)
    running_task_dict.update({i: 3})
    temp.remove(i)


print(temp)
runnable_task = temp

print("running_task_dict",running_task_dict)
print("runnable_task",runnable_task)