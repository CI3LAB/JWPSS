import numpy as np

# this the calss of task, which is the basic activity/task in a project
# it includes the task_idx, task_content, task_duration, and the resource consumption
class Task(object):
    def __init__(self, task_idx, task_content, task_duration, resource1, resource2, resource3, resource4):
        self.task_idx = int(task_idx)
        self.task_content = task_content
        self.task_duration = task_duration

        # the resource consumption of the task
        self.resource1 = resource1
        self.resource2 = resource2
        self.resource3 = resource3
        self.resource4 = resource4

        # the relationship between tasks
        self.task_parent_nodes = []
        self.task_child_nodes = []

        # task's predecessor and successor (it is the set rather the instant nodes)
        self.task_parent_set = []
        self.task_child_set = []
