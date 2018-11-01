import argparse

import torch

class args:
    def __init__(self):
        self.algo = 'ppo'
        self.lr = 2.5e-4
        self.lr_schedule = None # default None
        self.gamma = 0.99
        self.alpha = 0.99 # default 0.99
        self.value_loss_coef = 0.5
        self.ppo_epoch = 4 # default 4
        self.num_mini_batch = 32
        self.clip_param = 0.2
        self.num_stack = 1
        self.cuda = torch.cuda.is_available()
        self.env_name = "PommeFFACompetitionFast-v0"
        self.eval_interval = 1000 # default 1000
        self.entropy_coef = 0.01
        self.num_steps = 5 # used?
        self.num_frames = 5e6 # default 5e7
        self.eps = 1e-5 # epsilon?
        self.max_grad_norm = 0.5
        self.num_processes = 8 # default 16
        self.use_gae = True # default False
        self.tau = 0.95
        self.save_interval = 100 # default 100
        self.save_dir = "./trained_models/"
        self.log_interval = 10
        self.seed = 0
        self.no_norm = True # default False
        self.add_timestep = False
        self.recurrent_policy = False


def get_args():
    return args
