import argparse

import torch

class args:
    def __init__(self):
        self.algo = 'a2c'
        self.lr = 2.5e-4
        self.lr_schedule = 25000000 # default None
        self.gamma = 0.99
        self.alpha = 0.99 # default 0.99
        self.value_loss_coef = 0.5
        self.ppo_epoch = 4 # default 4
        self.num_mini_batch = 32
        self.clip_param = 0.2
        self.num_stack = 1
        self.log_dir = "/tmp/gym"
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

        # To use visdom, type 'python -m visdom.server' in command line. Not working with windows for some reason
        self.vis = False
        self.vis_interval = 100
        self.port = 8097




def get_args():
    #parser.add_argument('--lr-schedule', type=float, default=None,
    #                    help='learning rate step schedule (default: None)')
    #parser.add_argument('--alpha', type=float, default=0.99,
    #                    help='RMSprop optimizer apha (default: 0.99)')
    #                    help='gae parameter (default: 0.95)')
    #parser.add_argument('--max-grad-norm', type=float, default=0.5,
    #                    help='max norm of gradients (default: 0.5)')
    #parser.add_argument('--seed', type=int, default=1,
    #                    help='random seed (default: 1)')
    #parser.add_argument('--num-processes', type=int, default=16,
    #                    help='how many training CPU processes to use (default: 16)')
    #parser.add_argument('--num-steps', type=int, default=5,
    #                    help='number of forward steps in A2C (default: 5)')
    #parser.add_argument('--sil-update-ratio', type=float, default=1.0,
    #                    help='sil off-policy updates per on-policy updates (default: 1.0)')
    #parser.add_argument('--sil-epochs', type=int, default=1,
    #                    help='number of sil epochs (default: 1)')
    #parser.add_argument('--sil-batch-size', type=int, default=80,
    #                    help='sil batch size (default: 80)')
    #parser.add_argument('--sil-entropy-coef', type=float, default=0.01,
    #                    help='entropy term coefficient (default: 0.0)')
    #parser.add_argument('--sil-value-loss-coef', type=float, default=0.01,
    #                    help='value loss coefficient (default: 0.01)')
    #parser.add_argument('--log-interval', type=int, default=10,
    #                    help='log interval, one log per n updates (default: 10)')
    #parser.add_argument('--save-interval', type=int, default=100,
    #                    help='save interval, one save per n updates (default: 100)')
    #parser.add_argument('--eval-interval', type=int, default=1000,
    #                    help='eval interval, one eval per n updates (default: None)')
    #parser.add_argument('--vis-interval', type=int, default=100,
    #                    help='vis interval, one log per n updates (default: 100)')
    #parser.add_argument('--env-name', default='PongNoFrameskip-v4',
    #                    help='environment to train on (default: PongNoFrameskip-v4)')
    #parser.add_argument('--save-dir', default='./trained_models/',
    #                    help='directory to save agent logs (default: ./trained_models/)')
    #parser.add_argument('--no-cuda', action='store_true', default=False,
    #                    help='disables CUDA training')
    #parser.add_argument('--add-timestep', action='store_true', default=False,
    #                    help='add timestep to observations')
    #parser.add_argument('--recurrent-policy', action='store_true', default=False,
    #                    help='use a recurrent policy')
    #parser.add_argument('--no-vis', action='store_true', default=False,
    #                    help='disables visdom visualization')
    #parser.add_argument('--port', type=int, default=8097,
    #                    help='port to run the server on (default: 8097)')
    #parser.add_argument('--no-norm', action='store_true', default=False,
    #                    help='disables normalization')
    #args = parser.parse_args()

    #
    #args.vis = not args.no_vis

    return args
