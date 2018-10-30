from envs import make_vec_envs
from models import create_policy
from models import model_pomm
import torch
import hyperparams
import os
import pommerman
from pommerman import agents
from pommerman import characters
import glob

args = hyperparams.args()


class LoadedModelAgent(pommerman.agents.BaseAgent):

    def __init__(self, character=pommerman.characters.Bomber):
        super(LoadedModelAgent, self).__init__(character)
        self.device = torch.device("cuda:0" if args.cuda else "cpu")
        self.actor_critic = None
        self.model = None
        self.loadModel()

    def act(self, obs, action_space):
        #eval_episode_rewards = []
        #obs = eval_envs.reset()
        eval_recurrent_hidden_states = torch.zeros(args.num_processes,
                                                   self.actor_critic.recurrent_hidden_state_size, device=self.device)
        eval_masks = torch.zeros(args.num_processes, 1, device=self.device)

        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = self.model.act(
                obs, eval_recurrent_hidden_states, eval_masks, deterministic=True)

            return action

    def loadModel(self):
        # TODO - trim below
        eval_log_dir = args.log_dir + "_eval"
        train_envs = make_vec_envs(
            args.env_name, args.seed, args.num_processes, args.gamma, args.no_norm, args.num_stack,
            args.log_dir, args.add_timestep, self.device, allow_early_resets=False)

        #if args.eval_interval:
        #    eval_envs = make_vec_envs(
        #        args.env_name, args.seed + args.num_processes, args.num_processes, args.gamma,
        #        args.no_norm, args.num_stack, eval_log_dir, args.add_timestep, self.device,
        #        allow_early_resets=True, eval=True)

        #    if eval_envs.venv.__class__.__name__ == "VecNormalize":
        #        eval_envs.venv.ob_rms = train_envs.venv.ob_rms
        #else:
        #    eval_envs = None

        self.actor_critic = create_policy(
            train_envs.observation_space,
            train_envs.action_space,
            name='pomm',
            nn_kwargs={
                'batch_norm': False if args.algo == 'acktr' else True,
                'recurrent': args.recurrent_policy,
                'hidden_size': 512,
            },
            train=True)
        self.model = self.actor_critic
        #self.model = model_pomm.PommNet(train_envs.observation_space.shape)
        load_path = os.path.join(args.save_dir, args.algo)
        load_path = os.path.join(load_path, args.env_name + ".pt")
        print("Attempting to load in model")
        print(self.model.state_dict().keys())
        #pretrained_model = torch.load(load_path, map_location=lambda storage, loc: storage)
        pretrained_model = torch.load(load_path)
        print
        print(pretrained_model[0].keys())
        self.model.load_state_dict(pretrained_model[0])
        print("Model loaded successfully")
        # model.to(device)
        self.model.eval()



def main():
    print("Starting up Pommerman")
    print(pommerman.REGISTRY)

    loaded_model_agent = LoadedModelAgent()
    # Create a set of agents (exactly four)
    agent_list = [
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.RandomAgent(),
        agents.PlayerAgent(agent_control='wasd')
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
    env.close()

if __name__ == '__main__':
    main()
