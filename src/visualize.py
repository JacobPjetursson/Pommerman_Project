import argparse
import torch
from models.factory import create_policy
from envs import make_vec_envs

import SearchAgent_2 as SearchAgent
import pommerman
from pommerman import agents


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=10,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=1,
                    help='number of frames to stack (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PommeFFACompetitionFast-v0',
                    help='environment to train on (default: PommeFFACompetitionFast-v0)')
parser.add_argument('--load-path', default='trained_models/a2c/PommeFFACompetitionFast-v0.pt',
                    help='path to checkpoint file')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
parser.add_argument('--no-norm', action='store_true', default=True,
                    help='disables normalization')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA')
parser.add_argument('--use-search', action='store_true', default = False)
parser.add_argument('--hide', action='store_true', default = False)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")


def run_agent(args):
    num_env = 1
    env = make_vec_envs(args.env_name, args.seed + 1000,
                        num_env, gamma=None, no_norm=args.no_norm,
                        num_stack=args.num_stack, log_dir=None,
                        device=device, eval=True, allow_early_resets=False)

    # Get a render function
    render_func = None
    tmp_env = env
    while True:
        if hasattr(tmp_env, 'envs'):
            render_func = tmp_env.envs[0].render
            break
        elif hasattr(tmp_env, 'venv'):
            tmp_env = tmp_env.venv
        elif hasattr(tmp_env, 'env'):
            tmp_env = tmp_env.env
        else:
            break

    # We need to use the same statistics for normalization as used in training
    state_dict, ob_rms = torch.load(args.load_path)

    actor_critic = create_policy(
        env.observation_space,
        nn_kwargs={
            'batch_norm': True,
            'hidden_size': 512,
        },
        train=False)

    actor_critic.load_state_dict(state_dict)
    actor_critic.to(device)

    masks = torch.zeros(num_env, 1).to(device)

    obs = env.reset()

    if args.hide:
        render_func = None

    if render_func is not None:
        render_func('human')

    if args.env_name.find('Bullet') > -1:
        import pybullet as p

        torsoId = -1
        for i in range(p.getNumBodies()):
            if p.getBodyInfo(i)[0].decode() == "torso":
                torsoId = i

    rewards = []
    wins = 0
    deaths = 0

    step = 0

    while True:
        step = step + 1
        with torch.no_grad():
            value, action, _ = actor_critic.act(
                obs, masks, deterministic=True)

        obs, reward, done, _ = env.step(action)

        masks = torch.tensor([[0.0] if done_ else [1.0] for done_ in done]).to(device)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        for i, d in enumerate(done):
            if d:
                rewards.append(reward[i].item())
                if reward[i].item() > 0:
                    wins = wins + 1
                if reward[i].item() < 0 and step <= 800:
                    deaths = deaths + 1
                print("Game ended in {} steps, total games played: {}. Win rate: {}. Survival rate {}".format(step-1, len(rewards), float(wins) / len(rewards), 1.0-float(deaths)/len(rewards)))
                step = 0

        if render_func is not None:
            render_func('human')

def run_search_agent():
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        SearchAgent.SearchAgent()  # BLACKMAN, TOP RIGTH CORNER
    ]

    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
    env.seed(args.seed)
    #env = pommerman.make('PommeFFA-v1', agent_list)

    lost = [0, 0, 0, 0]
    wins = 0
    deaths = 0
    rewards = []

    for i_episode in range(100):
        print("Starting", i_episode, end=" ", flush=True)
        state = env.reset()
        done = False

        run_bool = True
        old_reward = [0, 0, 0, 0]
        while not done and run_bool:
            if state[0]["step_count"] % 20 == 0:
                print(".", end="", flush=True)
            run_bool = False
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if old_reward[3] == 0:
                run_bool = True
            old_reward = reward
            if not args.hide:
                env.render()

        for index in range(4):
            lost[index] += reward[index]
        rewards.append(reward[3])
        if reward[3] > 0:
            wins = wins + 1
        if reward[3] < 0 and state[3]["step_count"] < 800:
            deaths = deaths + 1

        print("Game ended in {} steps, total games played: {}. Win rate: {}. Survival rate {}".format(state[0]["step_count"], len(rewards), float(wins) / len(rewards), 1.0-float(deaths)/len(rewards)))

    env.close()


def main():
    use_search = args.use_search

    if use_search:
        run_search_agent()
    else:
        run_agent(args)




if __name__ == '__main__':
    main()