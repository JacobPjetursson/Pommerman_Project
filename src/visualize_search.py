import SearchAgent_2 as SearchAgent
import pommerman
from pommerman import agents

import torch
from models.policy import Policy


def get_RL_agent():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    nn = PommNet(torch.Size([9, 11, 11]))
    model_dict = torch.load("trained_models/a2c/PommeFFACompetitionFast-v0.pt")
    nn.load_state_dict(model_dict)
    nn = nn.to(device)

    policy = Policy(nn, action_space=Discrete(6))
    policy = policy.to(device)


class run_search_agent():
    agent_list = [
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        agents.SimpleAgent(),
        SearchAgent.SearchAgent()  # NIGGA, TOP RIGHT CORNER
    ]

    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

    lost = [0, 0, 0, 0]
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
            env.render()

        for index in range(4):
            lost[index] += reward[index]
        print('Episode {} finished in {}, {}'.format(i_episode + 1, state[0]["step_count"], lost))

    env.close()


def main():
    run_search_agent()


if __name__ == '__main__':
    main()
