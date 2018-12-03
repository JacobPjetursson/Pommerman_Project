import PytorchAgent1
import pommerman
from pommerman import agents

from models.model_pomm import PommNet
from models.policy import Policy
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
from our_ppo import PPO
import math
import torch
import random
import copy


def main():
    load_bool = True

    nn = PommNet(torch.Size([5, 11, 11]))
    if load_bool:
        model_dict = torch.load("trained_models/ppo_net.pt")
        nn.load_state_dict(model_dict)
    policy = Policy(nn, action_space=Discrete(6))
    ppo = PPO(policy, 2.5e-3)
    ppo.set_deterministic(False)

    agent_list = [
        PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
        PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
        PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
        #agents.SimpleAgent(),  # PytorchAgent(ppo),
        #agents.SimpleAgent(),  # PytorchAgent(ppo),
        #agents.SimpleAgent(),  # PytorchAgent(ppo),
        PytorchAgent1.PytorchAgent(ppo)  # BLACKMAN, TOP RIGTH CORNER
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

    lost = [0, 0, 0, 0]

    for i_episode in range(5000):
        state = env.reset()
        done = False
        ppo.set_deterministic(False)

        run_bool = True
        old_reward = None
        while not done and run_bool:
            run_bool = False
            if i_episode % 5 == 0 and i_episode > 0:
                ppo.set_deterministic(True)
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if not old_reward:
                old_reward = reward

            # Train the agent
            for agent_index in range(4):
                if isinstance(agent_list[agent_index], PytorchAgent1.PytorchAgent):
                    if old_reward[agent_index] == 0:
                        agent_list[agent_index].model_step(state[agent_index], reward[agent_index])
                        run_bool = True
            old_reward = reward

        if i_episode % 5 == 0 and i_episode > 0:
            ppo.set_deterministic(True)
            env.render()

        for index in range(4):
            lost[index] += reward[index]

        print('Episode {} finished, {}'.format(i_episode + 1, lost))

        if i_episode % 10 == 0 and i_episode > 0:
            save_model = copy.deepcopy(nn).cpu().state_dict()
            torch.save(save_model, "trained_models/ppo_net.pt")

    env.close()


if __name__ == '__main__':
    main()
