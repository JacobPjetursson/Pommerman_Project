import PytorchAgent
import pommerman
from pommerman import agents

from models.model_pomm import PommNet
from models.policy import Policy
from gym.spaces import Discrete
from our_ppo import PPO
import torch
import copy


def main():
    load_bool = False
    show_every = 1

    nn = PommNet(torch.Size([5, 11, 11]))
    if load_bool:
        model_dict = torch.load("trained_models/ppo_net.pt")
        nn.load_state_dict(model_dict)
    policy = Policy(nn, action_space=Discrete(6))
    policy = policy.cuda()
    ppo = PPO(policy, 2.5e-5)
    ppo.set_deterministic(False)

    agent_list = [
        #PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
        #PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
        #PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
        agents.SimpleAgent(),  # PytorchAgent(ppo),
        agents.SimpleAgent(),  # PytorchAgent(ppo),
        agents.SimpleAgent(),  # PytorchAgent(ppo),
        PytorchAgent.PytorchAgent(ppo)  # BLACKMAN, TOP RIGTH CORNER
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

    lost = [0, 0, 0, 0]

    for i_episode in range(5000):
        state = env.reset()
        done = False
        ppo.set_deterministic(False)

        run_bool = True
        old_reward = [0, 0, 0, 0]
        while not done and run_bool:
            run_bool = False
            if i_episode % show_every == 0 and i_episode > 0:
                ppo.set_deterministic(True)
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            # Train the agent
            for agent_index in range(4):
                agent = agent_list[agent_index]
                if isinstance(agent, PytorchAgent.PytorchAgent):
                    if old_reward[agent_index] == 0:
                        agent.model_step(state[agent_index], reward[agent_index])
                        run_bool = True
            old_reward = reward

        if i_episode % show_every == 0 and i_episode > 0:
            env.render()

        for index in range(4):
            lost[index] += reward[index]

        print('Episode {} finished in {}, {}'.format(i_episode + 1, state[0]["step_count"], lost))

        if i_episode % show_every == 0 and i_episode > 0:
            save_model = copy.deepcopy(nn).cpu().state_dict()
            torch.save(save_model, "trained_models/ppo_net.pt")

    env.close()


if __name__ == '__main__':
    main()
