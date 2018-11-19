import pommerman
from pommerman import agents

from gym.spaces import Discrete
import PytorchAgent1

import time

def main():
    agent_list = [
        agents.PlayerAgent(),
        agents.PlayerAgent(),
        agents.PlayerAgent(),
        PytorchAgent1.PytorchAgent() #BLACKMAN, TOP RIGTH CORNER
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

    total_rew = 0

    for i_episode in range(10):

        state = env.reset()
        done = False
        hmm = 0
        while not done and hmm == 0:
            if i_episode % 1 == 0:
                env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)

            #Train the agent
            agent_list[3].model_step(state[3], reward[3])
            hmm = reward[3]
        if i_episode % 1 == 0:
            env.render()
        if reward[3] == 1:
            total_rew = total_rew + 1

        print('Episode {} finished, {}, {}'.format(i_episode+1, total_rew, float(total_rew)/(i_episode+1)))
    env.close()


if __name__ == '__main__':
    main()
