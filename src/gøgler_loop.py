import pommerman
from pommerman import agents

import PytorchAgent1


def main():
    agent_list = [
        PytorchAgent1.PytorchAgent(),
        PytorchAgent1.PytorchAgent(),
        PytorchAgent1.PytorchAgent(),
        PytorchAgent1.PytorchAgent()
    ]
    # Make the "Free-For-All" environment using the agent list
    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i_episode in range(10):
        state = env.reset()
        done = False
        while not done:
            env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            agent_list[3].step(reward[3])
        print('Episode {} finished'.format(i_episode))
    env.close()


if __name__ == '__main__':
    main()
