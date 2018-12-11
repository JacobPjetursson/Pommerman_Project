import PytorchAgent
import pommerman
from pommerman import agents

from models.model_pomm import PommNet
from models.policy import Policy
from gym.spaces import Discrete
from our_ppo import PPO
import torch
import copy
import threading
import random


def train(nn, number, save_bool=False):
	print("Starting thread " + str(number))
	policy = Policy(nn, action_space=Discrete(6))
	policy = policy.cuda()
	ppo = PPO(policy, 2.5e-4)
	ppo.set_deterministic(False)

	agent_list = [
		agents.SimpleAgent(),  # PytorchAgent(ppo),
		agents.SimpleAgent(),
		agents.SimpleAgent()  # PytorchAgent(ppo),
		#PytorchAgent.PytorchAgent(ppo)  # BLACKMAN, TOP RIGTH CORNER
	]
	agent_list.insert(random.randint(0,3),PytorchAgent.PytorchAgent(ppo))

	env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

	print("Working " + str(number))
	for i_episode in range(100):
		state = env.reset()
		done = False
		ppo.set_deterministic(False)

		run_bool = True
		old_reward = [0, 0, 0, 0]
		while not done and run_bool:
			run_bool = False
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

	env.close()
	if save_bool:
		save_model = copy.deepcopy(nn).cpu().state_dict()
		torch.save(save_model, "trained_models/ppo_net.pt")

def showcase(ppo):
	agent_list = [
		#PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
		#PytorchAgent1.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
		agents.SimpleAgent(),
		agents.SimpleAgent(),
		#PytorchAgent.PytorchAgent(ppo),  # BLACKMAN, TOP RIGTH CORNER
		agents.SimpleAgent(),  # PytorchAgent(ppo),
		#agents.SimpleAgent(),
		PytorchAgent.PytorchAgent(ppo)  # BLACKMAN, TOP RIGTH CORNER
	]
	#agent_list[3].set_train(False)

	env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

	lost = [0, 0, 0, 0]
	for i_episode in range(10):
		state = env.reset()
		done = False

		run_bool = True
		old_reward = [0, 0, 0, 0]
		while not done and run_bool:
			run_bool = False
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
		env.render()

		for index in range(4):
			lost[index] += reward[index]

		print('Episode {} finished in {}, {}'.format(i_episode + 1, state[0]["step_count"], lost))

	env.close()

def main():

	train = False

	load_bool = True
	save_bool = False
	number_of_threads = 8
	torch.set_num_threads(number_of_threads)

	nn = PommNet(torch.Size([5, 11, 11]))
	if load_bool:
		model_dict = torch.load("trained_models/ppo_net.pt")
		nn.load_state_dict(model_dict)
	
	nn = nn.cuda()

	if train:
		for i in range(number_of_threads):
			threading.Thread(target=train, args=(nn, i, save_bool)).start()
	else:
		policy = Policy(nn, action_space=Discrete(6))
		policy = policy.cuda()
		ppo = PPO(policy, 2.5e-4)
		ppo.set_deterministic(True)
		showcase(ppo)


if __name__ == '__main__':
	main()
