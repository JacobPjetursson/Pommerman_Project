import PytorchAgent
import SearchAgent_2 as SearchAgent
import pommerman
from pommerman import agents

from models.model_pomm import PommNet
from models.policy import Policy
from gym.spaces import Discrete
from our_ppo import PPO
import torch
import copy
import random
from threading import Thread
from multiprocessing import Process, Lock, Pool, Manager
import os
import platform
import time

lock = Lock()

class Train_env():
	def __init__(self, name, rounds, *args, **kwargs):
		self.rounds = rounds

		self.name = name
		self.nn = PommNet(torch.Size([4, 11, 11]))

		model_dict = torch.load("trained_models/ppo_net.pt")
		self.nn.load_state_dict(model_dict)
		self.nn = self.nn.cuda()
		self.time_loaded = self.get_time_path()	

		policy = Policy(self.nn, action_space=Discrete(6))
		policy = policy.cuda()
		self.ppo = PPO(policy, 2.5e-4)
		self.ppo.set_deterministic(False)

		our_agent = PytorchAgent.PytorchAgent(self.ppo)

		self.agent_list = [agents.RandomAgent() for i in range(3)]
		self.our_index = random.randint(0,3)
		self.agent_list.insert(self.our_index, our_agent)
		
		self.env = pommerman.make('PommeFFACompetitionFast-v0', self.agent_list)

	def get_nn(self):
		return self.nn

	def get_time_path(self):
		if platform.system() == 'Windows':
			return os.path.getctime('trained_models/ppo_net.pt')
		else:
			stat = os.stat('trained_models/ppo_net.pt')
			return stat.st_birthtime

	def load_new(self, i_episode):
		return
		t_time = self.get_time_path()
			
		if t_time > self.time_loaded:
			print(str(self.name), "-")
			self.time_loaded = t_time
			model_dict = torch.load("trained_models/ppo_net.pt")
			self.nn.load_state_dict(model_dict)
			self.nn = self.nn.cuda()
		
	def train(self):
		print("Starting training " + str(self.name))
		for i_episode in range(self.rounds):
			self.load_new(i_episode)			
			state = self.env.reset()
			done = False

			run_bool = True
			old_reward = [0, 0, 0, 0]
			old_state = state[self.our_index].copy()
			while not done and run_bool:
				run_bool = False
				actions = self.env.act(state)
				state, reward, done, info = self.env.step(actions)

				# Train the agent
				if old_reward[self.our_index] == 0:
					self.agent_list[self.our_index].model_step(old_state, reward[self.our_index])
					run_bool = True
				old_reward = reward
				old_state = state[self.our_index].copy()
					
		self.env.close()

def showcase(ppo):
	agent_list = [
		agents.SimpleAgent(),
		agents.SimpleAgent(),
		agents.SimpleAgent(), 
		PytorchAgent.PytorchAgent(ppo)  # BLACKMAN, TOP RIGTH CORNER
	]

	env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)

	lost = [0, 0, 0, 0]
	for i_episode in range(10):
		state = env.reset()
		done = False

		run_bool = True
		old_reward = [0, 0, 0, 0]
		while not done and run_bool:
			run_bool = False
			actions = env.act(state)
			state, reward, done, info = env.step(actions)
			if old_reward[3] == 0:
				run_bool = True
			old_reward = reward
			env.render()

			if 'episode' in info.keys():
				print(info)

		for index in range(4):
			lost[index] += reward[index]
		print('Episode {} finished in {}, {}'.format(i_episode + 1, state[0]["step_count"], lost))

	env.close()

class run_search_agent():
	agent_list = [
		agents.SimpleAgent(),
		agents.SimpleAgent(),
		agents.SimpleAgent(),
		SearchAgent.SearchAgent()  # BLACKMAN, TOP RIGTH CORNER
	]

	#env = pommerman.make('PommeFFACompetition-v0', agent_list)
	env = pommerman.make('PommeFFA-v1', agent_list)

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
			#env.render()

		for index in range(4):
			lost[index] += reward[index]
		print('Episode {} finished in {}, {}'.format(i_episode + 1, state[0]["step_count"], lost))

	env.close()
class ProcessCreator():
	def __init__(self, name, *args, **kwargs):
		self.name = name
		self.nn = None

	def train(self):
		threads = []
		nns = []
		envs = []
		for i in range(4):
			t = Train_env(str(self.name) + " " + str(i) , 32)
			threads.append(Thread(target=t.train))
			envs.append(t)
			threads[i].start()

		for t in threads:
			t.join()

		for env in envs:
			nns.append(env.get_nn())

		best_nn = ai_fighter(nns)
		save_model = copy.deepcopy(best_nn).cpu().state_dict()
		torch.save(save_model, "trained_models/ppo_net_"+ str(self.name) +".pt")

def ai_fighter(nns, show=False):
	agent_list = []
	for nn in nns:
		nn.eval()
		policy = Policy(nn, action_space=Discrete(6))
		policy = policy.cuda()
		ppo = PPO(policy, 2.5e-4)
		#ppo.set_deterministic(True)

		pygent = PytorchAgent.PytorchAgent(ppo)
		agent_list.append(pygent)

	env = pommerman.make('PommeFFACompetitionFast-v0', agent_list)
	state = env.reset()
	done = False
	while not done:
		actions = env.act(state)
		state, reward, done, info = env.step(actions)
		if show:
			env.render()
	if show:
		print(reward, state[0]["step_count"])
	env.close()
	for i in range(len(reward)):
		if reward[i] == 1:
			return nns[i]
	return random.choice(nns)

def main():
	print("Using", torch.cuda.get_device_name(torch.cuda.current_device())) 
	start = time.time()

	is_training = False
	create_bool = False #If we need to reset the stuff

	search_agent = True

	if not search_agent:
		number_of_threads = 4
		number_of_procs = 4
		epochs = 10000
		
		if create_bool:
			print("Generating new net")
			nn = PommNet(torch.Size([4, 11, 11]))
			save_model = copy.deepcopy(nn).cpu().state_dict()
			torch.save(save_model, "trained_models/ppo_net.pt")

		if is_training:
			torch.set_num_threads(number_of_threads*number_of_procs)
			for _ in range(epochs):
				processes = []
				for i in range(number_of_procs):
					t = ProcessCreator(i)
					processes.append(Process(target=t.train))
					processes[i].start()
				nns = []	
				for p in processes:
					p.join()

				# Time to fight the big boys
				nns =[]
				for i in range(number_of_procs):
					nn = PommNet(torch.Size([4, 11, 11]))
					model_dict = torch.load("trained_models/ppo_net_"+ str(i) +".pt")
					nn.load_state_dict(model_dict)
					nn = nn.cuda()
					nns.append(nn)

				best_nn = ai_fighter(nns, True)
				print("Saving", end=" ")
				save_model = copy.deepcopy(best_nn).cpu().state_dict()
				torch.save(save_model, "trained_models/ppo_net.pt")
				print("Done")

			end = time.time()
			print("Time is: ", end-start)

		else:
			nn = PommNet(torch.Size([4, 11, 11]))
			model_dict = torch.load("trained_models/ppo_net_0.pt")
			nn.load_state_dict(model_dict)
			nn = nn.cuda()
			nn.eval()

			policy = Policy(nn, action_space=Discrete(6))
			policy = policy.cuda()
			ppo = PPO(policy, 2.5e-4)
			ppo.set_deterministic(True)
			showcase(ppo)
	else:
		run_search_agent()

if __name__ == '__main__':
	main()


