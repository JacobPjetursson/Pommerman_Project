import SearchAgent_2 as SearchAgent
import pommerman
from pommerman import agents

from models.model_pomm import PommNet
from models.policy import Policy
from gym.spaces import Discrete
import torch
import copy
import random
from threading import Thread
from multiprocessing import Process, Lock, Pool, Manager
import os
import platform
import time


class run_search_agent():
	agent_list = [
		agents.SimpleAgent(),
		agents.SimpleAgent(),
		agents.SimpleAgent(),
		SearchAgent.SearchAgent()  # BLACKMAN, TOP RIGTH CORNER
	]

	env = pommerman.make('PommeFFACompetition-v0', agent_list)
	#env = pommerman.make('PommeFFA-v1', agent_list)

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
	print("Using", torch.cuda.get_device_name(torch.cuda.current_device())) 
	run_search_agent()

if __name__ == '__main__':
	main()


