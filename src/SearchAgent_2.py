'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent
from gym.spaces import Discrete
import math
import random
import pommerman
import numpy as np

from collections import deque


class SearchAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def __init__(self, *args, **kwargs):
        super(SearchAgent, self).__init__(*args, **kwargs)

        self.zero = np.zeros(1)
        self.still = self.zero
        self.up = self.zero + 1
        self.down = self.zero + 2
        self.left = self.zero + 3
        self.right = self.zero + 4
        self.bomb = self.zero + 5
        self.pos_actions = [self.still, self.up, self.down, self.left, self.right, self.bomb]

    def getFeatures(self, obs):
        me = obs["position"]

        obs_persons = (obs["board"] > 9)

        obs_rigid_wall = (obs["board"] == 1)
        obs_wood_wall = (obs["board"] == 2)

        # Board with passage 
        obs_board_passage = np.logical_and(obs["board"] >= 5, obs["board"] <= 8) + (obs["board"] == 0)

        #obs_board_passage = torch.FloatTensor((obs["board"] == 0).astype(int)) + torch.FloatTensor((obs["board"] == 5).astype(int)) + torch.FloatTensor((obs["board"] == 6).astype(int)) + torch.FloatTensor((obs["board"] == 7).astype(int)) + torch.FloatTensor((obs["board"] == 8).astype(int))

        # Getting the blastmap
        #obs_board_flames = torch.FloatTensor((obs["board"] == 4).astype(int))
        obs_board_bombs_life = (obs["bomb_life"])

        obs_board_bombs_blast = (obs["bomb_blast_strength"])
        obs_blast_map = self.get_blast_map(obs_board_bombs_blast, obs_board_bombs_life, obs_rigid_wall, obs_wood_wall)

        del obs_board_bombs_life, obs_board_bombs_blast

        return obs_persons, obs_blast_map, obs_board_passage, obs_wood_wall, obs_rigid_wall


    def get_possible_me_map_second_step(self, person, passage, me_map, action):
        if self.next_me(person, self.up, passage) and action != self.down:
            me_map[person[0] - 1, person[1]] += 0.25
        if self.next_me(person, self.down, passage) and action != self.up:
            me_map[person[0] + 1, person[1]] += 0.25
        if self.next_me(person, self.left, passage) and action != self.right:
            me_map[person[0], person[1] - 1] += 0.25
        if self.next_me(person, self.right, passage) and action != self.left:
            me_map[person[0], person[1] + 1] += 0.25

    def get_possible_me_map(self, person, passage, two_steps=False):
        me_map = np.zeros((11,11))
        me_map[person[0], person[1]] += 1

        if self.next_me(person, self.up, passage):
            me_map[person[0] - 1, person[1]] += 1
            if two_steps:
                self.get_possible_me_map_second_step((person[0] - 1, person[1]), passage, me_map, self.up)
        if self.next_me(person, self.down, passage):
            me_map[person[0] + 1, person[1]] += 1
            if two_steps:
                self.get_possible_me_map_second_step((person[0] + 1, person[1]), passage, me_map, self.down)
        if self.next_me(person, self.left, passage):
            me_map[person[0], person[1] - 1] += 1
            if two_steps:
                self.get_possible_me_map_second_step((person[0], person[1] - 1), passage, me_map, self.left)
        if self.next_me(person, self.right, passage):
            me_map[person[0], person[1] + 1] += 1
            if two_steps:
                self.get_possible_me_map_second_step((person[0], person[1] + 1), passage, me_map, self.right)
        return me_map

    def next_me(self, person, action, passage, ammo=0):
        if action == self.still:
            return True
        elif action == self.bomb and ammo > 0:
            return True
        elif action == self.up:
            return person[0] > 0 and passage[person[0] - 1, person[1]] == 1
        elif action == self.down:
            return person[0] < 10 and passage[person[0] + 1, person[1]] == 1
        elif action == self.left:
            return person[1] > 0 and passage[person[0], person[1] - 1] == 1
        elif action == self.right:
            return person[1] < 10 and passage[person[0], person[1] + 1] == 1

    def add_to_score(self, person, obs_board_passage, action_score):
        bool_x = False
        bool_y = False
        if self.next_me(person, self.up, obs_board_passage):
            action_score[person[0], person[1]] += 1
            bool_y = True
        if self.next_me(person, self.down, obs_board_passage):
            action_score[person[0], person[1]] += 1
            bool_y = True
        if self.next_me(person, self.left, obs_board_passage):
            action_score[person[0], person[1]] += 1
            bool_x = True
        if self.next_me(person, self.right, obs_board_passage):
            action_score[person[0], person[1]] += 1
            bool_x = True
        if bool_x and bool_y:
            action_score[person[0], person[1]] += 1

    def get_action_score(self, me, obs_persons, obs_board_passage):
        action_score = np.zeros((11,11))
        temp_board_passage = obs_board_passage.copy()
        temp_board_passage[me[0], me[1]] = 1
        to_look_at1, to_look_at2 = obs_persons.nonzero()
        for i in range(len(to_look_at1)):
            loc = (to_look_at1[i], to_look_at2[i])
            self.add_to_score(loc, temp_board_passage, action_score)
        to_look_at1, to_look_at2 = obs_board_passage.nonzero()
        for i in range(len(to_look_at1)):
            loc = (to_look_at1[i], to_look_at2[i])
            self.add_to_score(loc, temp_board_passage, action_score)
        return action_score


    def get_blast_map(self, blast_strength, blast_life, wall_map, wood_wall_map):
        default_value = pommerman.constants.DEFAULT_BOMB_LIFE
        # Større score = Større farer
        blast_map = np.zeros((11, 11))   #flames * default_value

        ys, xs = blast_strength.nonzero()

        for i in range(len(ys)):
            y = ys[i]
            x = xs[i]
            st = blast_strength[y, x]
            st = int(st)
            if st > 0:
                y_start = max(y - (st - 1), 0)
                y_end = min(y + st, 11)
                x_start = max(x - (st - 1), 0)
                x_end = min(x + st, 11)
                value = (default_value - blast_life[y, x] + 1)

                for i in reversed(range(y_start, y)):
                    if (blast_map[i, x] < value):
                        if wall_map[i, x]:
                            break
                        blast_map[i, x] = value
                        if wood_wall_map[i, x]:
                            break;
                for i in range(y, y_end):
                    if (blast_map[i, x] < value):  
                        if wall_map[i, x]:
                            break
                        blast_map[i, x] = value
                        if wood_wall_map[i, x]:
                            break;
                for i in reversed(range(x_start, x)):  
                    if (blast_map[y, i] < value):
                        if wall_map[y, i]:
                            break
                        blast_map[y, i] = value
                        if wood_wall_map[y, i]:
                            break;
                for i in range(x, x_end):  
                    if (blast_map[y, i] < value):
                        if wall_map[y, i]:
                            break
                        blast_map[y, i] = value 
                        if wood_wall_map[y, i]:
                            break;
        return blast_map*2

    def fake_blast_map_put_bomb(self, me, blast_strength, blast_map, passage, wall_map, wood_wall_map):
        default_value = pommerman.constants.DEFAULT_BOMB_LIFE
        new_blast = blast_map.copy()

        y, x  = me[0], me[1]
        y_start = max(me[0] - (blast_strength - 1), 0)
        y_end = min(me[0] + blast_strength, 11)
        x_start = max(me[1] - (blast_strength - 1), 0)
        x_end = min(me[1] + blast_strength, 11)

        value = 1

        for i in reversed(range(y_start, y)):
            if (new_blast[i, x] < value):
                if wall_map[i, x]:
                    break
                new_blast[i, x] = value
                if wood_wall_map[i, x]:
                    break;
        for i in range(y, y_end):
            if (new_blast[i, x] < value):  
                if wall_map[i, x]:
                    break
                new_blast[i, x] = value
                if wood_wall_map[i, x]:
                    break;
        for i in reversed(range(x_start, x)):  
            if (new_blast[y, i] < value):
                if wall_map[y, i]:
                    break
                new_blast[y, i] = value
                if wood_wall_map[y, i]:
                    break;
        for i in range(x, x_end):  
            if (new_blast[y, i] < value):
                if wall_map[y, i]:
                    break
                new_blast[y, i] = value 
                if wood_wall_map[y, i]:
                    break;
        return new_blast

    def get_best(self, me, persons_map, blast_map, fake_blast_map, passage):
        action_score = self.get_action_score(me, persons_map, passage)
        enemy_persons_map = persons_map.copy()        
        enemy_persons_map[me[0], me[1]] = 0

        point_map_normal = action_score - blast_map
        point_map_bomb = action_score - fake_blast_map

        # Approx to make it faster
        enemy_1, enemy_2 = enemy_persons_map.nonzero()
        enemies = [(enemy_1[i], enemy_2[i]) for i in range(len(enemy_1))]
        enemy_my_maps = [self.get_possible_me_map(enemy, passage) for enemy in enemies]
        enemy_score_normal = np.mean([(enemy_map*point_map_normal).max() for enemy_map in enemy_my_maps]) 
        enemy_score_bomb = np.mean([(enemy_map*point_map_bomb).max() for enemy_map in enemy_my_maps]) 

        bomb_ratio = 1
        if (not math.isnan(enemy_score_normal)) and (not math.isnan(enemy_score_bomb)) and enemy_score_bomb*enemy_score_normal > 0:
            bomb_ratio = (enemy_score_bomb / enemy_score_normal)

        me_map = self.get_possible_me_map(me, passage, two_steps=True)
        new_map = me_map * (action_score - blast_map)

        pos1, pos2 = new_map.nonzero()
        possible = [(pos1[i], pos2[i]) for i in range(len(pos1))]

        best_coor = me
        best_score = new_map[int(best_coor[0]), int(best_coor[1])]

        if len(possible) > 0:
            for i in possible:
                if new_map[int(i[0]), int(i[1])] >= best_score:
                    best_coor = i
                    best_score = new_map[int(i[0]), int(i[1])]

        if new_map[int(me[0]), int(me[1])] * bomb_ratio > best_score:
            print("B", end="", flush=True)
            return 5
        if int(best_coor[0]) < me[0]:
            return 1
        elif int(best_coor[0]) > me[0]:
            return 2
        elif int(best_coor[1]) < me[1]:
            return 3
        elif int(best_coor[1]) > me[1]:
            return 4
        return 0

    def act(self, obs, action_space):
        me = obs["position"]
        blast_strength = obs["blast_strength"]
        persons_map, blast_map, passage, wood_wall, rigid_wall = self.getFeatures(obs)
        fake_blast_map = self.fake_blast_map_put_bomb(me, blast_strength, blast_map, passage, rigid_wall, wood_wall)
        best = self.get_best(me, persons_map, blast_map, fake_blast_map, passage)
        return best
