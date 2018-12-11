import torch
import torch.optim as optim
import torch.nn as nn

class PPO():
    def __init__(self, policy, lr):

        self.policy = policy
#        self.critic = critic

        self.lr = lr
        self.clip_param = 0.2

        self.optimizer_policy = optim.Adam(policy.parameters(), lr=lr)
#        self.optimizer_critic = optim.Adam(critic.parameters(), lr=lr, weight_decay=1e-5)

    def set_deterministic(self, bo):
        self.policy.set_deterministic(bo)

    def update(self, states, critic_values, rewards, actions, action_log_probs):
        losses = torch.zeros(len(actions))

        for i in range(len(actions)):
            state = states[i]
            action = actions[i]
            critic_value = critic_values[i]
            action_log_prob = action_log_probs[i]
            reward = rewards[i]

            new_critic_value, new_action_log_prob, new_dist_entropy = self.policy.evaluate_actions(state, action)

            ratio = (new_action_log_prob - action_log_prob).exp()
            surr1 = ratio * (reward - critic_value)
            surr2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * (reward - critic_value)

            actor_loss = - surr2.mean()
            critic_loss = torch.mean((new_critic_value - reward)**2)

            t_loss = critic_loss + actor_loss - 0.001 * new_dist_entropy
            losses[i] = t_loss


        self.optimizer_policy.zero_grad()
        loss = losses.mean()
        loss.backward()
        self.optimizer_policy.step()


