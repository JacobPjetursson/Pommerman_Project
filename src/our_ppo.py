import torch
import torch.optim as optim

class PPO():
    def __init__(self, policy, lr):

        self.policy = policy
#        self.critic = critic

        self.lr = lr
        self.clip_param = 0.2

        self.optimizer_policy = optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)
#        self.optimizer_critic = optim.Adam(critic.parameters(), lr=lr, weight_decay=1e-5)

    def set_deterministic(self, bo):
        self.policy.set_deterministic(bo)

    def update(self, states, critic_values, rewards, actions, action_log_probs):
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

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (reward - new_critic_value).mean()

            loss = actor_loss - 0.05 * new_dist_entropy
            #if reward != 0:
            loss += 0.25 * critic_loss

            self.optimizer_policy.zero_grad()
            loss.backward()
            self.optimizer_policy.step()


