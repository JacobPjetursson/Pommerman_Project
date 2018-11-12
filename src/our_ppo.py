import torch
import torch.optim as optim

class PPO():
    def __init__(self, policy, lr):

        self.policy = policy

        self.lr = lr
        self.clip_param = 0.2

        self.optimizer = optim.Adam(policy.parameters(), lr=lr)

    def set_deterministic(self, bo):
        self.policy.set_deterministic(bo)

    def update(self, states, critic_values, rewards, actions, action_log_probs):
        loss = 0
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
            critic_loss = (reward - new_critic_value).pow(2).mean()

            loss += 0.5 * critic_loss + actor_loss - 0.001 * new_dist_entropy

        loss = loss / len(actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

