import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OUR_PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 lr_schedule=None,
                 eps=None,
                 max_grad_norm=None):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        if lr_schedule is not None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_schedule)
        else:
            self.scheduler = None

    def update(self, rollouts, update_index, replay=None):
        if self.scheduler is not None:
            self.scheduler.step(update_index)

        advantages = rollouts.returns - rollouts.value_preds
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            length = 10

            #REMEMBER THIS SHIT
            obs_list = []
            obs_batch = torch.FloatTensor().cuda() #length, 1, 1092).cuda()
            #obs_batch = torch.FloatTensor(length).cuda()
            recurrent_hidden_states_batch = torch.FloatTensor().cuda()#length, 1092).cuda()
            actions_batch = torch.LongTensor().cuda()
            return_batch = torch.FloatTensor().cuda()
            masks_batch = torch.FloatTensor().cuda()
            old_action_log_probs_batch = torch.FloatTensor().cuda()
            adv_targ = torch.FloatTensor().cuda()

            for sample in data_generator:
                obs_batch_t, recurrent_hidden_states_batch_t, actions_batch_t, return_batch_t, masks_batch_t, old_action_log_probs_batch_t, adv_targ_t = sample
                for i in range(obs_batch_t.shape[0]):
                    obs_batch = torch.cat([obs_batch, obs_batch_t[i,:]])
                    recurrent_hidden_states_batch = torch.cat([recurrent_hidden_states_batch, recurrent_hidden_states_batch_t[i,:]])
                    actions_batch = torch.cat([actions_batch, actions_batch_t[i,:]])
                    return_batch = torch.cat([return_batch, return_batch_t[i,:]])
                    masks_batch = torch.cat([masks_batch, masks_batch_t[i,:]])
                    masks_batch = torch.cat([masks_batch, masks_batch_t[i,:]])
                    old_action_log_probs_batch = torch.cat([old_action_log_probs_batch, old_action_log_probs_batch_t[i,:]])
                    adv_targ = torch.cat([adv_targ, adv_targ_t[i,:]])

            #values, action_log_probs, dist_entropy, states = self.actor_critic.evaluate_actions(
            #    obs_batch, recurrent_hidden_states_batch,
            #    masks_batch, actions_batch)

            values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                obs_batch.view(-1, *obs_shape),
                recurrent_hidden_states_batch[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
                masks_batch.view(-1, 1), actions_batch.view(-1, action_shape))


            ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                       1.0 + self.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(return_batch, values)

            self.optimizer.zero_grad()
            (value_loss * self.value_loss_coef + action_loss -
             dist_entropy * self.entropy_coef).backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)
            self.optimizer.step()

            value_loss_epoch += value_loss.item()
            action_loss_epoch += action_loss.item()
            dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, {}
