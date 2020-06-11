import torch
import torch.nn as nn
import torch.optim as optim


class SARSA():
    def __init__(self,
                 model,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 beta_1=None,
                 beta_2=None,
                 algo='adam'):

        self.model = model

        self.max_grad_norm = max_grad_norm
        if algo == 'adam':
            self.optimizer = optim.Adam(model.parameters(), 
                    lr, eps=eps, betas=(beta_1, beta_2))
        elif algo == 'rmsprop':
            self.optimizer = optim.RMSprop(model.parameters(), 
                    lr, eps=eps, alpha=beta_2, momentum=beta_1)
        elif algo == 'sgd':
            self.optimizer = optim.SGD(model.parameters(), lr, momentum=beta_1)
        else:
            raise NotImplementedError()


    def update(self, rollouts, explore_policy=None, softmax_temp=None):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        
        qs = self.model(rollouts.obs[:-1].view(-1, *obs_shape))
        values = qs.gather(1, rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        self.optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return value_loss.item() 
