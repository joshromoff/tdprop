import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from tdprop_backpack_online import TDprop
from backpack import backpack, extend
from backpack.extensions import BatchGrad


class SARSA():
    def __init__(self,
                 model,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 beta_1=None,
                 beta_2=None, 
                 gamma=0.99,
                 num_processes=16,
                 n=5):
        self.model = model
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm

        self.optimizer = TDprop(model.parameters(), 
                lr, eps=eps, betas=(beta_1, beta_2), 
                num_processes=num_processes, n=n, gamma=gamma)

    def update(self, rollouts, explore_policy, exploration):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        qs = self.model(rollouts.obs.view(-1, *obs_shape)).view(
                num_steps + 1, num_processes, -1)
        values = qs[:-1].gather(-1, rollouts.actions).view(num_steps, num_processes, 1)
        
        probs, _ = explore_policy(qs[-1].detach(), exploration)
        next_values = (probs * qs[-1]).sum(-1).unsqueeze(-1).unsqueeze(0) 
        
        advantages = rollouts.returns[:-1] - values
        
        with backpack(BatchGrad()):
            self.optimizer.zero_grad()
            torch.cat([values, next_values], dim=0).sum().backward()
            # store the td errors and masks to use inside tdprop
            self.optimizer.temp_store_td_errors(advantages.detach())
            self.optimizer.temp_store_masks(rollouts.cumul_masks)
            # extract grads: grad = -2 * td * grad_v 
            self.optimizer.extract_grads_from_batch()
            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            self.optimizer.step()

        return advantages.pow(2).mean().item() 
