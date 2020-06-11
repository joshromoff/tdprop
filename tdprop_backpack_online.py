import math
import torch
import torch.optim as optim


class TDprop(optim.Optimizer):
    r"""Implements TDprop algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta_1 (float, optional): smoothing constant for first order info (default: 0.9)
        beta_2 (float, optional): smoothing constant for second order info default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
            gamma=0.99, n=5, num_processes=16, bias_correction=False):
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0]:
            raise ValueError("Invalid beta parameter at index: {}".format(betas[0]))
        if not 0.0 <= betas[1]:
            raise ValueError("Invalid beta parameter at index: {}".format(betas[1]))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                n=n, num_processes=num_processes, gamma=gamma,
                bias_correction=bias_correction)
        super(TDprop, self).__init__(params, defaults)

    
    def extract_grads_from_batch(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                # grad_batch has grad(v)
                grad = p.grad_batch.data[:-group['num_processes']]
                # extract true grad by multiplying grad_v by td
                grad = -2 * self._temp_td_errors[(..., ) + (None, ) * (grad.dim() - 1)] * grad
                p.grad.data = grad.mean(0) 
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                if grad.is_sparse:
                    raise RuntimeError('TDprop does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization 
                if len(state) == 0:
                    state['step'] = 0
                    state['z'] = torch.ones_like(p.data)
                    state['g'] = torch.zeros_like(p.data)
                    #state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['gamma_vector'] = torch.FloatTensor(
                            [group['gamma'] ** i for i in range(group['n'], 0, -1)]).type(
                                    p.type()).unsqueeze(-1).unsqueeze(-1) 

                z = state['z']
                g = state['g']
                beta_1, beta_2 = group['betas']
                n = group['n']
                num_processes = group['num_processes']
                gamma = group['gamma']
                gamma_vector = state['gamma_vector']
                state['step'] += 1
                bias_correction1 = 1 - beta_1 ** state['step'] if group['bias_correction'] else 1.0
                bias_correction2 = 1 - beta_2 ** state['step'] if group['bias_correction'] else 1.0
                
                self.clip_coef = 1. if self.clip_coef >= 1. else self.clip_coef
                # grad_batch contains [grad(v), grad(v')], use clipcoef computed by max grad norm
                grad_batch = p.grad_batch.data 
                grad_v = grad_batch[:-num_processes]
                grad_v_prime = grad_batch[-num_processes:]
                # discount * mask (terminal states) * grad(v(s_{t+n}))
                grad_v_prime = (self._temp_masks * gamma_vector * 
                        grad_v_prime.reshape(1, num_processes, -1)).view(grad_v.size())
                grad_td = grad_v_prime - grad_v
                # compute z for sampled batch
                cur_z = (2 * self.clip_coef * grad_td * grad_v).pow(2).mean(0)
                # update tracking params
                g.mul_(beta_1).add_(1 - beta_1, grad)
                z.mul_(beta_2).add_(1 - beta_2, cur_z)
                denom = (z.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(-step_size, g, denom)
        self._temp_td_errors = None
        self._temp_masks = None
        self.clip_coef = None
        return loss

    def temp_store_td_errors(self, td_errors):
        self._temp_td_errors = td_errors.view(-1) 

    def temp_store_masks(self, masks):
        self._temp_masks = masks
