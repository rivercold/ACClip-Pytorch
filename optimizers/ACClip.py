import torch
from torch.optim import Optimizer
import math

class ACClip(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), eps=1e-5,
                 weight_decay=0, alpha = 1, mod=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 1.0 <= alpha <= 2.0:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, alpha=alpha, mod=mod)
        super(ACClip, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ACClip, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ACClip does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # the momentum term, i.e., m_0
                    state['momentum'] = torch.zeros_like(p.data)
                    # the clipping value, i.e., \tao_0^{\alpha}
                    state['clip'] = torch.zeros_like(p.data)

                    # second-order momentum, i.e., v_t
                    state['second_moment'] = torch.zeros_like(p.data)

                momentum, clip, second_moment = state['momentum'], state['clip'], state['second_moment']

                beta1, beta2 = group['betas']

                alpha = group['alpha']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # update momentum and clip
                momentum.mul_(beta1).add_(1-beta1, grad)
                clip.mul_(beta2).add_(1 - beta2, grad.abs().pow(alpha))
                second_moment.mul_(beta2).addcmul_(1-beta2, grad, grad)

                # truncate large gradient
                denom = clip.pow(1/alpha).div(momentum.abs().add(group['eps'])).clamp(min=0.0, max=1.0)

                # calculate eta_t
                if group['mod'] == 1:
                    denom.div_(second_moment.mul(beta2).sqrt().add(group['eps']))

                p.data.addcmul_(-group['lr'], denom, momentum)



        return loss

