'''
A nice and efficient mixed AdamW/Muon Combined Optimizer.
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.
'''

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
'''AdamW optimizer with fused kernel'''

@torch.compile(dynamic=False, fullgraph=True)
def adam_step_fused(
  p: Tensor,            # (32768, 768) - parameter tensor
  grad: Tensor,         # (32768, 768) - gradient, same shape as p
  exp_avg: Tensor,      # (32768, 768) - first moment, same shape as p
  exp_avg_sq: Tensor,   # (32768, 768) - second moment, same shape as p
  step_t: Tensor,       # () - 0-D CPU tensor, step count
  lr_t: Tensor,         # () - 0-D CPU tensor, learning rate
  beta1_t: Tensor,      # () - 0-D CPU tensor, beta1
  beta2_t: Tensor,      # () - 0-D CPU tensor, beta2
  eps_t: Tensor,        # () - 0-D CPU tensor, epsilon
  wd_t: Tensor          # () - 0-D CPU tensor, weight decay
) -> None:
  '''
  Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
  All in one compiled graph to eliminate python overhead between ops.
  The 0-D CPU tensors avoid recompilation when hyperparameters values change.
  '''

  # weight decay (decoupled, applied before the update)
  p.mul_(1 - lr_t * wd_t)
  # update the running averages (lerp_ is cleaner and fuses well)
  exp_avg.lerp_(grad, 1 - beta1_t)
  exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
  # Bias corrections
  bias1 = 1 - beta1_t ** step_t
  bias2 = 1 - beta2_t ** step_t
  # compute and update
  denom = (exp_avg_sq / bias2).sqrt() + eps_t
  step_size = lr_t / bias1
  p.add_(exp_avg / denom, alpha=-step_size) # each element of exp_avg/denom is scaled by alpha before adding

# -----------------------------------------------------------------------------
'''
Muon optimizer

Background:
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
zero even beyond the point where the iteration no longer converges all the way to one everywhere
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
performance at all relative to UV^T, where USV^T = G is the SVD.

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
Polar Express Sign Method for orthogonalization.
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
'''