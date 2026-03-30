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
def adamw_step_fused(
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

Some of the changes in this implementation:
- Uses a simpler, more general approach to parameter grouping and stacking
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)

References:
https://kellerjordan.github.io/posts/muon/
https://jeremybernste.in/writing/deriving-muon
https://www.youtube.com/watch?v=bO5nvE289ec
'''

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
  stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
  stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
  momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
  second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
  momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
  lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
  wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
  beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
  ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
  red_dim: int,                   # -1 or -2 - reduction dimension for variance
) -> None:
  '''
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    All in one compiled graph to eliminate Python overhead between ops.
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
  '''

  # Nesterov momentum
  momentum = momentum_t.to(stacked_grads.dtype)
  # This updates the Velocity (the moving average of gradients). This is a standard exponential moving average. 
  # The buffer now holds the 'accumulated speed' of the descent.
  momentum_buffer.lerp_(stacked_grads, 1 - momentum) 
  # This is the 'Nesterov Look-ahead' step. Instead of just using the gradient, we mix the gradient with the newly updated buffer.
  g = stacked_grads.lerp_(momentum_buffer, momentum)

  # Polar express
  # Cast to bf16 for speed when available; skip cast otherwise (fp16 is unstable here due to limited exponent range)
  X = g.bfloat16()

  # Before starting the iterative loop, the matrix is scaled down. 
  # This ensures the 'spectral radius' (the largest eigenvalue) is less than 1. 
  # If we didn't do this, the following loop would likely explode rather than converge.
  X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)

  # In older versions of this logic (Newton-Schulz), the update was a simple polynomial like $X = 1.5X - 0.5X(X^TX)$. 
  # While reliable, it took many iterations to reach a truly orthogonal state.
  # Polar Express uses optimized coefficients derived from Chebyshev polynomials. 
  # Instead of a 'one-size-fits-all' update, each step in polar_express_coeffs is mathematically tuned to 'squeeze' the eigenvalues toward 1 as fast as possible.
  if g.size(-2) > g.size(-1): # Tall matrix
    for a, b, c in polar_express_coeffs[:ns_steps]:
      A = X.mT @ X # (N x M).mT @ (N x M) -> (M x M)
      # This is a third-order polynomial update. 
      # By including the $A^2$ term (the square of the covariance), 
      # the algorithm can correct errors much more aggressively than a linear update. It's essentially 'Newton's Method on steroids' for matrices.
      B = b * A + c * (A @ A)
      X = a * X + X @ B
  else:
    for a, b, c, in polar_express_coeffs[:ns_steps]:
      # This is a third-order polynomial update. 
      # By including the $A^2$ term (the square of the covariance), 
      # the algorithm can correct errors much more aggressively than a linear update. It's essentially 'Newton's Method on steroids' for matrices.
      A = X @ X.mT
      B = b * A + c * (A @ A)
      X = a * X + B @ X

  g = X

  # variance reduction
  beta2 = beta2_t.to(g.dtype)
  v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
  red_dim_size = g.size(red_dim)
  v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
  v_norm = v_norm_sq.sqrt()
  second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
  step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
  scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
  v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
  final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
  g = g * final_scale.to(g.dtype)

  # Cautious weight decay + parameter update
  lr = lr_t.to(g.dtype)
  wd = wd_t.to(g.dtype)
  mask = (g * stacked_params) >= 0
  stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# Used mostly for reference, debugging and testing.

class MuonAdamW(torch.optim.Optimizer):
    '''
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.

    AdamW - Fused AdamW optimizer step.

    Muon - MomentUm Orthogonalized by Newton-schulz
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    '''
    def __init__(self, param_groups: list[dict]):
      super().__init__(param_groups, defaults={})
      # 0-D CPU tensors to avoid torch.compile recompilation when values change
      # AdamW tensors
      self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
       # Muon tensors
      self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
      self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    
    def _step_adamw(self, group: dict) -> None:
      '''
        AdamW update for each param in the group individually.
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
      '''
      for p in group['params']:
        if p.grad is None:
          continue
        grad = p.grad
        state = self.state[p]

        # state init
        if not state:
          state['step'] = 0
          state['exp_avg'] = torch.zeros_like(p)
          state['exp_avg_sq'] = torch.zeros_list(p)
        
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        state['step'] += 1

        # Fill 0-D tensors with current values
        self._adamw_step_t.fill_(state['step'])
        self._adamw_lr_t.fill_(group['lr'])
        self._adamw_beta1_t.fill_(group['betas'][0])
        self._adamw_beta2_t.fill_(group['betas'][1])
        self._adamw_eps_t.fill_(group['eps'])
        self._adamw_wd_t.fill_(group['weight_decay'])

        # Fused update: weight_decay -> momentum -> bias_correction -> param_update
        adamw_step_fused(
          p, grad, exp_avg, exp_avg_sq,
          self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
          self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
        )

    
    def _step_muon(self, group: dict) -> None:
      '''
        Muon update for all params in the group (stacked for efficiency).
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
      '''
      params: list[Tensor] = group['params']

      if not params:
        return

      # Get or create group-level buffers (stored in first param's state for convenience)
      p = params[0]
      state = self.state[p]
      num_params = len(params)
      shape, device, dtype = p.shape, p.device, p.dtype

      # Momentum for every individual parameter
      if 'momentum_buffer' not in state:
        state['momentum_buffer'] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
      
      momentum_buffer = state['momentum_buffer']
      
      # Second momentum buffer is factored, either per-row or per-column
      if 'second_momentum_buffer' not in state:
        state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
        state['second_momentum_buffer'] = torch.zeros(state_shape, dtype=dtype, device=device)
      
      second_momentum_buffer = state['second_momentum_buffer']
      red_dim = -1 if shape[-2] >= shape[-1] else -2

      # Stack grads and params (NOTE: this assumes all params have the same shape)
      stacked_grads = torch.stack([p.grad for p in params])
      stacked_params = torch.stack(params)

      # Fill all the 0-D tensors with current values
      self._muon_momentum_t.fill_(group["momentum"])
      self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
      self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
      self._muon_wd_t.fill_(group["weight_decay"])

      # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
      muon_step_fused(
        stacked_grads,
        stacked_params,
        momentum_buffer,
        second_momentum_buffer,
        self._muon_momentum_t,
        self._muon_lr_t,
        self._muon_wd_t,
        self._muon_beta2_t,
        group["ns_steps"],
        red_dim,
      )

      # Copy back to original params
      torch._foreach_copy_(params, list(stacked_params.unbind(0)))
    
    @torch.no_grad()
    def step(self):
      for group in self.param_groups:
        if group['kind'] == 'adamw':
          self._step_adamw(group)
        elif group['kind'] == 'muon':
          self._step_muon(group)
        else:
          raise ValueError((f'Unknown optimizer kind: {group['kind']}'))


# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# Used for training on multiple GPUs.

class DistMuonAdamW(torch.optim.Optimizer):
  '''
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.
    This class adds distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:x
  '''