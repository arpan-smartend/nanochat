'''
Understanding AdamW and DDP get_future
cd dev
torchrun --nproc_per_node=4 1_figuring_out_adamw.py
'''
import torch
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from torch import Tensor
import os
import torch.nn as nn
import torch.nn.functional as F

torch._dynamo.config.reorderable_logging_functions.add(print)

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
  p: Tensor,
  grad: Tensor,
  exp_avg: Tensor,
  exp_avg_sq: Tensor,
  step_t: Tensor,
  lr_t: Tensor,
  beta1_t: Tensor,
  beta2_t: Tensor,
  eps_t: Tensor,
  wd_t: Tensor,
) -> None:
  '''
  Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
  All in one compiled graph to eliminate Python overhead between operations
  The 0-D CPU tensors avoid recompilation when hyperparameter values change.
  '''
  # p=Parameter containing:tensor([[-0.0014]], requires_grad=True)
  # grad=tensor([[0.]])
  # exp_avg=tensor([[0.]])
  # exp_avg_sq=tensor([[0.]])
  # step_t=1.0
  # lr_t=0.20000000298023224
  # beta1_t=0.8999999761581421
  # beta2_t=0.9990000128746033
  # eps_t=9.99999993922529e-09
  # wd_t=0.009999999776482582

  # 1. weight decay (decoupled, applied before the update)
  print(f'start: {p=}')
  p.mul_(1 - lr_t * wd_t)
  print(f'after mul: {p=}')

  # 2. update running averages (lerp_ is cleaner and fuses well)
  print(f'start: {exp_avg=}')
  print(f'start: {exp_avg_sq=}')
  exp_avg.lerp_(grad, 1 - beta1_t)
  exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
  print(f'after lerp: {exp_avg=}')
  print(f'after lerp: {exp_avg_sq=}')

  # 3. bias corrections
  bias1 = 1 - beta1_t ** step_t
  bias2 = 1 - beta2_t ** step_t

  # 4. Compute update and apply
  denom = (exp_avg_sq / bias2).sqrt() + eps_t
  step_size = lr_t / bias1
  p.add_(exp_avg / denom, alpha=step_size)
  print(f'final p: {p=}')


class DistAdamW(torch.optim.Optimizer):
  '''
    Distributed AdamW optimizer.
    In the style of ZeRO-2, i.e. sharded optimizer states and gradient reduction
  '''
  def __init__(self, param_groups, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # validate
    if rank == 0:
      for group in param_groups:
        assert isinstance(group, dict), 'expecting param_groups to be a list of dicts'
        assert isinstance(group['params'], list), 'expecting group["params"] to be a list of tensors'
        for p in group['params']:
          sliced = p.numel() >= 1024
          print(f'AdamW: 1 param of shape {p.shape}, sliced={sliced}')

          # large tensors with number of parameters >= 1024 will be operated in slices
          if sliced:
            assert p.shape[0] % world_size == 0, f'First dim of parameter shape {p.shape} must be divisible by world size {world_size}'
    
    super().__init__(param_groups, defaults)

    # 0-D CPU tensors to avoid torch.compile recompilation when values change
    self._step_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    self._lr_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    self._beta1_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    self._beta2_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    self._eps_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
    self._wd_t = torch.tensor(0.0, dtype=torch.float32, device='cpu')
  
  @torch.no_grad()
  def step(self):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    reduce_futures: list[torch.Future] = []
    gather_futures: list[torch.Future] = []
    grad_slices = []
    is_small = []  # track which params are small (use all_reduce) vs large (use reduce_scatter)

    for group in self.param_groups:
      # print(f'{group=} --- {rank=}', '\n')
      # print('-' * 7)
      params: list[Tensor] = group['params']
      for p in params:
        grad = p.grad
        # print(f'{grad=} --- {rank=}')
        # print('-' * 7)
        # Small params: use all_reduce (no scatter/gather needed)
        if p.numel() < 1024:
          is_small.append(True)
          # reduce at once if small
          reduce_futures.append(dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
          grad_slices.append(grad)

        else:
          is_small.append(False)
          rank_size = grad.shape[0] // world_size # p.shape[0] % world_size == 0 is checked in __init__
          grad_slice = torch.empty_like(grad[:rank_size])
          # reduce_scatter_tensor - Reduces, then scatters a tensor to all ranks in a group.
          # grad tensor to be reduced and scattered. Its size should be grad_slice size times the world size. The input tensor can have one of the following shapes: 
          # (i) a concatenation of the output tensors along the primary dimension, or 
          # (ii) a stack of the output tensors along the primary dimension.
          reduce_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
          grad_slices.append(grad_slice)
    
    # print(f'{is_small=} --- {rank=}', '\n')
    # print('-' * 7)
    # print(f'{reduce_futures=} --- {rank=}', '\n')
    # print('-' * 7)
    # print(f'{grad_slices=} --- {rank=}', '\n')
    # print('-' * 7)
    idx = 0
    for group in self.param_groups:
      beta1, beta2 = group['betas']
      eps = group['eps']
      wd = group['weight_decay']
      params = group['params']

      for p in params:
        # print(f'Before future wait grad: {grad_slices[idx]} - {rank=}', '\n')
        # print(f'Before future wait reduce: {reduce_futures[idx]} - {rank=}', '\n')
        reduce_futures[idx].wait()
        # print(f'After future wait grad: {grad_slices[idx]} - {rank=}', '\n')
        # print(f'After future wait reduce: {reduce_futures[idx]} - {rank=}', '\n')
        # print('-' * 7)
        g_slice = grad_slices[idx]
        lr = group['lr'] * getattr(p, 'lr_mul', 1.0)
        state = self.state[p]

        # For small params operate on full params, for large operate on slice
        if is_small[idx]:
          p_slice = p

        else:
          rank_size = p.shape[0] // world_size
          p_slice = p[rank * rank_size:(rank + 1) * rank_size]
        
        # state init
        if not state:
          state['step'] = 0
          state['exp_avg'] = torch.zeros_like(p_slice)
          state['exp_avg_sq'] = torch.zeros_like(p_slice)
        
        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']
        state['step'] += 1

        # Fill 0-D tensors with current values
        eff_wd = wd * getattr(p, 'wd_mul', 1.0)
        self._step_t.fill_(state['step'])
        self._lr_t.fill_(lr)
        self._beta1_t.fill_(beta1)
        self._beta2_t.fill_(beta2)
        self._eps_t.fill_(eps)
        self._wd_t.fill_(eff_wd)

        # Fused update: weight_decay -> momentum -> bias_correction -> param_update
        adamw_step_fused(p_slice, g_slice, exp_avg, exp_avg_sq, self._step_t, self._lr_t, self._beta1_t, self._beta2_t, self._eps_t, self._wd_t)

        # gather large params
        if not is_small[idx]:
          gather_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        
        idx += 1
    
    if gather_futures:
      torch.futures.collect_all(gather_futures).wait()



class SLM(nn.Module):
  def __init__(self):
    super().__init__()
    self.linear = nn.Linear(1, 1)
  
  def forward(self, x, y):
    logits = self.linear(x)
    loss = F.cross_entropy(logits, y)
    return logits, loss

  def setup_optimizer(self):
    linear_params = list(self.linear.parameters())
    adam_groups = [
      dict(params=linear_params, lr=0.2)
    ]
    adamw_kwargs = dict(betas=(0.9, 0.999), eps=1e-10, weight_decay=0.1)
    adamw_optim = DistAdamW(adam_groups, adamw_kwargs)

    for group in adamw_optim.param_groups:
      group["initial_lr"] = group["lr"]
    return adamw_optim
  

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='gloo')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cpu:{ddp_local_rank}'
    print(f'{ddp_rank=}, {ddp_local_rank=}, {ddp_world_size=}, {device=}')
    torch.cpu.set_device(device)


x = torch.randn((1, 1))
y = torch.randn((1, 1))

model  = SLM()
optim = model.setup_optimizer()
logits, loss = model(x, y)
loss.backward()
optim.step()
optim.zero_grad(set_to_none=True)

if ddp:
  destroy_process_group()