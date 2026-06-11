import torch

import triton
import triton.language as tl

@triton.jit
def mul_kernel(
  x_ptr,
  y_ptr,
  n_elements,
  BLOCK_SIZE: tl.constexpr
)

def mul(x: torch.Tensor, y: torch. Tensor):
  