import torch

import triton
import triton.language as tl

@triton.jit
def mul_kernel(
  x_ptr,
  y_ptr,
  output_ptr,
  n_elements,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_K: tl.constexpr
):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(0)
  offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  offs_k = tl.arange(0, BLOCK_K)

  acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

  for k in range(0, K, BLOCK_K):
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc += tl.dot(a, b)
  
  tl.store(output_ptr+offs_k, acc)


def mul(x: torch.Tensor, y: torch. Tensor):
  pass