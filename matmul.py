import torch

import triton
import triton.language as tl

@triton.jit
def mul_kernel(
  A,
  B,
  C,
  M: tl.constexpr,
  N: tl.constexpr,
  K: tl.constexpr,
  stride_am: tl.constexpr,
  stride_ak: tl.constexpr,
  stride_bk: tl.constexpr,
  stride_bn: tl.constexpr,
  BLOCK_M: tl.constexpr,
  BLOCK_N: tl.constexpr,
  BLOCK_K: tl.constexpr,
):
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)
  offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  offs_k = tl.arange(0, BLOCK_K)

  acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

  for k in range(0, K, BLOCK_K):
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc += tl.dot(a, b)
    offs_k = k + tl.arange(0, BLOCK_K)
  
  tl.store(C + offs_m[:, None] * stride_am + offs_n[None, :] * stride_bn, acc)

def mul(x: torch.Tensor, y: torch.Tensor):
  M, K = x.shape
  K, N = y.shape
  stride_am = x.stride(0)
  stride_ak = x.stride(1)
  stride_bk = y.stride(0)
  stride_bn = y.stride(1)
  
  w = torch.empty([M, N], device="cuda")
  
  grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
  mul_kernel[grid](
    x,y,w,
    M,N,K,
    stride_am, stride_ak, stride_bk, stride_bn,
    BLOCK_M=256, BLOCK_N=256, BLOCK_K=256
  )
  
  return w

def main():
  M = 1024
  N = 2048
  K = 4096
  x = torch.randn([M, K], device='cuda')
  y = torch.randn([K, N], device='cuda')
  
  output_torch = x @ y
  output_triton = mul(x, y)
  
  print(output_torch)
  print(output_triton)
  print(torch.max(torch.abs(output_torch - output_triton)))

if __name__ == '__main__':
  main()