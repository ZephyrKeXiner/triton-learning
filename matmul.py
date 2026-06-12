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
  stride_cm: tl.constexpr,
  stride_cn: tl.constexpr,
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
    offs_k = k + tl.arange(0, BLOCK_K)
    mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)
    b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=mask_b, other=0.0)
    
    acc += tl.dot(a, b, input_precision="ieee")
  
  mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
  tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask_c)

def mul(x: torch.Tensor, y: torch.Tensor):
  M, K = x.shape
  K, N = y.shape
  stride_am = x.stride(0)
  stride_ak = x.stride(1)
  stride_bk = y.stride(0)
  stride_bn = y.stride(1)
  
  w = torch.empty([M, N], device="cuda", dtype=torch.float32)
  stride_cm = w.stride(0)
  stride_cn = w.stride(1)
  
  grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
  mul_kernel[grid](
    x,y,w,
    M,N,K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M=32, BLOCK_N=32, BLOCK_K=32
  )
  
  return w

def main():
  M = 1024
  N = 256
  K = 512
  torch.manual_seed(42)
  x = torch.randn([M, K], device='cuda', dtype=torch.float32)
  y = torch.randn([K, N], device='cuda', dtype=torch.float32)
  
  output_torch = x @ y
  output_triton = mul(x, y)
  
  print(output_torch)
  print(output_triton)
  print(torch.max(torch.abs(output_torch - output_triton)))

if __name__ == '__main__':
  main()