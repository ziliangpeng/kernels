import torch
import deep_gemm

M, K, N = 512, 4096, 4096
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device="cuda")
g.manual_seed(42)
A32 = torch.randn(M, K, device="cuda", generator=g)
W32 = torch.randn(N, K, device="cuda", generator=g)  # weight (N, K)
ref32 = torch.matmul(A32, W32.t())  # A @ W^T — like a linear layer

# DeepSeek recipe helpers (built into deep_gemm):
# per_token_cast_to_fp8: activation per-1x128-block quant
# per_block_cast_to_fp8: weight per-128x128-block quant
A_q, A_s = deep_gemm.per_token_cast_to_fp8(A32, use_ue8m0=False)   # 1x128 blocks
W_q, W_s = deep_gemm.per_block_cast_to_fp8(W32, use_ue8m0=False)   # 128x128 blocks
print("A_q", tuple(A_q.shape), A_q.dtype, "A_s", tuple(A_s.shape))
print("W_q", tuple(W_q.shape), W_q.dtype, "W_s", tuple(W_s.shape))

# fp8_gemm_nt expects (A_q, A_s), (W_q, W_s) -> out (M, N) bf16
out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
deep_gemm.fp8_gemm_nt((A_q, A_s), (W_q, W_s), out)
out = out.float()

d = (out - ref32).abs()
rms = ref32.pow(2).mean().sqrt().item()
fro = d.pow(2).sum().sqrt().item() / ref32.pow(2).sum().sqrt().item() * 100.0
mx = d.max().item() / ref32.abs().max().item() * 100.0
p99 = d.flatten().quantile(0.99).item() / rms * 100.0
print(f"\nDeepGEMM BLOCKWISE (1x128 act, 128x128 w): fro={fro:.4f}%  max={mx:.4f}%  p99={p99:.4f}%")
print("compare: H100 tensorwise=3.75% rowwise=3.75% | MI325X aiter blockwise=3.68%")
