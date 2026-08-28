import torch, aiter

M, K, N = 512, 4096, 4096
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device="cuda")
g.manual_seed(42)
A32 = torch.randn(M, K, device="cuda", generator=g)
W32 = torch.randn(N, K, device="cuda", generator=g)
ref32 = torch.matmul(A32, W32.t())

# Activation: per-1x128 (triton path works)
q_act = aiter.get_triton_quant(aiter.QuantType.per_1x128)
A8, s_a = q_act(A32, quant_dtype=aiter.dtypes.fp8)
print("A8:", tuple(A8.shape), A8.dtype, "s_a:", tuple(s_a.shape), s_a.dtype)

# Weight per-128x128: use torch-level block quant manually then call CK gemm.
# DeepSeek recipe: W (N,K) quantized per 128x128 block -> scale (N/128, K/128)
def per_block_quant_128x128(t):
    """Quantize (N,K) tensor per 128x128 block. Returns (fp8, scale (N/128, K/128))."""
    N_, K_ = t.shape
    assert N_ % 128 == 0 and K_ % 128 == 0
    tb = t.view(N_ // 128, 128, K_ // 128, 128).float()
    amax = tb.abs().amax(dim=(1, 3), keepdim=True).squeeze(dim=(1, 3))  # (Nb, Kb)
    scale = amax / 240.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (tb / scale.unsqueeze(1).unsqueeze(3)).to(aiter.dtypes.fp8)
    return q.view(N_, K_), scale

W8, s_w = per_block_quant_128x128(W32)
print("W8:", tuple(W8.shape), W8.dtype, "s_w:", tuple(s_w.shape), s_w.dtype)

try:
    out = aiter.gemm_a8w8_blockscale(A8, W8, s_a, s_w, bias=None)
    out = out.float()
    d = (out - ref32).abs()
    rms = ref32.pow(2).mean().sqrt().item()
    fro = d.pow(2).sum().sqrt().item() / ref32.pow(2).sum().sqrt().item() * 100.0
    mx = d.max().item() / ref32.abs().max().item() * 100.0
    p99 = d.flatten().quantile(0.99).item() / rms * 100.0
    print(f"\nBLOCKWISE result: fro={fro:.4f}%  max={mx:.4f}%  p99={p99:.4f}%")
    print("(compare: tensorwise=3.75%, rowwise=3.75%)")
except Exception as e:
    print("gemm fail:", str(e)[:400])
