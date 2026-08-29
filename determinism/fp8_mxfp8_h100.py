import torch, math

def ru(x, a=128): return ((x + a - 1) // a) * a
def ru4(x): return ((x + 3) // 4) * 4

M, K, N = 512, 4096, 4096
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device="cuda")
g.manual_seed(42)
A32 = torch.randn(M, K, device="cuda", generator=g)
B32 = torch.randn(K, N, device="cuda", generator=g)  # B (K,N) like _scaled_mm
ref32 = torch.matmul(A32, B32)

# --- MXFP8 (1x32, e8m0 scales, swizzled) via _scaled_mm ---
# scale numel: round_up(M,128) * round_up(ceil(K/32),4) for A
#              round_up(N,128) * round_up(ceil(K/32),4) for B (checked against b.t() dims? source says b.size(1)=N, b.size(0)=K)
sa_numel = ru(M) * ru4(math.ceil(K / 32))
sb_numel = ru(N) * ru4(math.ceil(K / 32))

def quant_1x32_e8m0(t):
    """Per 1x32 group quant with e8m0 scale. Returns (fp8, scale_swizzled)."""
    m, k = t.shape
    t5 = t.view(m, k // 32, 32).float()
    amax = t5.abs().amax(dim=2)  # (m, k/32)
    # e8m0 = power of 2 ceil
    exp = torch.ceil(torch.log2(amax.clamp(min=1e-12)))
    scale = torch.pow(2.0, exp)  # (m, k/32) — scale to divide by
    q = (t5 / scale.unsqueeze(2)).to(torch.float8_e4m3fn)
    return q.view(m, k), scale  # (m, k/32) fp32->e8m0 later

# quantize
A8, sa_raw = quant_1x32_e8m0(A32)  # (M,K) fp8, (M, K/32) scales as pow2
W8, sw_raw = quant_1x32_e8m0(B32.t().contiguous())  # quantize (N,K) per row
B8 = W8.t()  # (K,N) col-major view

# convert scales to e8m0 dtype
sa_e = sa_raw.to(torch.float8_e8m0fnu).contiguous()
sw_e = sw_raw.to(torch.float8_e8m0fnu).contiguous()

# need swizzled layout: SWIZZLE_32_4_4 for both A and B
# swizzle: (M, K/32) -> (ceil(M/128)*128, ceil(K/32/4)*4) pad then group
def swizzle_scale(s, rows, groups):
    """Pad to 128-row multiples and 4-block multiples, layout SWIZZLE_32_4_4."""
    pad_r = ru(rows)
    pad_g = ru4(groups)
    sp = torch.ones(pad_r, pad_g, device=s.device, dtype=torch.float8_e8m0fnu)
    sp[:rows, :groups] = s.to(torch.float8_e8m0fnu)
    # SWIZZLE_32_4_4: reshape (rows/32, 32, groups/4, 4) -> permute -> contiguous
    sp = sp.view(pad_r // 32, 32, pad_g // 4, 4)
    sp = sp.permute(0, 2, 1, 3).contiguous()  # (r/32, g/4, 32, 4)
    return sp.view(pad_r // 32 * pad_g // 4 * 32 * 4)

sa_sw = swizzle_scale(sa_raw, M, K // 32).to(torch.float8_e8m0fnu)
sw_sw = swizzle_scale(sw_raw, N, K // 32).to(torch.float8_e8m0fnu)
print("sa_sw numel:", sa_sw.numel(), "expected:", sa_numel)
print("sw_sw numel:", sw_sw.numel(), "expected:", sb_numel)

try:
    out = torch._scaled_mm(A8, B8, scale_a=sa_sw, scale_b=sw_sw, out_dtype=torch.bfloat16, use_fast_accum=False)
    out = out.float()
    d = (out - ref32).abs()
    rms = ref32.pow(2).mean().sqrt().item()
    fro = d.pow(2).sum().sqrt().item() / ref32.pow(2).sum().sqrt().item() * 100.0
    mx = d.max().item() / ref32.abs().max().item() * 100.0
    p99 = d.flatten().quantile(0.99).item() / rms * 100.0
    print(f"\nH100 MXFP8 (1x32 e8m0 swizzled): fro={fro:.4f}%  max={mx:.4f}%  p99={p99:.4f}%")
    print("(compare: tensorwise=3.75%, rowwise=3.75%, AMD blockwise 1x128=3.68%)")
except Exception as e:
    print("mxfp8 FAIL:", str(e)[:400])
