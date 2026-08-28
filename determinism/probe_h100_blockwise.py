import torch, math

def ru(x, a=128): return ((x + a - 1) // a) * a
def ru4(x): return ((x + 3) // 4) * 4

M, K, N = 512, 4096, 4096
A8 = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
B8 = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t()
ck, cn, cm = 32, 32, 4

# (128x128 on A, 1x128 on B) — DeepSeek recipe
sa = torch.rand(cm, ck, device="cuda") * 0.01 + 0.99
sb = torch.rand(ck, N, device="cuda") * 0.01 + 0.99
for fa in [False, True]:
    try:
        out = torch._scaled_mm(A8, B8, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16, use_fast_accum=fa)
        print(f"h100 (128x128,1x128) fa={fa}: OK", out.shape)
        break
    except Exception as e:
        print(f"h100 (128x128,1x128) fa={fa} FAIL:", str(e)[:110])

# (1x128, 128x128)
sa2 = torch.rand(M, ck, device="cuda") * 0.01 + 0.99
sb2 = torch.rand(ck, cn, device="cuda") * 0.01 + 0.99
try:
    out = torch._scaled_mm(A8, B8, scale_a=sa2, scale_b=sb2, out_dtype=torch.bfloat16, use_fast_accum=False)
    print("h100 (1x128,128x128) fa=False: OK", out.shape)
except Exception as e:
    print("h100 (1x128,128x128) FAIL:", str(e)[:110])

# MXFP8 1x32: e8m0 swizzled scales
ra = ru(M) * ru4(math.ceil(K / 32))
rb = ru(N) * ru4(math.ceil(K / 32))
sa3 = torch.ones(ra, device="cuda", dtype=torch.float8_e8m0fnu)
sb3 = torch.ones(rb, device="cuda", dtype=torch.float8_e8m0fnu)
try:
    out = torch._scaled_mm(A8, B8, scale_a=sa3, scale_b=sb3, out_dtype=torch.bfloat16, use_fast_accum=False)
    print("h100 mxfp8 1x32: OK", out.shape)
except Exception as e:
    print("h100 mxfp8 FAIL:", str(e)[:130])
