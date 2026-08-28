import torch, itertools

M, K, N = 512, 4096, 4096
A8 = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
B8 = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn).t()  # (K,N) col-major

ck, cn, cm = K // 128, N // 128, M // 128  # 32, 32, 4

# From PyTorch source (ScaledBlas.cpp):
# is_blockwise_1x128_scaling(t, scale):
#   scale.dim()==2, check_size_stride(scale,0,t.size(0),1),
#   check_size_stride(scale,1,ceil(t.size(1)/128), t.size(0))
#   => scale shape (t.size(0), t.size(1)/128), stride (1, t.size(0))
#
# is_blockwise_128x128_scaling(t, scale):
#   check_size_stride(scale,0,t.size(0)/128, t.size(1)/128)
#   check_size_stride(scale,1,t.size(1)/128, 1)
#   => scale shape (t.size(0)/128, t.size(1)/128), stride (t.size(1)/128, 1) = row-major
#
# get_joint_scaling does: is_desired_scaling(a, scale_a, lhs) && is_desired_scaling(b.t(), scale_b.t(), rhs)
# b.t() = (N, K) row-major (since B8 is (K,N) col-major from (N,K) row-major .t())
#
# For (1x128 on A, 128x128 on B):
#   a = A8 (M,K): scale_a (M, K/128) stride (1, M)  [col-major]
#   b.t() = (N,K): scale_b.t() checked as 128x128 on (N,K):
#     scale_b.t() shape (N/128, K/128) stride (K/128, 1) [row-major]
#     => scale_b shape (K/128, N/128) stride (1, K/128) [col-major]
#   So: sa = (M, ck) col-major, sb = (ck, cn) col-major

# For (128x128 on A, 1x128 on B):
#   a = A8 (M,K): scale_a (M/128, K/128) stride (K/128, 1) [row-major]
#   b.t() = (N,K): scale_b.t() checked as 1x128 on (N,K):
#     scale_b.t() shape (N, K/128) stride (1, N) [col-major]
#     => scale_b shape (K/128, N) stride (1, K/128)... wait
#     .t() of (K/128, N) = (N, K/128) with stride flipped
#     We need (N, K/128) stride (1, N) => original (K/128, N) stride (N, 1) = row-major
#   So: sa = (cm, ck) row-major, sb = (ck, N) row-major

def make_tensor(shape, stride):
    """Allocate strided tensor filled with 1.0."""
    storage_size = max(1, (shape[0] - 1) * abs(stride[0]) + (shape[1] - 1) * abs(stride[1]) + 1)
    buf = torch.ones(storage_size, device="cuda", dtype=torch.float32)
    return torch.as_strided(buf, shape, stride)

cases = {
    "(1x128, 128x128)": (
        make_tensor((M, ck), (1, M)),      # sa col-major
        make_tensor((ck, cn), (1, ck)),    # sb col-major
    ),
    "(128x128, 1x128)": (
        make_tensor((cm, ck), (ck, 1)),    # sa row-major
        make_tensor((ck, N), (N, 1)),      # sb row-major
    ),
    "(1x128, 1x128)": (
        make_tensor((M, ck), (1, M)),      # sa col-major
        make_tensor((ck, N), (N, 1)),      # sb row-major (b.t() is 1x128 on (N,K))
    ),
    "(128x128, 128x128)": (
        make_tensor((cm, ck), (ck, 1)),    # sa row-major
        make_tensor((ck, cn), (cn, 1)),    # sb... check: b.t()=(N,K), 128x128 => sb.t() (cn, ck) row-major => sb (ck, cn) col-major
    ),
}

for name, (sa, sb) in cases.items():
    for fa in [False, True]:
        try:
            out = torch._scaled_mm(A8, B8, scale_a=sa, scale_b=sb,
                                   out_dtype=torch.bfloat16, use_fast_accum=fa)
            print(f"{name} fa={fa}: OK  out={tuple(out.shape)}")
        except Exception as e:
            short = "shape-reject" if "Invalid scaling" in str(e) else str(e)[:80]
            print(f"{name} fa={fa}: {short}")
