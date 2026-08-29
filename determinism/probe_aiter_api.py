import torch, aiter, inspect

print("gemm_a8w8_blockscale signature:")
print(inspect.signature(aiter.gemm_a8w8_blockscale))
q = aiter.get_hip_quant(aiter.QuantType.per_1x128)
print("quant fn:", q)
try:
    print(inspect.signature(q))
except Exception as e:
    print("sig fail:", e)
print("quant doc:", (q.__doc__ or "")[:300])
