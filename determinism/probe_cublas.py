import ctypes, torch
print("torch cuda:", torch.version.cuda)
try:
    l = ctypes.CDLL("libcublasLt.so.12")
    print("cublasLt version:", l.cublasLtGetVersion())
except Exception as e:
    print("probe fail:", e)
# torch 2.7 bundled nvidia libs
import glob
for p in glob.glob("/home/ziliang/.local/lib/python3.10/site-packages/nvidia/cublas/lib/*"):
    print(p)
