#!/bin/bash
#SBATCH --job-name=ziliang-fp32acc
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/ziliang/fp32acc_%j.log

export PATH=/usr/local/nvidia/bin:$PATH
# torch 2.7.1+cu126 (user site) needs its bundled NCCL; Slurm batch env
# otherwise resolves system libnccl.so.2 (2.20.5) which lacks
# ncclCommInitRankScalable -> ImportError on import torch.
export LD_PRELOAD=/home/ziliang/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

python3 /home/ziliang/fp32_vs_lowprec.py --gpu h100
