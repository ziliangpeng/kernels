#!/bin/bash
#SBATCH --job-name=ziliang-fp8scale
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/home/ziliang/fp8scale_%j.log

export PATH=/usr/local/nvidia/bin:$PATH
export LD_PRELOAD=/home/ziliang/.local/lib/python3.10/site-packages/nvidia/nccl/lib/libnccl.so.2

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

python3 /home/ziliang/fp8_scaling_compare.py --gpu h100
