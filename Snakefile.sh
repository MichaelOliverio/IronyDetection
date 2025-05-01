#!/bin/bash
#SBATCH -o Snakefile-ces-ukr.out
#SBATCH -e Snakefile-ces-ukr.out
#SBATCH --gres=gpu:1 
#SBATCH -p epito


source ~/mambaforge/etc/profile.d/conda.sh
conda activate llm

# Esporta il PYTHONPATH di PyTorch
export PYTHONPATH=/opt/pytorch-2.2.0/lib/python3.8/site-packages

# Verifica PyTorch e CUDA
python -c "import torch; print('Torch Version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Compiled with CUDA:', torch.version.cuda)"

# Lancia il tuo script
python fine-tuning-rhetorical-figures.py