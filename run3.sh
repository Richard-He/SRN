a=truncate
d=arxiv
n=0
dp=0
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='SAGE'
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GEN'
#CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GAT'
#CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='MLP' --runs=1
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=random --dataset=$d --wd=$dp --gnn='GEN'
