a=hybrid
d=Pubmed
n=1
dp=0
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='SAGE'
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GAT'
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='MLP' --runs=1
