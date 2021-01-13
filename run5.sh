a=truncate
d=protein
n=1
wd=0
p=pbb
dr=0.3
# CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=0.3 --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=$dr --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=$dr --style=hybrid --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=0.3 --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GAT'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='MLP' --runs=1