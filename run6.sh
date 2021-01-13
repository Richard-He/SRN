a=hybrid
d=protein
n=3
wd=1e-5
p=bbp
dr=0
# CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=0.3 --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
#CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=$dr --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=$dr --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN' \
--start=0.40 --end=1
# CUDA_VISIBLE_DEVICES=$n python3 SRGNN_pp.py --dropout=0.3 --style=$a --pstyle=$p --dataset=$d --wd=$wd --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GAT'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='MLP' --runs=1
#