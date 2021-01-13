# CUDA_VISIBLE_DEVICES=1 python3 old.py --dropout=0 --style=random --dataset=dblp --wd=0 --gnn='SAGE'
# CUDA_VISIBLE_DEVICES=1 python3 old.py --dropout=0 --style=random --dataset=dblp --wd=0 --gnn='GCN'
# CUDA_VISIBLE_DEVICES=1 python3 old.py --dropout=0 --style=random --dataset=dblp --wd=0 --gnn='GAT'
# CUDA_VISIBLE_DEVICES=1 python3 old.py --dropout=0 --style=random --dataset=dblp --wd=0 --gnn='GEN'
a=truncate
d=dblp
n=0
dp=0.3
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='SAGE'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
# CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GAT'
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GAT'
CUDA_VISIBLE_DEVICES=$n python3 old.py --dropout=0 --style=$a --dataset=$d --wd=$dp --gnn='GCN'
