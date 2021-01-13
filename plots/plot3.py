import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import pickle
import torch
num_sample = 10
# start=10
# end =21
# 1 original
# 2 greedy

rate1 = []
with open('../result/A1_hybrid_rate','r') as f:
    for line in f:
        s_str = line.strip().split()
        rate1 += [float(i) for i in s_str]

acc1 = []
with open('../result/A1_hybrid_zzz','r') as f:
    for line in f:
        s_str = line.strip().split()
        acc1 += [float(i) for i in s_str]

# rate2 = []
# with open('../result/A1_random_rate','r') as f:
#     for line in f:
#         s_str = line.strip().split()
#         rate2 += [float(i) for i in s_str]

# acc2 = []
# with open('../result/A1_random_acc','r') as f:
#     for line in f:
#         s_str = line.strip().split()
#         acc2 += [float(i) for i in s_str]

# rate3 = []
# with open('../result/A1_truncate_rate','r') as f:
#     for line in f:
#         s_str = line.strip().split()
#         rate3 += [float(i) for i in s_str]

# acc3 = []
# with open('../result/A1_truncate_acc','r') as f:
#     for line in f:
#         s_str = line.strip().split()
#         acc3 += [float(i) for i in s_str]

loss = True
weight_decay = 0
dropout =0
dataset = 'protein'
gnn = 'GCN'
styles = ['hybrid', 'random', 'truncate']
pstyles = ['bbp','pbb']
style = styles[0]
pstyle = pstyles[1]
train =[0.27771476 ,0.27749644 ,0.27840606 ,0.27912146 ,0.28851295 ,0.29045433,
0.29579503 ,0.298406  , 0.30020384]


ratio1 = ratio2 = [0.05        , 0.16666667, 0.33333334 ,0.5      ,  0.66666663 ,0.83333331,0.9,0.95,1
      ]

test =[0.30831249 ,0.30448948, 0.3019821  ,0.28831133 ,0.31690541 ,0.31288743,
0.33215619 ,0.33222317, 0.34257126]
# # print(pd.read_pickle(file_name[0]))
# # for i in range(len(file_name)):
# #     df = pd.read_pickle(file_name[i])
# #     if 'num_gnns' not in df.columns :
# #         new_df = pd.DataFrame({'Layers': df['layers'].to_list()*2 , 'epochs': df['epochs'].to_list()*2, 'CrossEntropyLoss': torch.Tensor(df['train_loss'].to_list()
# #                                 + df['test_loss'].to_list()).numpy(), 'set': ['train'] * len(df['layers'].to_list())+['test'] * len(df['layers'].to_list())})
# #     else:
# #         new_df = pd.DataFrame({'Layers': df['num_gnns'].to_list()*2 , 'epochs': df['epochs'].to_list()*2, 'CrossEntropyLoss': torch.Tensor(df['train_loss'].to_list()
# #
# #                                 + df['test_loss'].to_list()).numpy(), 'set': ['train'] * len(df['test_loss'].to_list())+['test'] * len(df['test_loss'].to_list())})
# #celoss = [0.2506,0.2499,0.2393,0.2381,0.2436,0.2407,0.2406,0.2577,0.2396,0.2342,0.2425,0.2404,0.2419,0.2426,0.2380,0.2385,0.2387,0.2455,0.2391,0.2370] +[0.2888,0.3178,0.2913,0.2881,0.2927,0.2850,0.2748,0.2765,0.2781,0.2830,0.2754,0.2916,0.2730,0.2768,0.2767,0.2762,0.2776,0.2780,0.2820,0.2801]
# #celoss = [0.3382,0.3152,0.2762,0.2771,0.2757,0.2748,0.2744,0.2740,0.2738,0.2735,0.2733,0.2732,0.2730,0.2729,0.2729,0.2728,0.2728,0.2728,0.2728,0.2728]  + [0.3422,0.3350,0.3250,0.3292,0.3273,0.3225,0.3212,0.3212,0.3206,0.3210,0.3214,0.3213,0.3211,0.3209,0.3208,0.3205,0.3196,0.3188,0.3186,0.3178]


# acc = torch.Tensor(acc1)
# rate = torch.Tensor(rate1)
# index = torch.abs(acc - 0.399)>0.001
# acc = acc[index]
# keys = (acc==1).nonzero().squeeze()
# print(len(acc))
# print(len(keys))
# acc = acc.numpy()
# rate = rate[index].numpy()
# A = (keys[0] + 1).item()
# B = (keys[1] - keys[0]).item()
# C = (keys[2] - keys[1]).int().item()
acc = train+test
rate = ratio1+ratio2
print(len(acc),len(rate))
new_df = pd.DataFrame({'CEloss':acc, 'rate':rate, 'type': ['train']*len(train) +['test']*len(test) })
sns.set(style="darkgrid", palette="bright",font_scale=1.4)
ax = sns.relplot(
    data=new_df, x='rate', y="CEloss", hue='type', style='type', kind='line', markers=True
)
ax.savefig(f'./SRGNN_loss_dataset_{dataset}_gnn_{gnn}_wd_{weight_decay}_dropout_{dropout}_style_{style}_pstyle_{pstyle}.jpeg')