import os
import pickle
import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')
sns.set()

def process():
    # test_idx = pickle.load(open('./project/DisenSemi/Our/models/test_idx.pkl', 'rb'))
    embs = []
    for i in range(6,7):
        # emb_test = pickle.load(open('./project/DisenSemi/Our/models/outs_u_test_'+str(i)+'.pkl', 'rb'))
        # emb_val = pickle.load(open('./project/DisenSemi/Our/models/outs_u_val_'+str(i)+'.pkl', 'rb'))
        emb_test = pickle.load(open('./project/DisenSemi/Our/models/outs_s_test_'+str(i)+'.pkl', 'rb'))
        emb_val = pickle.load(open('./project/DisenSemi/Our/models/outs_s_val_'+str(i)+'.pkl', 'rb'))
        # print(len(list(emb_test.values())[:3]))
        # print(len(emb_test))
        # for emb_ in emb_test.values():
        #     print(emb_)
        #     print(emb_[:5,:])
        emb_test = [torch.cat(emb_,dim=-1)[:,:] for emb_ in emb_test.values()]
        emb_val = [torch.cat(emb_,dim=-1)[:,:] for emb_ in emb_val.values()]
        # emb = emb_test + emb_val
        # emb = emb_val
        emb = emb_test
        emb = torch.cat(emb, dim=0).cpu().numpy()
        print(emb.shape)
        embs.append(emb)
    embs = np.concatenate(embs, axis=0)
    print(embs.shape)
    # print(embs)
    # choose = random.choice(test_idx).item()
    # emb = embs[choose].numpy()
    correlation = np.zeros((embs.shape[1], embs.shape[1]))
    for i in range(embs.shape[1]):
        for j in range(embs.shape[1]):
            cof = scipy.stats.pearsonr(embs[:, i], embs[:, j])[0]
            correlation[i][j] = cof

    plot_corr(np.abs(correlation))
    # plot_corr(correlation)

def plot_corr(data):
    root = './project/DisenSemi/plot'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rcParams['axes.labelsize'] = 16
    # matplotlib.rcParams['xtick.labelsize'] = 16
    # matplotlib.rcParams['ytick.labelsize'] = 16
    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, cmap='YlGnBu')
    ax.add_patch(
     patches.Rectangle(
         (0, 0),
         32.0,
         32.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (32, 32),
         32.0,
         32.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (64, 64),
         32.0,
         32.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (96, 96),
         32.0,
         32.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )

    plt.subplots_adjust(top=0.975, right=0.99, left=0.09, bottom=0.09)
    plt.savefig(os.path.join(root, 'fig8_b.pdf'))
    plt.close()

process()