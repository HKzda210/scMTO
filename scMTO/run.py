import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import random
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scMTO.preprocess import prepro, normalize
from scMTO.model import scMTO
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
cudnn.enabled =  True
torch.set_num_threads(2)


if __name__ == "__main__":
    import torch
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Quake_10x_Spleen', help="name of scRNA-seq dataset")
    parser.add_argument('--pre_lr', type=float, default=4e-4, help="learning rate of pre-training")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate of formal-training")
    parser.add_argument('--pre_epoch', type=int, default=250, help="epoch numbers of pre-training")
    parser.add_argument('--train_epoch', type=int, default=500, help="epoch numbers of formal-training")
    parser.add_argument('--latent_dim', default=32, type=int, help="dimension of latent space")
    parser.add_argument('--device', type=int, default=3)
    args = parser.parse_args()

    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    np.random.seed(1)

    args.cuda = torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    device = torch.device("cuda" if args.cuda else "cpu")

    file_path = "data/" + args.name + "/data.h5"
    print("dataset: {}".format(args.name))

    # Read data
    print('Data reading...')
    x, y = prepro(file_path)
    x = np.ceil(x).astype(np.float32)

    # Preprocess data 
    # Multi-View 1
    print('Data preprocessing...')
    adata1 = sc.AnnData(x)
    adata1.obs['Group'] = y
    adata1 = normalize(adata1, copy=True, highly_genes=2000, size_factors=True, normalize_input=True, logtrans_input=True)

    # Multi-View 2
    adata2 = sc.AnnData(x)
    adata2.obs['Group'] = y   
    adata2 = normalize(adata2, copy=True, highly_genes=500, size_factors=True, normalize_input=True, logtrans_input=True)

    # Multi-View 3 (sparse topic patterns) 
    count = [adata1.X, adata2.X]
    highly_genes_index = [int(gene_idx) for gene_idx in list(adata1.var.index)]
    raw_data = np.ceil(adata1.raw.X[:, highly_genes_index]).astype(int)
    args.n_clusters = int(max(y) - min(y) + 1)

    # Initialize model 
    model = scMTO(n_z=args.latent_dim, n_clusters=args.n_clusters, x_raw=raw_data, device=device).to(device)

    # Pretraining stage
    model.pretrain(x=count, raw_data=raw_data, pre_lr=args.pre_lr, pre_epoch=args.pre_epoch)

    # Clustering stage
    z, w, h, pred, acc, nmi, ari = model.fit(x=count, y=y, raw_data=raw_data, lr=args.lr, train_epoch=args.train_epoch)
    
    # Cluster performance evaluated by ACC, NMI, ARI metrics
    if y is not None:
        print(f'The End: ACC {acc:.4f}, NMI {nmi:.4f}, ARI {ari:.4f}')



















