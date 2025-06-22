import torch
from opt import args
from utils import eva, target_distribution, calc_loss, index_generator, parse_minibatch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from model import MAGNN_nc_mb
import torch.sparse
from tqdm import tqdm
import pickle


def Train_model_DBLP(model, x, labels, g_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices, n_cluster, lr, in_dims,  device):
    DBLP_net = MAGNN_nc_mb(num_metapaths=3,
                    num_edge_type=6,
                    etypes_list=[[0, 1], [0, 2, 3, 1], [0, 4, 5, 1]],
                    feats_dim_list=in_dims,
                    hidden_dim=args.hidden_dim,
                    out_dim=args.out_dim,
                    num_heads=args.num_heads,
                    attn_vec_dim=args.attn_vec_dim,
                    rnn_type=args.rnn_type).to(device)
    
    with open('data/preprocessed/DBLP_processed/pre_batch2048_results.pkl', 'rb') as file:
        results = pickle.load(file)

    with torch.no_grad():
        Z_g_tmp = []
        for iteration_result in results:

            iteration, data_g_list, data_indices_list, data_idx_batch_mapped_list = iteration_result
            
            D_embedding = DBLP_net(
                        data_g_list, features_list, type_mask, data_indices_list, data_idx_batch_mapped_list)
            Z_g_tmp.append(D_embedding)

        Z_g = torch.cat(Z_g_tmp, 0)

        x_hat, Z_a, Z_g, S, q = model(x, Z_g)

    # ----------------------initialize k-means----------------------
    features = S.data.cpu().detach()
    kmeans_ = KMeans(n_clusters=n_cluster, n_init=20)
    _ = kmeans_.fit_predict(features)
    model.cluster_layer.data = torch.tensor(kmeans_.cluster_centers_).to(device)


    parameters = list(DBLP_net.parameters()) + list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)

    plt_loss = []
    acc_all = []
    nmi_all = []
    x_list = []

    for epoch in range(args.epoch):
    # for epoch in tqdm(range(args.epoch)):
        model.train()
        optimizer.zero_grad()


        info_loss = torch.tensor(0.0, requires_grad=True).to(device)

        Z_g_tmp = []
        for iteration_result in results:

            iteration, data_g_list, data_indices_list, data_idx_batch_mapped_list = iteration_result
            
            D_embedding = DBLP_net(
                        data_g_list, features_list, type_mask, data_indices_list, data_idx_batch_mapped_list)
            Z_g_tmp.append(D_embedding)

        Z_g = torch.cat(Z_g_tmp, 0)

        x_hat, Z_a, Z_g, S, q = model(x, Z_g)
        
        tmp_q = q.data
        p = target_distribution(tmp_q)
    
        # --------------------------loss--------------------------
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        info_loss = args.lambda1_value*calc_loss(S, Z_a,temperature=args.temperature) + args.lambda2_value*calc_loss(S, Z_g,temperature=args.temperature)
        construction_loss = F.mse_loss(x_hat, x)
        model_loss = construction_loss+ info_loss +  args.lambda3_value*kl_loss

        plt_loss.append(model_loss.cpu().detach())
        x_list.append(epoch+1)

        model_loss.backward()
        optimizer.step()

        # --------------------------eval--------------------------
        # if epoch % 100 ==0:

        if epoch % 10 ==0:
            kmeans_ = KMeans(n_clusters=args.n_cluster, n_init=20).fit(S.data.cpu().numpy())
            y_pred = kmeans_.predict(S.data.cpu().numpy())
            acc, nmi, ari, f1 = eva(labels, y_pred, epoch)
            acc_all.append(acc)
            nmi_all.append(nmi)