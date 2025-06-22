import warnings
import torch
from opt import args
import torch.nn.functional as F
import torch.sparse
import numpy as np
import os
import dgl
from Load_data import load_IMDB_data, load_DBLP_data, load_Netease_data
from model import KAC
from Train_model_DBLP import Train_model_DBLP
from Train_model_IMDB import Train_model_IMDB
from Train_model_Netease import Train_model_Netease
from utils import setup_seed
from sklearn.decomposition import PCA
import scipy.io as sio
from sklearn.cluster import KMeans
from utils import eva

warnings.filterwarnings('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

seed = 941
setup_seed(seed)
print('seed:{}'.format(seed))


def run_model(nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.name =='IMDB':
        features_list = [torch.FloatTensor(features.todense()).to(device) for features in features_list]
    elif args.name =='DBLP':
        features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    elif args.name =='Netease':
        features_list = [torch.FloatTensor(features).to(device) for features in features_list]

    in_dims = [features.shape[1] for features in features_list]

    x = features_list[0].to(device) #target samples representation


    x_dim = x.shape[1]
    

    labels = torch.LongTensor(labels).to(device)
    target_node_indices = np.where(type_mask == 0)[0]
    

    if args.name =='IMDB':
        edge_metapath_indices_lists = [[torch.LongTensor(indices).to(device) for indices in indices_list] for indices_list in
                                    edge_metapath_indices_lists]
        g_lists = []
        for nx_G_list in nx_G_lists:
            g_lists.append([])
            for nx_G in nx_G_list:
                g = dgl.DGLGraph(multigraph=True).to(device)
                g.add_nodes(nx_G.number_of_nodes())
                g.add_edges(*list(zip(*sorted(map(lambda tup: (int(tup[0]), int(tup[1])), nx_G.edges())))))
                g_lists[-1].append(g)

    model =KAC(x_dim=x_dim, n_cluster=args.n_cluster)
    model.to(device)


    print('data_name = {}, lr = {}, lambda1 = {}, lambda2 = {}, lambda3 = {}\n'.format(args.name, args.lr, args.lambda1_value, args.lambda2_value, args.lambda3_value))
    print('#---------------------------training---------------------------')
    if args.name =='IMDB':
        Train_model_IMDB(model, x, labels, g_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices, n_cluster=args.n_cluster, lr=args.lr, in_dims=in_dims, device=device)
    elif args.name =='DBLP':
        Train_model_DBLP(model, x, labels, nx_G_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices, n_cluster=args.n_cluster, lr=args.lr, in_dims=in_dims,  device=device)
    elif args.name =='Netease':
        Train_model_Netease(model, x, labels, nx_G_lists, features_list, type_mask, edge_metapath_indices_lists, target_node_indices, n_cluster=args.n_cluster, lr=args.lr, in_dims=in_dims,  device=device)
    print('#---------------------------finished---------------------------')
    

if __name__ == '__main__':
    if args.name =='IMDB':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels = load_IMDB_data()
        data_sequence_idx = []
    elif args.name =='DBLP':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels = load_DBLP_data()
    elif args.name =='Netease':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels = load_Netease_data()


    run_model(nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels)


