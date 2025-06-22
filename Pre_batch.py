import torch

from Load_data import load_DBLP_data, load_Netease_data
from utils import index_generator, parse_minibatch
from opt import args
import pickle
import warnings
warnings.filterwarnings('ignore')


def run_pre_batch(nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, data_sequence_idx):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_idx_generator = index_generator(batch_size=args.batch_size, indices=data_sequence_idx, shuffle=False)
    pre_batch_results = []

    print('Begin!')

    for iteration in range(data_idx_generator.num_iterations()):
        data_idx_batch = data_idx_generator.next()
        data_idx_batch.sort()
        data_g_list, data_indices_list, data_idx_batch_mapped_list = parse_minibatch(
            nx_G_lists, edge_metapath_indices_lists, data_idx_batch, device, args.samples)

        data_g_list = [g.to(device) for g in data_g_list]

        pre_batch_results.append((iteration, data_g_list, data_indices_list, data_idx_batch_mapped_list))
    
    # return pre_batch_results

    print('Process successful!')

    # with open('data/preprocessed/DBLP_processed/pre_batch128_results.pkl', 'wb') as file:
    with open('data/preprocessed/Netease_processed/pre_batch2048_results.pkl', 'wb') as file:
        pickle.dump(pre_batch_results, file)




if __name__ == '__main__':
    if args.name =='DBLP':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels = load_DBLP_data()
        data_sequence_idx = list(range(4057))

    elif args.name =='Netease':
        nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels = load_Netease_data()
        data_sequence_idx = list(range(10793))

        # print(len(edge_metapath_indices_lists))
        # print(edge_metapath_indices_lists[0][4052].dtype)
        # exit()

    run_pre_batch(nx_G_lists, edge_metapath_indices_lists, features_list, adjM, type_mask, labels, data_sequence_idx)

