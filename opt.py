import argparse
import pandas as pd
import os
from utils import setup_seed
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser(description='KAC')
# parser.add_argument('--name', type=str, default='IMDB', help='dataset.')
# parser.add_argument('--n_cluster', type=int, default=3, help='number of cluster.')

parser.add_argument('--name', type=str, default='DBLP', help='dataset.')
parser.add_argument('--n_cluster', type=int, default=4, help='number of cluster.')

# parser.add_argument('--name', type=str, default='Netease', help='dataset.')
# parser.add_argument('--n_cluster', type=int, default=19, help='number of cluster.')

parser.add_argument('--batch-size', type=int, default=2048, help='Batch size. Default is 8.') 
parser.add_argument('--samples', type=int, default=100, help='Number of neighbors sampled. Default is 100.')
# Note that batch_size and samples are fixed in Pre_batch.py, if want to change, run Pre_batch.py again.

parser.add_argument('--lambda1_value', type=float, default=0.4, help='trade-off factor.') 
parser.add_argument('--lambda2_value', type=float, default=0.6, help='trade-off factor.')
parser.add_argument('--lambda3_value', type=float, default=1.0, help='trade-off factor.')


parser.add_argument('--dropout_rate', type=float, default=0.5, help='dropout_rate.')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate.') 
parser.add_argument('--epoch', type=int, default=500, help='Number of epochs. Default is 100.')
parser.add_argument('--temperature', type=float, default=1.0, help='temperature.')
# ap.add_argument('--n_components', type=int, default=100)

parser.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
parser.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
parser.add_argument('--out_dim', type=int, default=512, help='Z_g/Z_a dimension.') 
parser.add_argument('--layers', type=int, default=1, help='Number of layers. Default is 2.') 
parser.add_argument('--attn-vec-dim', type=int, default=128, help='Dimension of the attention vector. Default is 128.')



parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay.')
parser.add_argument('--feats-type', type=int, default=0,
                help='Type of the node features used. ' +
                        '0 - loaded features; ' +
                        '1 - only target node features (zero vec for others); ' +
                        '2 - only target node features (id vec for others); ' +
                        '3 - all id vec. Default is 2.')
parser.add_argument('--rnn-type', default='RotatE0', help='Type of the aggregator. Default is RotatE0.')
parser.add_argument('--patience', type=int, default=10, help='Patience. Default is 10.')
parser.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
parser.add_argument('--save-postfix', default='IMDB', help='Postfix for the saved model and result. Default is IMDB.')

args = parser.parse_args()
