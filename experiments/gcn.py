import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from datasets import get_planetoid_dataset
from train_eval import run

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--split', type=str, default='public')
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=0)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--logger', type=str, default=None)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--preconditioner', type=str, default=None)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--eps', type=float, default=0.01)
parser.add_argument('--update_freq', type=int, default=50)
parser.add_argument('--gamma', type=float, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--hyperparam', type=str, default=None)
args = parser.parse_args()

class Net_orig(torch.nn.Module):
    def __init__(self, dataset):
        super(Net2, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class CRD(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True) 
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x
    
class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.crd = CRD(dataset.num_features, args.hidden, args.dropout)
        self.cls = CLS(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x

dataset = get_planetoid_dataset(name=args.dataset, normalize_features=args.normalize_features, split=args.split)

kwargs = {
    'dataset': dataset, 
    'model': Net(dataset), 
    'str_optimizer': args.optimizer, 
    'str_preconditioner': args.preconditioner, 
    'runs': args.runs, 
    'epochs': args.epochs, 
    'lr': args.lr, 
    'weight_decay': args.weight_decay, 
    'early_stopping': args.early_stopping, 
    'logger': args.logger, 
    'momentum': args.momentum,
    'eps': args.eps,
    'update_freq': args.update_freq,
    'gamma': args.gamma,
    'alpha': args.alpha,
    'hyperparam': args.hyperparam
}

if args.hyperparam == 'eps':
    for param in np.logspace(-3, 0, 10, endpoint=True):
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        run(**kwargs)
elif args.hyperparam == 'update_freq':
    for param in [4, 8, 16, 32, 64, 128]:
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        run(**kwargs)
elif args.hyperparam == 'gamma':
    for param in np.linspace(1., 10., 10, endpoint=True):
        print(f"{args.hyperparam}: {param}")
        kwargs[args.hyperparam] = param
        run(**kwargs)
else:
    run(**kwargs)