from qm9_dataset import QM93D, QM93D_old
from leftnet import LEFTNet
from schnet import SchNet
from schnetCHA import SchNetCHA, SchNet_MS, SchNetCHA_Pure
from comenet import ComENet
from comenetCHA import ComENetCHA

import argparse
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import time
import pandas as pd


def statistics_qm9(args):
    if args.dataset == 0:
        dataset = QM93D(root='/root/workspace/A_data/qm93d/dataset')
    elif args.dataset == 1:
        dataset = QM93D_old(root='/root/workspace/A_data/qm93d/dataset2')
    
    target = args.target
    dataset.data.y = dataset.data[target]
    split_idx = dataset.get_idx_split(len(dataset.data.y), 
                                      train_size=args.train_size, 
                                      valid_size=args.valid_size, 
                                      seed=args.seed)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.vt_batch_size, shuffle=False)

    # statistics on training set
    for i, batch_data in enumerate(tqdm(train_loader, disable=False)):
        target = torch.cat([batch_data.mu.unsqueeze(1),
                            batch_data.alpha.unsqueeze(1),
                            batch_data.homo.unsqueeze(1),
                            batch_data.lumo.unsqueeze(1),
                            batch_data.gap.unsqueeze(1),
                            batch_data.r2.unsqueeze(1),
                            batch_data.zpve.unsqueeze(1),
                            batch_data.U0.unsqueeze(1),
                            batch_data.U.unsqueeze(1),
                            batch_data.H.unsqueeze(1),
                            batch_data.G.unsqueeze(1),
                            batch_data.Cv.unsqueeze(1),],
                            dim=1)
        batch_abs_mean = torch.mean(torch.abs(target), dim=0)
        batch_abs_std = torch.std(torch.abs(target), dim=0)
        if i == 0:
            data_abs_mean = batch_abs_mean
            data_abs_std = batch_abs_std
        else:
            data_abs_mean += batch_abs_mean
            data_abs_std += batch_abs_std
    
    data_abs_mean /= (i+1)
    data_abs_std /= (i+1)
    
    statistics_dict = {}
    name_arr = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 
                        'zpve','U0', 'U', 'H', 'G', 'Cv']
    for k in range(12):
        temp = {
            'abs_mean': data_abs_mean[k].cpu().item(),
            'abs_std': data_abs_std[k].cpu().item()
        }
        statistics_dict[name_arr[k]] = temp

    df = pd.DataFrame.from_dict(statistics_dict, orient='index',
                                columns=['abs_mean', 'abs_std'])
    df.to_csv('statistics_qm93d_{}.csv'.format(args.dataset))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QM9')
    parser.add_argument('--dataset', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_size', type=int, default=128000)
    parser.add_argument('--valid_size', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--vt_batch_size', type=int, default=64)
    parser.add_argument('--target', type=str, default='U0')
    args = parser.parse_args()
    statistics_qm9(args)
