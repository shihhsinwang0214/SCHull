### Based on the code in https://github.com/divelab/DIG/tree/dig-stable/dig/threedgraph

from qm9_dataset import QM93D
import sys
sys.path.append('/root/workspace/UnitSphere/models')
from leftnet import LEFTNet
from schnet import SchNet
from schnetCHA import SchNetCHA
from comenet import ComENet
from comenetCHA import ComENetCHA
from spherenet import SphereNet
from spherenetCHA import SphereNetCHA
from leftnetCHA import LEFTNetCHA
from dimenetpp import DimeNetPP
from dimenetppCHA import DimeNetPPCHA

import argparse
import os
import torch
from torch.optim import Adam
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingWarmRestarts
from tqdm import tqdm
import time
import wandb

# Hartree = 2.72114e-5

def func_std_mae(out, target, loss_func, istest=True, name_=None):    
    loss = loss_func(out, target)
    if istest:
        err_arr = torch.mean(torch.abs(out - target), dim=0)
    else:
        err_arr = None
    return loss, err_arr
        
def run(device, train_dataset, valid_dataset, test_dataset, model, scheduler_name, loss_func, epochs=800, batch_size=32, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
    save_dir='models/', log_dir='', disable_tqdm=False, cfg=None):     

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name == 'steplr':
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
    elif scheduler_name == 'onecyclelr':
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs) 

    best_valid = float('inf')
    test_valid = float('inf')
        
    if save_dir != '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if log_dir != '':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
    
    start_epoch = 1
    
    for epoch in range(start_epoch, epochs + 1):
        print("=====Epoch {}".format(epoch), flush=True)
        t_start = time.perf_counter()
        
        train_mae = train(model, optimizer, scheduler, scheduler_name, train_loader, loss_func, device, disable_tqdm, cfg)
        valid_mae, val_arr = val(model, valid_loader, loss_func, device, disable_tqdm, cfg)
        

        if log_dir != '':
            writer.add_scalar('train_mae', train_mae, epoch)
            writer.add_scalar('valid_mae', valid_mae, epoch)
        
        if valid_mae < best_valid:
            test_mae, _ = val(model, test_loader, loss_func, device, disable_tqdm, cfg)
            # writer.add_scalar('test_mae', test_mae, epoch)

            best_valid = valid_mae
            test_valid = test_mae
            best_val_arr = val_arr

            if save_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))

        t_end = time.perf_counter()
        print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae, 'Best valid': best_valid, 'Test@ best valid': test_valid, 'Duration': t_end-t_start})
        if cfg.trgt == 'ALL':
            wandb.log({
                'epoch': epoch,
                'train_mae': train_mae,
                'valid_mae': valid_mae,
                'test_mae': test_mae,
                'best_mae': best_valid,
                'test_best valid': test_valid,
                'best_val_mu': best_val_arr[0],
                'best_val_alpha': best_val_arr[1],
                'best_val_homo': best_val_arr[2],
                'best_val_lumo': best_val_arr[3],
                'best_val_gap': best_val_arr[4],
                'best_val_r2': best_val_arr[5],
                'best_val_zpve': best_val_arr[6],
                'best_val_U0': best_val_arr[7],
                'best_val_U': best_val_arr[8],
                'best_val_H': best_val_arr[9],
                'best_val_G': best_val_arr[10],
                'best_val_Cv': best_val_arr[11]
            })
        else:
            wandb.log({
                'epoch': epoch,
                '{}: train_mae'.format(cfg.trgt): train_mae,
                '{}: valid_mae'.format(cfg.trgt): valid_mae,
                '{}: test_mae'.format(cfg.trgt): test_mae,
                '{}: best_mae'.format(cfg.trgt): best_valid,
                '{}: test_best valid'.format(cfg.trgt): test_valid,
            })

        if scheduler_name == 'steplr':
            scheduler.step()

    print(f'Best validation MAE so far: {best_valid}')
    print(f'Test MAE when got best validation result: {test_valid}')
    
    if log_dir != '':
        writer.close()

def train(model, optimizer, scheduler, scheduler_name, train_loader, loss_func, device, disable_tqdm, cfg):  
    model.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader, disable=disable_tqdm)):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        if cfg.trgt == 'ALL':
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
        else:
            target = batch_data[cfg.trgt].unsqueeze(1)
        loss, _ = func_std_mae(out, target, loss_func, istest=False)

        if torch.isnan(loss):
            print(loss)
            exit()

        loss.backward()
        optimizer.step()
        if scheduler_name == 'onecyclelr':
            scheduler.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)

def val(model, data_loader, loss_func, device, disable_tqdm, cfg):   
    model.eval()
    loss_accum = 0

    for step, batch_data in enumerate(tqdm(data_loader, disable=disable_tqdm)):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        if cfg.trgt == 'ALL':
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
        else:
            target = batch_data[cfg.trgt].unsqueeze(1)
        loss, mae_arr = func_std_mae(out, target, loss_func, istest=True)

        if torch.isnan(loss):
            print(loss)
            exit()

        if step == 0:
            mae_arr_accu = mae_arr.cpu()
        else:
            mae_arr_accu += mae_arr.cpu()
        loss_accum += loss.detach().cpu().item()
        
    std_mae = loss_accum / (step + 1)
    return std_mae, mae_arr_accu / (step + 1)


parser = argparse.ArgumentParser(description='QM9')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--target', type=str, default='U0')

parser.add_argument('--train_size', type=int, default=80000)
parser.add_argument('--valid_size', type=int, default=25000)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--cutoff', type=float, default=5.0)
parser.add_argument('--num_radial', type=int, default=32)



parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--vt_batch_size', type=int, default=64)

parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--lr', type=float, default=7.5e-4)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=60)
parser.add_argument('--weight_decay', type=float, default=5e-7)

parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--disable_tqdm', default=False, action='store_true')
parser.add_argument('--scheduler', type=str, default='steplr')
parser.add_argument('--norm_label', default=False, action='store_true')

parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hull_cos', default=False, action='store_true')
parser.add_argument('--isangle_emb_hull', default=False, action='store_true')
parser.add_argument('--cha_rate', type=float, default=4/8)
parser.add_argument('--cha_scale', type=float, default=1)

parser.add_argument('--hidden_channels', type=int, default=110)
parser.add_argument('--out_channels', type=int, default=1)

parser.add_argument('--trgt', type=str, default='ALL', help = ['ALL', 
                                                               'mu', 'alpha', 
                                                               'homo', 'lumo', 
                                                               'gap', 'r2', 
                                                               'zpve','U0', 
                                                               'U', 'H', 
                                                               'G', 'Cv'])
parser.add_argument('--model_name', default='schnetCHA', help=['schnet', 'schnetCHA',
                                                                  'comenet', 'comenetCHA', 
                                                                  'LeftNet', 'leftnetCHA',
                                                                  'sphereNet', 'sphereNetCHA',
                                                                  'dimenetpp', 'dimenetppCHA'])
parser.add_argument('--exp_id', default=1)
args = parser.parse_args()

print(args)
print(args.save_dir)
if args.trgt == 'ALL':
    args.out_channels = 12
if args.model_name in ['comenetCHA_new', 'schnetCHA', 'leftnetCHA', 'sphereNetCHA', 'dimenetppCHA']:
    name_ = '{}_lyr{}_isangle_emb_hull{}_cha{:.2}_{}'.format(args.model_name, args.num_layers, 
                                                             args.isangle_emb_hull, args.cha_rate,  
                                                             args.cha_scale)
else:
    name_ = '{}_lyr{}'.format(args.model_name, args.num_layers)

kwargs = {
        'entity': 'utah-math-data-science', 
        'project': 'Unit_Sphere_QM9_V1(ComeNet_Fea)',
        'mode': 'disabled',
        'name': name_,
        'config': args,
        'settings': wandb.Settings(_disable_stats=True), 'reinit': True
        }
wandb.init(**kwargs)
wandb.save('*.txt')

dataset = QM93D(root='/root/workspace/A_data/qm93d/dataset')
target = args.target
dataset.data.y = dataset.data[target]
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=args.train_size, valid_size=args.valid_size, seed=args.seed)
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
print('train, validaion, test:', len(train_dataset), len(valid_dataset), len(test_dataset))

if args.norm_label:
    y_mean = torch.mean(train_dataset.data.y).item()
    y_std = torch.std(train_dataset.data.y).item()
    print('y_mean, y_std:', y_mean, y_std)
else:
    y_mean = 0
    y_std = 1


if args.model_name == 'schnet':
    model = SchNet(energy_and_force=False, 
                   cutoff=args.cutoff, 
                   num_layers=args.num_layers,
                   hidden_channels=args.hidden_channels*2,
                   out_channels=args.out_channels)
elif args.model_name == 'schnetCHA':
    model = SchNetCHA(energy_and_force=False, 
                      cutoff=args.cutoff, 
                      num_layers=args.num_layers,
                      hidden_channels=args.hidden_channels,
                      out_channels=args.out_channels,
                      cha_rate = args.cha_rate,
                      cha_scale = args.cha_scale,
                      hull_cos = args.hull_cos)

elif args.model_name == 'comenet':
    model = ComENet(cutoff=args.cutoff, 
                    num_layers=args.num_layers,
                    hidden_channels=args.hidden_channels,
                    out_channels=args.out_channels,
                    iscovhull=False)
    
elif args.model_name == 'comenetCHA':
    model = ComENetCHA(cutoff=args.cutoff, 
                    num_layers=args.num_layers,
                    hidden_channels=args.hidden_channels,
                    out_channels=args.out_channels,
                    cha_rate = args.cha_rate,
                    cha_scale = args.cha_scale,
                    hull_cos = args.hull_cos,
                    isangle_emb_hull = args.isangle_emb_hull
                    )
    
elif args.model_name == 'LeftNet':
    model = LEFTNet(pos_require_grad=False, 
                    cutoff=args.cutoff, 
                    num_layers=args.num_layers,
                    hidden_channels=args.hidden_channels, 
                    out_channels=args.out_channels,
                    num_radial=args.num_radial, 
                    y_mean=y_mean, y_std=y_std)
    
elif args.model_name == 'leftnetCHA':
    model = LEFTNetCHA(pos_require_grad=False, 
                       cutoff=args.cutoff, 
                       num_layers=args.num_layers,
                       hidden_channels=args.hidden_channels, 
                       out_channels=args.out_channels,
                       num_radial=args.num_radial, 
                       y_mean=y_mean, y_std=y_std,
                       cha_rate = args.cha_rate,
                       cha_scale = args.cha_scale,
                       hull_cos=False,
                       isangle_emb_hull = args.isangle_emb_hull)

elif args.model_name == 'sphereNet':
    model = SphereNet(energy_and_force=False, 
                      cutoff=args.cutoff, 
                      num_layers=args.num_layers,
                      hidden_channels=args.hidden_channels,
                      out_channels=args.out_channels
                      )

elif args.model_name == 'sphereNetCHA':
    model = SphereNetCHA(energy_and_force=False, 
                      cutoff=args.cutoff, 
                      num_layers=args.num_layers,
                      hidden_channels=args.hidden_channels,
                      out_channels=args.out_channels,
                      cha_rate = args.cha_rate,
                      cha_scale = args.cha_scale,
                      hull_cos = args.hull_cos,
                      isangle_emb_hull = args.isangle_emb_hull
                      )

elif args.model_name == 'dimenetpp':
    model = DimeNetPP(
        energy_and_force=False, 
        cutoff=args.cutoff, 
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        out_channels=args.out_channels
    )

elif args.model_name == 'dimenetppCHA':
    model = DimeNetPPCHA(energy_and_force=False, 
                      cutoff=args.cutoff, 
                      num_layers=args.num_layers,
                      hidden_channels=args.hidden_channels,
                      out_channels=args.out_channels,
                      cha_rate = args.cha_rate,
                      cha_scale = args.cha_scale,
                      hull_cos = args.hull_cos,
                      isangle_emb_hull = args.isangle_emb_hull
                      )

loss_func = torch.nn.L1Loss()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print('device',device)
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = torch.nn.DataParallel(model)
model.to(device)

run(device=device, 
    train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, 
    model=model, scheduler_name=args.scheduler, loss_func=loss_func, 
    epochs=args.epochs, batch_size=args.batch_size, vt_batch_size=args.batch_size, 
    lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, 
    weight_decay=args.weight_decay, 
    save_dir=args.save_dir, log_dir=args.save_dir, disable_tqdm=args.disable_tqdm, cfg=args)