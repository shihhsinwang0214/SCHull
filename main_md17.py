from md17_dataset import MD17
import sys
sys.path.append('/root/workspace/UnitSphere/models')
from leftnet import LEFTNet
from leftnetCHA import LEFTNetCHA
import sys, os
import argparse
import os
import torch
from torch.optim import Adam,AdamW
from torch_geometric.data import DataLoader
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingLR
from tqdm import tqdm
import time
import wandb

def run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, eval_steps=50, eval_start=0,
        epochs=800, batch_size=4, vt_batch_size=32, lr=0.0005, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0,
        energy_and_force=True, p=100, save_dir='models/', log_dir='', cfg=None):
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print('num_parameters:', num_params)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)
    best_valid = float('inf')
    test_valid = float('inf')
    best_val_force_mae = float('inf')
    valid_test_force_mae = float('inf')
    test_force_mae = float('inf')
    start_epoch = 1

    if save_dir != '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    if log_dir != '':
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(start_epoch, epochs + 1):
        print("=====Epoch {}".format(epoch), flush=True)
        t_start = time.perf_counter()
        test_mae = float('inf')

        train_mae = train(model, optimizer, train_loader, energy_and_force, p, loss_func, device)
        t_end = time.perf_counter()
        valid_mae, val_force_mae = val(model, valid_loader, energy_and_force, p, device, cfg)
        # if epoch > eval_start and epoch % eval_steps == 0: 
        #     print('Testing')
        #     test_mae, test_force_mae = val(model, test_loader, energy_and_force, p, device)

        if log_dir != '':
            writer.add_scalar('train_mae', train_mae, epoch)
            writer.add_scalar('valid_mae', valid_mae, epoch)
            writer.add_scalar('test_mae', test_mae, epoch)
        
        # if epoch > eval_start and epoch % eval_steps == 0: 
        #     print('Testing')
        #     test_mae, test_force_mae = val(model, test_loader, energy_and_force, p, device, cfg)

        if val_force_mae < best_val_force_mae:
        # if valid_mae < best_valid:
            if epoch > eval_start:
                print('Testing')
                test_mae, test_force_mae = val(model, test_loader, energy_and_force, p, device, cfg)
            best_valid = valid_mae
            test_valid = test_mae
            best_val_force_mae = val_force_mae
            valid_test_force_mae = test_force_mae
            if save_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid,
                              'num_params': num_params}
                torch.save(checkpoint, os.path.join(save_dir, 'valid_checkpoint.pt'))
        
        
        print({'Train': train_mae, 'Validation': valid_mae, 
               'Test': test_mae, 'Best valid': best_valid, 
               'Duration': t_end-t_start})
        
        wandb.log({
                'epoch': epoch,
                '{}(p={}): train_mae'.format(cfg.name, p): train_mae,
                '{}(p={}): valid_mae'.format(cfg.name, p): valid_mae,
                # '{}: test_mae'.format(cfg.name): test_mae,
                '{}(p={}): val_force_mae'.format(cfg.name, p): val_force_mae,
                # '{}: test_best valid'.format(cfg.name): test_valid,
                '{}(p={}): best_val_force_mae'.format(cfg.name, p): best_val_force_mae,
                '{}(p={}): valid_test_force_mae'.format(cfg.name, p): valid_test_force_mae,
                '{}(p={}): duration'.format(cfg.name, p): t_end-t_start,
            })

        scheduler.step()

    print(f'Best validation MAE so far: {best_valid}')
    print(f'Test MAE when got best validation result: {test_valid}')

    if log_dir != '':
        writer.close()


def train(model, optimizer, train_loader, energy_and_force, p, loss_func, device):
    model.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader, disable=False)):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out,forces = model(batch_data)
        NUM_ATOM = batch_data.force.size()[0]

        out = out * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM
        forces = forces * FORCE_MEAN_TOTAL
        if energy_and_force:
            force = -grad(outputs=out, inputs=batch_data.posc, grad_outputs=torch.ones_like(out), create_graph=True,
                                         retain_graph=True)[0] + forces/1000
            e_loss = loss_func(out, batch_data.y.unsqueeze(1))
            f_loss = loss_func(force, batch_data.force)
            loss = e_loss + p * f_loss
        else:
            loss = loss_func(out, batch_data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        loss_accum += loss.detach().cpu().item()
        
    return loss_accum / (step + 1)


def val(model, data_loader, energy_and_force, p, device, cfg):
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)

    if energy_and_force:
        preds_force = torch.Tensor([]).to(device)
        targets_force = torch.Tensor([]).to(device)

    for step, batch_data in enumerate(tqdm(data_loader, disable=False)):
        batch_data = batch_data.to(device)
        out, forces = model(batch_data)

        out = out * FORCE_MEAN_TOTAL + ENERGY_MEAN_TOTAL * NUM_ATOM
        forces = forces * FORCE_MEAN_TOTAL

        if energy_and_force:
            force = -grad(outputs=out, inputs=batch_data.posc, grad_outputs=torch.ones_like(out), create_graph=True,
                        retain_graph=True)[0] + forces/1000
            if torch.sum(torch.isnan(force)) != 0:
                mask = torch.isnan(force)
                force = force[~mask].reshape((-1, 3))
                batch_data.force = batch_data.force[~mask].reshape((-1, 3))
            preds_force = torch.cat([preds_force, force.detach()], dim=0)
            targets_force = torch.cat([targets_force, batch_data.force], dim=0)

        preds = torch.cat([preds, out.detach()], dim=0)
        targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

        if step > 2500:
            break

    if energy_and_force:
        energy_mae = torch.mean(torch.abs(preds - targets)).cpu().item()
        force_mae = torch.mean(torch.abs(preds_force - targets_force)).cpu().item()
        print({'Energy MAE': energy_mae, 'Force MAE': force_mae})
        return energy_mae + p * force_mae, force_mae

    return torch.mean(torch.abs(preds - targets)).cpu().item(), None



parser = argparse.ArgumentParser(description='MD17')
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--eval_steps', type=int, default=10)
parser.add_argument('--eval_start', type=int, default=500)
parser.add_argument('--epochs', type=int, default=1200)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--vt_batch_size', type=int, default=32)

parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=180)

parser.add_argument('--p', type=int, default=100)

parser.add_argument('--cutoff', type=float, default=8)
parser.add_argument('--num_layers', type=int, default=4)
parser.add_argument('--hidden_channels', type=int, default=200)
parser.add_argument('--num_radial', type=int, default=32)

parser.add_argument('--hull_cos', default=False, action='store_true')
parser.add_argument('--isangle_emb_hull', default=False, action='store_true')
parser.add_argument('--cha_rate', type=float, default=7/8)
parser.add_argument('--cha_scale', type=float, default=1)

parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--name', type=str, default='uracil') #aspirin, benzene2017, ethanol, malonaldehyde, naphthalene, salicylic, toluene, uracil
parser.add_argument('--model_name', default='leftnetCHA', help=['LeftNet', 'leftnetCHA'])
args = parser.parse_args()
print(args)

if args.model_name in ['comenetCHA_new', 'schnetCHA', 'leftnetCHA', 'sphereNetCHA', 'dimenetppCHA']:
    name_ = '{}_lyr{}_isangle_emb_hull{}_cha{:.2}_{}'.format(args.model_name, args.num_layers, 
                                                             args.isangle_emb_hull, args.cha_rate,  
                                                             args.cha_scale)
else:
    name_ = '{}_lyr{}'.format(args.model_name, args.num_layers)
kwargs = {
        'entity': 'utah-math-data-science', 
        'project': 'Unit_Sphere_MD17_V1',
        'mode': 'online',
        'name': name_,
        'config': args,
        'settings': wandb.Settings(_disable_stats=True), 'reinit': True
        }
wandb.init(**kwargs)
wandb.save('*.txt')

dataset = MD17(name=args.name, root = '/root/workspace/A_data/md17/dataset')
split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=1000, valid_size=1000, seed=42)
y_mean = None
y_std = None
train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
print('device', device)

y_mean = 0
y_std = 1

force_mean = 0

ENERGY_MEAN_TOTAL = 0
FORCE_MEAN_TOTAL = 0
NUM_ATOM = None
for data in train_dataset:
    energy = data.y
    force = data.force
    NUM_ATOM = force.size()[0]
    energy_mean = energy / NUM_ATOM
    ENERGY_MEAN_TOTAL += energy_mean
    force_rms = torch.sqrt(torch.mean(force.square()))
    FORCE_MEAN_TOTAL += force_rms
ENERGY_MEAN_TOTAL /= len(train_dataset)
FORCE_MEAN_TOTAL /= len(train_dataset)
ENERGY_MEAN_TOTAL = ENERGY_MEAN_TOTAL.to(device)
FORCE_MEAN_TOTAL = FORCE_MEAN_TOTAL.to(device)


if args.model_name == 'LeftNet':
    model = LEFTNet(pos_require_grad=True, 
                    cutoff=args.cutoff, 
                    num_layers=args.num_layers,
                    hidden_channels=args.hidden_channels, 
                    num_radial=args.num_radial, 
                    y_mean=y_mean, y_std=y_std)
    
elif args.model_name == 'leftnetCHA':
    model = LEFTNetCHA(pos_require_grad=True, 
                       cutoff=args.cutoff, 
                       num_layers=args.num_layers,
                       hidden_channels=args.hidden_channels, 
                       num_radial=args.num_radial, 
                       y_mean=y_mean, y_std=y_std,
                       cha_rate = args.cha_rate,
                       cha_scale = args.cha_scale,
                       hull_cos=False,
                       isangle_emb_hull = args.isangle_emb_hull)

loss_func = torch.nn.L1Loss()

run(device, train_dataset, valid_dataset, test_dataset, model, loss_func, 
eval_steps=args.eval_steps, eval_start=args.eval_start,
epochs=args.epochs, batch_size=args.batch_size, vt_batch_size=args.vt_batch_size,
lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size,
p=args.p, save_dir=args.save_dir, cfg=args)