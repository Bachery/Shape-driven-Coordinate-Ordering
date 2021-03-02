import argparse
import datetime
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import shape
from shape import ShapeDataset, MixShapeDataset
from shape_train import RnnModel_fc, RnnModel_fcn, RnnModel_fcn2, RnnModel_n2d, RnnModel_d2n

def str2bool(v):
      return v.lower() in ('true', '1')


def validate(net, data_loader, render_fn, net_type, use_cuda, save_dir ):
    """Used to monitor progress on a validation set & optionally plot solution."""

    # if render_fn is not None:
    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)
    net.eval()

    for batch_idx, batch in enumerate(data_loader):
        data, points, gt = batch
        # points, gt = batch
        if use_cuda:
            data = data.cuda()
            points = points.cuda()
            gt = gt.cuda().detach()

        # Full forward pass through the dataset
        with torch.no_grad():
            if net_type == 'fc':
                output = net(data)
            else:
                output = net(points)

        if render_fn is not None:
            name = 'net_%03d.png'%(batch_idx)
            path = os.path.join(save_dir, name)
            # render_fn(data, output, gt, path )
            render_fn(data, output, gt, path )

    net.train()

def train(net, train_data, valid_data, 
            batch_size, lr, 
            max_grad_norm, epoch_num, 
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    date = datetime.datetime.now()
    now = '%s' % date.date()
    now += '-%s' % date.hour
    now += '-%s' % date.minute
    now = str(now)
    save_dir = os.path.join('shape', 'curve',
        str( kwargs['dim_num_max'] ) + 'd-' + kwargs['net_type'] + '-' + kwargs['multi_dim']  + '-note-' + kwargs['note'] ) #+ '-' + now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    net_optim = optim.Adam(net.parameters(), lr=lr)
    loss_function=nn.MSELoss()

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, len(valid_data), shuffle=False, num_workers=0)

    best_reward = np.inf
    my_rewards = []
    my_losses = []

    train_size = kwargs['train_size']
    log_step = int(train_size / batch_size)
    if log_step > 100:
        log_step = int(100)
    if log_step == 0:
        log_step = int(1)

    for epoch in range(epoch_num):

        net.train()

        times, losses, rewards = [], [], []

        epoch_start = time.time()
        start = epoch_start
        valid_dir = os.path.join(save_dir, 'render', '%s' % epoch)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)

        for batch_idx, batch in enumerate(train_loader):
            if kwargs['just_test'] == True:
                continue

            data, points, gt = batch
            # points, gt = batch
            use_cuda = kwargs['use_cuda']
            if use_cuda:
                data = data.cuda()
                points = points.cuda()
                gt = gt.cuda()

            # Full forward pass through the dataset
            if kwargs['net_type'] == 'fc':
                output = net(data)
            else:
                output = net(points)

            loss = loss_function( output, gt.detach() )
            net_optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            net_optim.step()

            losses.append(torch.mean(loss.detach()).item())

            if (batch_idx + 1) % log_step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-log_step:])
                my_losses.append(mean_loss)

                print('    Epoch %d  Batch %d/%d, loss: %2.4f, took: %2.4fs' %
                      (epoch, batch_idx, len(train_loader), mean_loss,
                       times[-1]))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'net.pt')
        torch.save(net.state_dict(), save_path)

        # Save rendering of validation set dims
        valid_dir = os.path.join(save_dir, 'render', '%s' % epoch)

        validate(
            net,
            valid_loader,
            render_fn=kwargs['render_fn'],
            net_type=kwargs['net_type'],
            use_cuda=kwargs['use_cuda'],
            save_dir=valid_dir
        )

        print('Epoch %d,  mean epoch loss: %2.4f, took: %2.4fs '\
              '(%2.4fs / %d batches) ' % \
              (epoch, mean_loss, time.time() - epoch_start, np.mean(times), log_step  ) )
            #   '(%2.4fs / %d batches)  | shape: %s' % \
            #   (epoch, mean_loss, time.time() - epoch_start, np.mean(times), log_step, data.shape  ) )

        plt.close('all')
        plt.title('Loss')
        plt.plot(range(len(my_losses)), my_losses, '-')
        plt.savefig(save_dir + '/loss.png' , bbox_inches='tight', dpi=400)

    np.savetxt(save_dir + '/losses.txt', my_losses)

def load_and_train(args):

    def num_range_list( num_min, num_max, num_step ):
        if num_max == num_min:
            return [num_max]
        return [
            i for i in range(
                num_min, num_max+1, num_step
            )
        ]

    dim_num_range = num_range_list( args.dim_num_min, args.dim_num_max, args.dim_num_step )
    use_cuda = args.use_cuda

    print('Loading data...')

    total_type_num = len(dim_num_range)
    each_train_size = int( args.train_size / total_type_num )
    each_valid_size = int( args.valid_size / total_type_num )

    train_files, valid_files = shape.create_mix_dataset(
        dim_num_range,
        each_train_size,
        each_valid_size,
        args.sample_num,
        args.standard,
        seed=args.seed,
    )
    if args.just_generate == True:
        return

    train_data = MixShapeDataset(  train_files, args.batch_size, args.train_size, args.seed )
    valid_data = MixShapeDataset(  valid_files, args.valid_size, args.valid_size, args.seed + 1 )

    input_size = 2
    if args.net_type == 'n2d':
        RnnModel = RnnModel_n2d
    elif args.net_type == 'd2n':
        RnnModel = RnnModel_d2n
    elif args.net_type == 'fc':
        RnnModel = RnnModel_fc
    elif args.net_type == 'fcn':
        RnnModel = RnnModel_fcn
    elif args.net_type == 'fcn2':
        RnnModel = RnnModel_fcn2
        input_size = 4

    net = RnnModel(
        input_size,
        args.hidden_size,
        args.num_layers,
        args.dropout,
    )

    if use_cuda:
        net = net.cuda()

    kwargs = vars(args)
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['render_fn'] = shape.render


    multi_str = ''
    if len(dim_num_range) > 1:
        multi_str += 'M'
    else:
        multi_str += 'S'

    kwargs['multi_dim'] = multi_str
    print(multi_str)


    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'net.pt')
        net.load_state_dict(torch.load(path))

        print('Loading pre-train model', path)
    
    train(net, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--epoch_num', default=1, type=int)

    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--cuda', default='0', type=str)

    parser.add_argument('--train_size',default=32, type=int)
    parser.add_argument('--valid_size', default=32, type=int)
    
    parser.add_argument('--net_type', default='fc', type=str)
    
    parser.add_argument('--dim_num_min', default=16, type=int)
    parser.add_argument('--dim_num_max', default=16, type=int)
    parser.add_argument('--dim_num_step', default=2, type=int)
    
    parser.add_argument('--sample_num', default=80, type=int)
    parser.add_argument('--standard', default=0.1, type=float)

    parser.add_argument('--hidden_size', default=128, type=int)

    parser.add_argument('--just_test', default=False, type=str2bool)
    parser.add_argument('--just_generate', default=False, type=str2bool)

    parser.add_argument('--note', default='debug', type=str)

    args = parser.parse_args()
    print('Sample num:       %s' % args.sample_num)        
    print('Dim min num:      %s' % args.dim_num_min)        
    print('Dim max num:      %s' % args.dim_num_max)        
    print('Net type:         %s' % args.net_type)        
    print('Note:             %s' % args.note)
    if args.use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    load_and_train(args)
