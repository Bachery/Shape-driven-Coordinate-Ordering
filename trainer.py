import os
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from model import DRL, RnnEncoder, ConvEncoder, RnnEncoder_2, criticAttention, Critic

import vis
from vis import VISDataset, VISMDataset
import tools
import matplotlib.pyplot as plt

def str2bool(v):
      return v.lower() in ('true', '1')

def validate(data_loader, actor,
                reward_type, vis_type, 
                with_label, 
                # label_num,
                use_cuda, reward_fn, render_fn, 
                save_dir
            ):
    """Used to monitor progress on a validation set & optionally plot solution."""

    actor.eval()

    if render_fn is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        encoder_input, encoder_label, encoder_label_code, decoder_input, decoder_label, decoder_label_code, label_num = batch

        if use_cuda:
            encoder_input = encoder_input.cuda()
            encoder_label = encoder_label.cuda()
            encoder_label_code = encoder_label_code.cuda()
            decoder_input = decoder_input.cuda()
            decoder_label = decoder_label.cuda()
            decoder_label_code = decoder_label_code.cuda()

        label_num = int(label_num.data[0])
        import time
        # Full forward pass through the dataset
        with torch.no_grad():
            # t1 = time.time()
            # for i in range(100):
            dim_indices, dim_logp = actor(encoder_input, encoder_label_code, decoder_input, decoder_label_code)

            # print('time: ', (time.time() - t1)/100.0)
        # Sum the log probabilities for each dim
        reward = reward_fn(encoder_input, encoder_label, dim_indices, vis_type, reward_type, with_label, label_num=label_num)

        reward = reward.mean().item()
        rewards.append(reward)

        if render_fn is not None:
            name = 'net_%03d.png'%(batch_idx)
            path = os.path.join(save_dir, name)
            render_fn(encoder_input, encoder_label, dim_indices, path, vis_type, reward_type, with_label, label_num=label_num )

    actor.train()
    return np.mean(rewards)

def train(actor, critic, task, dim_num, train_data, valid_data, 
            reward_fn, render_fn, 
            batch_size, actor_lr, critic_lr, 
            max_grad_norm, epoch_num, 
          **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""

    date = datetime.datetime.now()
    now = '%s' % date.date()
    now += '-%s' % date.hour
    now += '-%s' % date.minute
    now = str(now)
    
    if kwargs['just_test'] == True:
        task = "vis_valid"

    save_dir = os.path.join(task, kwargs['vis_type'],   '%d' % dim_num, # 'curve', 
        kwargs['reward_type'] + '-' + kwargs['encoder_type'] + '-' + kwargs['data_type'] + '-' 
        + str(kwargs['with_label']) + '-' + str(kwargs['share_RNN']) + '-' 
        + '[%d-%d]d-' % ( kwargs['dim_num_min'], kwargs['dim_num_max'] ) 
        + '[%d-%d]n-' % ( kwargs['data_num_min'], kwargs['data_num_max'] ) 
        + '[%d-%d]c-' % ( kwargs['label_num_min'], kwargs['label_num_max'] ) 
        + kwargs['label_type'] + '-' + str(kwargs['standard']) + '-note-' + kwargs['note']   + '-' + now)

    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        os.makedirs(save_dir + '/img')

    if kwargs['optimizer'] == 'Adam':
        actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
        critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)
    else:
        actor_optim = optim.SGD(actor.parameters(), lr=actor_lr, momentum=0.9)
        critic_optim = optim.SGD(critic.parameters(), lr=critic_lr, momentum=0.9)


    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, len(valid_data), shuffle=False, num_workers=0)

    best_reward = np.inf
    my_rewards = []
    my_losses = []
    my_critic_losses = []

    train_size = kwargs['train_size']
    log_step = int(train_size / batch_size)
    if log_step > 100:
        log_step = int(100)
    if log_step == 0:
        log_step = int(1)
    
    # evaluate_tool = tools.EvaluateTools(kwargs['use_cuda'])
    evaluate_tool = tools.EvaluateTools(True)

    for epoch in range(epoch_num):

        actor.train()
        critic.train()

        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start
        valid_dir = os.path.join(save_dir, 'render', '%s' % epoch)
        if not os.path.exists(valid_dir):
            os.makedirs(valid_dir)

        for batch_idx, batch in enumerate(train_loader):
            if kwargs['just_test'] == True:
                continue
            encoder_input, encoder_label, encoder_label_code, \
                decoder_input, decoder_label, decoder_label_code, \
                label_num = batch

            use_cuda = kwargs['use_cuda']
            if use_cuda:
                encoder_input = encoder_input.cuda()
                encoder_label = encoder_label.cuda()
                encoder_label_code = encoder_label_code.cuda()
                decoder_input = decoder_input.cuda()
                decoder_label = decoder_label.cuda()
                decoder_label_code = decoder_label_code.cuda()

            label_num = int(label_num.data[0])

            # Full forward pass through the dataset
            dim_indices, dim_logp = actor(encoder_input, encoder_label_code, decoder_input, decoder_label_code)

            # Sum the log probabilities for each dim
            reward = reward_fn(encoder_input, encoder_label, dim_indices, kwargs['vis_type'], kwargs['reward_type'], 
                kwargs['with_label'], evaluate_tool, 
                label_num=label_num) # TODO

            # Query the critic for an estimate of the reward
            critic_input = torch.zeros_like(decoder_input)
            critic_label_code = torch.zeros_like(decoder_label_code)
            critic_est = critic(encoder_input, encoder_label_code, critic_input, critic_label_code ).view(-1)

            advantage = (reward - critic_est)
            actor_loss = torch.mean(advantage.detach() * dim_logp.sum(dim=1))
            critic_loss = torch.mean(advantage ** 2)

            actor_optim.zero_grad()
            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            critic_optim.step()

            critic_rewards.append(torch.mean(critic_est.detach()).item())
            rewards.append(torch.mean(reward.detach()).item())
            # rewards.append(np.mean(reward))
            losses.append(torch.mean(actor_loss.detach()).item())
            my_critic_losses.append(torch.mean(critic_loss.detach()).item())

            if (batch_idx + 1) % log_step == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-log_step:])
                mean_reward = np.mean(rewards[-log_step:])
                my_rewards.append( abs(mean_reward) )
                my_losses.append(mean_loss)

                print('    Epoch %d  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs  | shape: %s, Label num %d' %
                      (epoch, batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1], encoder_input.shape, label_num ))

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        save_path = os.path.join(epoch_dir, 'actor.pt')
        torch.save(actor.state_dict(), save_path)

        save_path = os.path.join(epoch_dir, 'critic.pt')
        torch.save(critic.state_dict(), save_path)

        # Save rendering of validation set dims
        valid_dir = os.path.join(save_dir, 'render', '%s' % epoch)

        mean_valid = validate(
            valid_loader, actor,
            reward_type=kwargs['reward_type'],
            vis_type=kwargs['vis_type'],
            use_cuda=kwargs['use_cuda'],
            with_label=kwargs['with_label'],
            # label_num=kwargs['label_num_max'], # TODO
            reward_fn=reward_fn,
            render_fn=render_fn,
            save_dir=valid_dir
        )

        # Save best model parameters
        if mean_valid < best_reward:

            best_reward = mean_valid

            save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), save_path)

            save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic.state_dict(), save_path)

        print('Epoch %d,  mean epoch reward: %2.4f    valid: %2.4f, loss: %2.4f, took: %2.4fs '\
              '(%2.4fs / %d batches)' % \
              (epoch, mean_reward, mean_valid, mean_loss, time.time() - epoch_start,
              np.mean(times), log_step  ))

        import matplotlib.pyplot as plt
        plt.close('all')
        plt.title('Reward')
        plt.plot(range(len(my_rewards)), my_rewards, '-')
        plt.savefig(save_dir + '/img/reward.png' , bbox_inches='tight', dpi=400)

    np.savetxt(save_dir + '/reawrds.txt', my_rewards)
    np.savetxt(save_dir + '/actor_losses.txt', my_losses)
    np.savetxt(save_dir + '/critic_losses.txt', my_critic_losses)
    
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.title('Actor loss')
    plt.plot(range(len(my_losses)), my_losses, '-')
    plt.savefig(save_dir + '/img/actor.png' , bbox_inches='tight', dpi=400)
    plt.close('all')
    plt.title('Critic loss')
    plt.plot(range(len(my_critic_losses)), my_critic_losses, '-')
    plt.savefig(save_dir + '/img/critic.png' , bbox_inches='tight', dpi=400)
    
def load_and_train(args):

    # from vis import TrainDataset

    print('Loading data...')
    use_cuda = args.use_cuda
    
    def num_range_list( num_min, num_max, num_step ):
        if num_max == num_min:
            return [num_max]
        return [
            i for i in range(
                num_min, num_max+1, num_step
            )
        ]

    dim_num_range = num_range_list( args.dim_num_min, args.dim_num_max, args.dim_num_step )
    data_num_range = num_range_list( args.data_num_min, args.data_num_max, args.data_num_step )
    label_num_range = num_range_list( args.label_num_min, args.label_num_max, args.label_num_step )

    total_type_num = len(dim_num_range) * len(data_num_range) * len(label_num_range)

    each_train_size = int( args.train_size / total_type_num )
    each_valid_size = int( args.valid_size / total_type_num )

    train_files, valid_files, dim_nums, data_nums, label_nums = vis.create_mix_dataset(
        each_train_size,
        each_valid_size,
        args.data_type,
        args.vis_type,
        dim_num_range,
        data_num_range,
        label_num_range,
        args.standard,
        args.seed
    )

    
    if args.just_generate == True:
        return

    # train_data = VISDataset( train_file, args.data_num, args.dim_num, args.train_size,  args.vis_type, args.with_label, args.seed )
    # valid_data = VISDataset( valid_file, args.dim_num, args.valid_size, args.vis_type, args.with_label, args.seed + 1 )

    train_data = VISMDataset( train_files, data_nums, dim_nums, label_nums, args.train_size, args.vis_type, args.batch_size, args.label_type, args.seed )
    valid_data = VISMDataset( valid_files, data_nums, dim_nums, label_nums, args.valid_size, args.vis_type, args.valid_size, args.label_type, args.seed + 1 )

    actor = DRL(args.encoder_hidden_size,
                args.decoder_hidden_size,
                vis.update_mask,
                args.vis_type,
                args.reward_type,
                args.encoder_type,
                args.with_label,
                args.share_RNN,
                args.use_cuda,
                args.num_layers,
                args.dropout,
                )

    critic = Critic(args.encoder_hidden_size, args.with_label, args.share_RNN,
        args.encoder_type, args.num_layers,
        args.n_process_blocks, args.dropout)

    actor = nn.DataParallel(actor) 
    critic = nn.DataParallel(critic)

    if use_cuda:
        actor = actor.cuda()
        critic = critic.cuda()

    kwargs = vars(args)
    kwargs['dim_num'] = args.dim_num_max
    kwargs['train_data'] = train_data
    kwargs['valid_data'] = valid_data
    kwargs['reward_fn'] = vis.reward
    kwargs['render_fn'] = vis.render

    # multi_str: MMM 曾经对应的是 data dim label, M 为multi, S 为single
    # multi_str: MMM 现在对应的是 dim data label, M 为multi, S 为single

    multi_str = ''
    if len(dim_num_range) > 1:
        multi_str += 'M'
    else:
        multi_str += 'S'
    if len(data_num_range) > 1:
        multi_str += 'M'
    else:
        multi_str += 'S'
    if len(label_num_range) > 1:
        multi_str += 'M'
    else:
        multi_str += 'S'

    kwargs['multi_data'] = multi_str
    print("DIM DATA LABEL")
    print(multi_str)

    if args.checkpoint:
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path))

        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path))

        print('Loading pre-train model', path)

    train(actor, critic, **kwargs)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=12345, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vis')
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--n_process_blocks', default=3, type=int)
    parser.add_argument('--optimizer', default='Adam', type=str)

    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch_num', default=2, type=int)

    parser.add_argument('--use_cuda', default=True, type=str2bool)
    parser.add_argument('--cuda', default='0', type=str)

    parser.add_argument('--train_size',default=20, type=int)
    parser.add_argument('--valid_size', default=10, type=int)
    
    parser.add_argument('--dim_num_min', default=16, type=int)
    parser.add_argument('--dim_num_max', default=16, type=int)
    parser.add_argument('--dim_num_step', default=2, type=int)

    parser.add_argument('--data_num_min', default=8, type=int)
    parser.add_argument('--data_num_max', default=8, type=int)
    parser.add_argument('--data_num_step', default=4, type=int)

    parser.add_argument('--label_num_min', default=2, type=int)
    parser.add_argument('--label_num_max', default=2, type=int)
    parser.add_argument('--label_num_step', default=1, type=int)
    
    parser.add_argument('--standard', default=0.1, type=float)

    parser.add_argument('--encoder_hidden', dest='encoder_hidden_size', default=256, type=int)
    parser.add_argument('--decoder_hidden', dest='decoder_hidden_size', default=256, type=int)

    parser.add_argument('--vis_type', default='star', type=str)
    parser.add_argument('--reward_type', default='sc_sil', type=str)
    parser.add_argument('--encoder_type', default='rn2', type=str)
    parser.add_argument('--data_type', default='dis', type=str)

    parser.add_argument('--label_type', default='center', type=str)

    parser.add_argument('--with_label', default=True, type=str2bool)
    parser.add_argument('--share_RNN', default=True, type=str2bool)

    parser.add_argument('--just_test', default=False, type=str2bool)
    parser.add_argument('--just_generate', default=False, type=str2bool)

    parser.add_argument('--note', default='debug', type=str)

    args = parser.parse_args()
    if args.task == 'vis':
        print('Visual type:      %s' % args.vis_type)  
        print('Reward type:      %s' % args.reward_type)
        print('Encoder type:     %s' % args.encoder_type)
        print('Label type:       %s' % args.label_type)
        print('Data type:        %s' % args.data_type)
        print('Label num:        %s' % args.label_num_max)

        print('Standard:         %s' % args.standard)
        print('Data num:         %s' % args.data_num_max)
        print('Dim num:          %s' % args.dim_num_max)

        print('With label:       %s' % args.with_label)
        print('Share RNN :       %s' % args.share_RNN)
        print('Note:             %s' % args.note)
        if args.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        # torch.set_num_threads(1)
        load_and_train(args)
    else:
        raise ValueError('Task <%s> not understood'%args.task)
