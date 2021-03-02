
"""Defines the main task for the PACK
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm
import itertools
import shape_context
import time
import sys
sys.path.append('../')
import tools

class ShapeDataset(Dataset):
    def __init__(self, data_file, num_samples, seed):
        '''
        Data initialization
        '''
        super(ShapeDataset, self).__init__()
        if seed is None:
            seed = np.random.randint(123456)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # data = np.loadtxt(data_file + 'data.txt').astype('float32')
        # data = torch.from_numpy(data)
        # data = data.view(num_samples, 2, -1)
        # data = data.transpose(2, 1)

        # dim_num = data.shape[1]
        # data_num = data.shape[2]

        gt = np.loadtxt(data_file + 'gt.txt').astype('float32')
        gt = torch.from_numpy(gt)
        gt = gt.view(num_samples, 1)

        
        points = np.loadtxt(data_file + 'points.txt').astype('float32')
        points = torch.from_numpy(points)
        # batch_size x sample_num x 2
        points = points.view(num_samples, -1, 2)
        # batch_size x 2 x sample_num
        # points = points.transpose(2, 1)

        # self.data = data
        self.points = points
        self.gt = gt
        
        # print('    Input shape:  ', self.data.shape)
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (input, points, gt)
        # return (self.data[idx], self.points[idx], self.gt[idx])
        return (self.points[idx], self.gt[idx])


class MixShapeDataset(Dataset):
    def __init__(self, data_files, batch_size, num_samples, seed):
        '''
        Data initialization
        '''
        super(MixShapeDataset, self).__init__()
        if seed is None:
            seed = np.random.randint(123456)
        np.random.seed(seed)
        torch.manual_seed(seed)

        def read_data(data_files, num_samples):

            data = np.loadtxt(data_file + 'data.txt').astype('float32')
            data = torch.from_numpy(data)
            data = data.view(num_samples, 2, -1)
            data = data.transpose(2, 1)

            gt = np.loadtxt(data_file + 'gt.txt').astype('float32')
            gt = torch.from_numpy(gt)
            gt = gt.view(num_samples, 1)

            
            points = np.loadtxt(data_file + 'points.txt').astype('float32')
            points = torch.from_numpy(points)
            # batch_size x sample_num x 2
            points = points.view(num_samples, -1, 2)
            # batch_size x 2 x sample_num

            return data, points, gt

        total_data = []
        total_points = []
        total_gt = []

        each_data_num = int(num_samples / len(data_files))
        self.each_data_num = each_data_num

        for i in range(len(data_files)):
            data_file = data_files[i]
            data, points, gt = read_data(data_file, each_data_num)

            total_data.append( data )
            total_points.append( points )
            total_gt.append( gt )

        self.data = total_data
        self.points = total_points
        self.gt = total_gt
        
        for index, data in enumerate(total_data):
            print('    Input shape:  ', data.shape)

        self.num_samples = num_samples

        self.batch_size = batch_size
        self.sample_num = 0
        self.data_types = len(data_files)
        self.data_id = np.random.randint(0, self.data_types)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = int( idx % self.each_data_num )

        if self.sample_num == self.batch_size:
            self.data_id = np.random.randint(0, self.data_types)
            self.sample_num = 0

        self.sample_num += 1

        return (self.data[self.data_id][idx], self.points[self.data_id][idx], self.gt[self.data_id][idx])


def create_dataset( dim_num, train_size, valid_size, sample_num, standard=0.1, seed=None ):
    dim_num = int(dim_num)
    if seed is None:
        seed = np.random.randint(123456789)
    np.random.seed(seed)

    train_dir = './data/rand/train-' + str(dim_num) + 'd-' + str(train_size) + '/'
    valid_dir = './data/rand/valid-' + str(dim_num) + 'd-' + str(valid_size) + '/'

    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(train_dir + 'gt.txt') and os.path.exists(valid_dir  + 'gt.txt'):
        return train_dir, valid_dir
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    
    sc = shape_context.ShapeContext()

    # if dim_num == 0:
    #     max_dim = 16
    #     min_dim = 24
    # else:
    if True:
        max_dim = dim_num+1
        min_dim = dim_num


    def generate_data(data_dir, data_size):

        data_f = open(data_dir + 'data.txt', 'w')
        points_f = open(data_dir + 'points.txt', 'w')
        gt_f = open(data_dir + 'gt.txt', 'w')

        # standard = 0.1

        for _ in tqdm(range(data_size)):

            dim_num = np.random.randint(min_dim, max_dim) 
            
            data_num = 2
            # sample_num = 80
            angles = np.linspace(0, 2*np.pi, dim_num+1)[:-1]
            cos_sin = np.array( [ (np.cos(ang), (np.sin(ang))) for ang in angles ])

            # data = np.zeros((data_num, dim_num))
            # base_mean = np.random.uniform( 10, 100, size=(dim_num))
            # if np.random.rand() >= 0.5:
            #     # data = np.random.rand(data_num, dim_num)
            #     for data_index in range(data_num):
            #         for dim_index in range(dim_num):
            #             mean = base_mean[dim_index]
            #             while True:
            #                 rand_data = np.random.normal( loc=mean, scale= standard * mean, size=(1) )
            #                 if (rand_data > 0).all():
            #                     break
            #             data[ data_index , dim_index] = rand_data
            # else:
            # # if same class
            #     for dim_index in range(dim_num):
            #         mean = base_mean[dim_index]
            #         while True:
            #             rand_data = np.random.normal( loc=mean, scale= standard * mean, size=(data_num) )
            #             if (rand_data > 0).all():
            #                 break
            #         data[:, dim_index] = rand_data
            
            large_data_num = 16
            label_num = 2
            if True:
                sub_labels = np.zeros(large_data_num)
                label_index = 0
                label_step = int( np.floor(large_data_num / label_num) )
                label_id = 0

                while label_id < label_num:
                    sub_labels[ label_index:label_index+label_step ] = label_id
                    label_index += label_step
                    label_id += 1
                np.random.shuffle(sub_labels)
                
            sub_data = np.zeros((large_data_num, dim_num))
            for i in range(label_num):

                for label_index in range(label_num):
                    # rand
                    label_mask = (sub_labels == label_index)
                    base_mean = np.random.uniform( 10, 100, size=(dim_num))

                    for dim_index in range(dim_num):
                        mean = base_mean[dim_index]
                        while True:
                            rand_data = np.random.normal( loc=mean, scale= standard * mean, size=(label_mask.sum()) )
                            if (rand_data > 0).all():
                                break
                        sub_data[label_mask, dim_index] = rand_data

            data_norm = np.linalg.norm( sub_data, axis=1 ).reshape(-1, 1).repeat(dim_num, 1)
            data = sub_data / data_norm
            data = data[:2]
            
            shape1 = shape_context.get_shape([data[0,:]], sample_num=sample_num)
            shape2 = shape_context.get_shape([data[1,:]], sample_num=sample_num)

            h1 = sc.compute(shape1)
            h2 = sc.compute(shape2)

            gt = sc.cost(h1, h2)

            shapes = np.row_stack( (shape1, shape2) )
            
            data_f.writelines( arr2str( data.flatten() ))
            points_f.writelines( arr2str(shapes.flatten()) )
            gt_f.writelines( str(gt) + '\n' )

        data_f.close()
        points_f.close()
        gt_f.close()


    if not os.path.exists(train_dir + 'gt.txt'):
        generate_data(train_dir, train_size)

    if not os.path.exists(valid_dir + 'gt.txt'):
        generate_data(valid_dir, valid_size)

    return train_dir, valid_dir


def create_mix_dataset( dim_num_ranges, train_size, valid_size, sample_num, standard, seed=None ):

    train_files = []
    valid_files = []

    for dim_num in dim_num_ranges:
        train_file, valid_file = create_dataset( dim_num, train_size, valid_size, sample_num, standard, seed )

        train_files.append(train_file)
        valid_files.append(valid_file)
    
    return train_files, valid_files


def render(data, score, gt, save_path):
    """Plots"""
    # batch x dim_num x data_num
    batch_size = data.shape[0]
    dim_num = data.shape[1]
    data_num = data.shape[2]

    # data = data.transpose(2, 1)

    # sc = shape_context.ShapeContext()
    data = data.cpu().numpy()
    score = score.cpu().numpy()
    gt = gt.cpu().numpy()
    

    for batch_index in range(batch_size):
        if batch_index >= 1:
            continue
        d = data[batch_index].transpose()
        # shape1 = shape_context.get_shape([d[0,:]], sample_num=5)
        # shape2 = shape_context.get_shape([d[1,:]], sample_num=5)

        # h1 = sc.compute(shape1)
        # h2 = sc.compute(shape2)

        # tmp = sc.cost(h1, h2)


        # import IPython
        # IPython.embed()

        # TODO draw image
        tools.draw( 
            d, None,
            vis_type='star',
            reward_type='sc', 
            save_title="Score: %.3f\nGt: %.3f" % (score[batch_index], gt[batch_index]),
            save_name=save_path[:-8] + '-%d' % (batch_index) )

    np.savetxt(save_path[:-8] + '-score.txt', score)
    np.savetxt(save_path[:-8] + '-gt.txt', gt)
    np.savetxt(save_path[:-8] + '-loss.txt', (score-gt)*(score-gt))

