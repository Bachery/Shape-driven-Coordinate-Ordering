import os
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm
import multiprocessing
import itertools
from scipy import stats
from scipy.integrate import trapz

import tools


def encode(data, labels, label_num, label_type):
    # data:  float array ( data_num, dim_num )
    # labels:  int array ( data_num )
    data_num = data.shape[0]
    dim_num = data.shape[1]
    
    if label_type == 'label':
        return labels
    labels = labels.astype(int)
    if label_type == 'matrix':
        labels = labels.reshape((-1, 1))
        # build matrix
        label_matrix = np.zeros( (data_num, data_num) ).astype('byte')
        for label_id in range(label_num):
            label_id_mask = (labels == label_id)
            label_id_matrix = np.matmul( label_id_mask, label_id_mask.transpose() )
            label_matrix = label_matrix | label_id_matrix
        label_matrix = label_matrix.astype('float32').flatten()
        
        label_matrix.reshape(( -1, 1 )).repeat( dim_num, -1 )
        
        return label_matrix

    elif label_type == 'triangle':
        data_num = data.shape[0]
        labels = labels.reshape((-1, 1))
        # build matrix
        label_matrix = np.zeros( (data_num, data_num) ).astype('byte')
        for label_id in range(label_num):
            label_id_mask = (labels == label_id)
            label_id_matrix = np.matmul( label_id_mask, label_id_mask.transpose() )
            label_matrix = label_matrix | label_id_matrix

        half = np.zeros( int(data_num * (data_num-1) / 2) )
        index = 0
        for i in range(data_num):
            for j in range(i+1, data_num):
                half[index] = label_matrix[i][j]
                index += 1
        half = half.astype('float32')
        half.reshape(( -1, 1 )).repeat( dim_num, -1 )
        return half
    elif label_type == 'center':
        # calc center
        label_centers = []
        for label_id in range(label_num):
            label_centers.append(data[ labels == label_id ].mean(axis=0))
        label_centers = np.array(label_centers)
        # data_num * dim_num
        return label_centers[labels]
    else:
        print('====> Error in vis encode')



class VISDataset(Dataset):
    def __init__(self, data_file, dim_num, num_samples, label_num, vis_type, with_label, label_type, seed):
        '''
        Data initialization
        ----
        params
        ----
            vis_type: str, the type of input data
                'parallel'
                'star'
                'radviz'
        '''
        super(VISDataset, self).__init__()
        # if seed is None:
        #     seed = np.random.randint(123456)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        data = np.loadtxt(data_file + 'data.txt').astype('float32')
        
        if vis_type == 'star':
            data = torch.from_numpy(data)
            data = data.view(num_samples, -1, dim_num)
            # data_norm = data.norm(dim=1).unsqueeze(1).expand_as(data)
            # data = data / data_norm

        else:
            data = data.reshape((num_samples, -1, dim_num))
            data_num = data.shape[1]
            # normalize
            max_data = np.max(data, axis=1).reshape(num_samples, 1, -1).repeat(data_num, 1)
            min_data = np.min(data, axis=1).reshape(num_samples, 1, -1).repeat(data_num, 1)
            data = (data - min_data) / (max_data - min_data)            
            
            # random_order = [ o for o in range(dim_num) ]
            # random_order = np.array(random_order)
            # np.random.shuffle(random_order)
            # data = data[:, :, random_order]
            # print(random_order)

            data = torch.from_numpy(data)


        data_num = data.shape[1]

        labels = np.loadtxt(data_file + 'label.txt').astype('float32')

        labels_code = []
        
        # TODO
        for i in tqdm(range(num_samples)):
            encode_l = encode(data[i], labels[i], label_num, label_type)
            labels_code.append(encode_l)
        
        labels_code = np.array(labels_code)
        labels_code = torch.from_numpy(labels_code)
        if label_type != 'center':
            labels_code = labels_code.view(num_samples, -1, 1).repeat(1, 1, dim_num)

        labels = torch.from_numpy(labels)
        labels = labels.view(num_samples, -1, 1).repeat(1, 1, dim_num)
        # labels = labels.view(num_samples, -1, 1).expand_as( data )

        self.encoder_input = data
        self.encoder_label = labels
        self.encoder_label_code = labels_code
        
        self.decoder_input = torch.zeros_like(self.encoder_input[0,:,0:1])
        self.decoder_label = torch.zeros_like(self.encoder_label[0,:,0:1])
        self.decoder_label_code = torch.zeros_like(self.encoder_label_code[0,:,0:1])

        
        print('    Input shape :  ', self.encoder_input.shape)
        print('    Input label :  ', self.encoder_label.shape)
        print('    Encode label:  ', self.encoder_label_code.shape)
        print('')
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (input, start_loc)
        return (self.encoder_input[idx], self.encoder_label[idx], self.encoder_label_code[idx],
            self.decoder_input, self.decoder_label, self.decoder_label_code )


class VISMDataset(Dataset):
    def __init__(self, data_files, data_nums, dim_nums, label_nums, num_samples, vis_type, batch_size, label_type, seed):
        '''
        Data initialization, load multi data to train
        ----
        params
        ----
            vis_type: str, the type of input data
                'parallel'
                'star'
                'radviz'
        '''
        super(VISMDataset, self).__init__()

        def read_data(data_file, data_num, dim_num, label_type):
            print(data_file)
            
            data = np.loadtxt(data_file + 'data.txt').astype('float32')
            if vis_type == 'star':
                data = data.reshape((-1, data_num, dim_num))
                # data = torch.from_numpy(data)
            else:
                data = data.reshape((-1, data_num, dim_num))
                # data_num = data.shape[1]
                # normalize
                max_data = np.max(data, axis=1).reshape(-1, 1, dim_num).repeat(data_num, 1)
                min_data = np.min(data, axis=1).reshape(-1, 1, dim_num).repeat(data_num, 1)
                data = (data - min_data) / (max_data - min_data)            
                # data = torch.from_numpy(data)
                
            # data_num = data.shape[1]
            labels = np.loadtxt(data_file + 'label.txt').astype('float32')

            num_samples = data.shape[0]
            
            labels_code = []
            
            # TODO
            for i in tqdm(range(num_samples)):
                encode_l = encode(data[i], labels[i], label_num, label_type)
                labels_code.append(encode_l)
            
            data = torch.from_numpy(data)
                
            labels_code = np.array(labels_code)
            labels_code = torch.from_numpy(labels_code)
            if label_type != 'center':
                labels_code = labels_code.view(num_samples, -1, 1).repeat(1, 1, dim_num)
            
            labels = torch.from_numpy(labels)
            labels = labels.view(num_samples, data_num, 1).expand_as( data )

            data_single = torch.zeros_like(data[0,:,0:1])
            label_single = torch.zeros_like(labels[0,:,0:1])
            label_single_code = torch.zeros_like(labels_code[0,:,0:1])
            
            return data, labels, labels_code, data_single, label_single, label_single_code

        total_data = []
        total_label = []
        total_label_code = []
        total_data_single = []
        total_label_single = []
        total_label_single_code = []

        total_label_num = []

        for i in range( len(data_files) ):
            data_file = data_files[i]
            data_num = data_nums[i]
            dim_num = dim_nums[i]
            label_num = label_nums[i]
            
            d, l, lc, d_single, l_single, l_single_c = read_data(data_file, data_num, dim_num, label_type)
            
            total_data.append( d )
            total_label.append( l )
            total_label_code.append( lc )
            total_data_single.append( d_single )
            total_label_single.append( l_single )
            total_label_single_code.append( l_single_c )
            total_label_num.append( label_num )


        self.encoder_input = total_data
        self.encoder_label = total_label
        self.encoder_label_code = total_label_code
        
        self.decoder_input = total_data_single
        self.decoder_label = total_label_single
        self.decoder_label_code = total_label_single_code
        
        self.label_nums = total_label_num
        
        for index, data in enumerate(total_data):
            print('    Input shape:  ', data.shape)
            print('    Input label:  ', total_label[index].shape)
            print('    Code shape :  ', total_label_code[index].shape)
            
        self.num_samples = num_samples

        self.batch_size = batch_size
        self.sample_num = 0
        self.data_types = len(data_files)
        self.data_id = np.random.randint(0, self.data_types)
        self.each_data_num = int( num_samples / len(total_data) )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (encoder_input, encoder_label, encoder_label_code)
        # (decoder_input, decoder_label, decoder_label_code)
        # (label_num, )
        idx = int( idx % self.each_data_num )

        if self.sample_num == self.batch_size:
            self.data_id = np.random.randint(0, self.data_types)
            self.sample_num = 0

        self.sample_num += 1

        return (self.encoder_input[self.data_id][idx], self.encoder_label[self.data_id][idx], self.encoder_label_code[self.data_id][idx], 
            self.decoder_input[self.data_id], self.decoder_label[self.data_id], self.decoder_label_code[self.data_id], 
            self.label_nums[self.data_id] )


def count_index(index_list, min_num):
    """
    count the index_list compute from all the dim data, 
    only the index repeat dim_num times could stay
    """
    counter = Counter(index_list)
    keys = np.array(list(counter.keys())).astype('int')
    values = np.array(list(counter.values()))

    good_index = keys[ values >= min_num ]
    return good_index

def get_densest_in_one_dimension(x):
    # import seaborn as sns
    # sns.distplot(x)

    min_x = np.min(x)
    max_x = np.max(x)
    s = (max_x - min_x) / 2
    if s == 0:
        return x[0]

    bandwidth = 1.06 * x.std() * x.size ** (-1 / 5.)
    support = np.linspace(min_x-s, max_x+s, 200)

    kernels = []
    for x_i in x:
        kernel = stats.norm(x_i, bandwidth).pdf(support)
        kernels.append(kernel)
    density = np.sum(kernels, axis=0)
    density /= trapz(density, support)

    return support[ np.where(density == np.max(density)) ]

def find_good_index( data, labels, label_index, standard=0.1 ):
    data = data[ labels == label_index ]

    data_num = data.shape[0]
    dim_num = data.shape[1]

    # dim_mean = np.mean(data, axis=0)
    dim_mean = np.zeros(dim_num)
    for dim in range(dim_num):
        x = data[:, dim]
        dim_mean[dim] = get_densest_in_one_dimension(x)

    good_var_index_list = []

    for dim_index in range(dim_num):

        data_dim_i = data[:, dim_index ]
        # mean
        dim_i_mean = dim_mean[dim_index]

        
        
        # var
        good_var = standard * dim_i_mean
        
        mask_left = (data_dim_i > (dim_i_mean- 3*good_var) )
        mask_right = ( data_dim_i < (dim_i_mean + 3*good_var) )

        good_var_index = np.where( mask_left * mask_right )[0]
        good_var_index_list.extend( good_var_index )

    good_index = count_index(good_var_index_list, dim_num)

    good_index = np.where( labels == label_index )[0][good_index]
    
    return good_index

class TrainDataset(Dataset):
    def __init__(self, data_name='winequality', category_index=-1):
        super(TrainDataset, self).__init__()

        data, category, label_set = self.read_data('./datasets/%s/%s.data' % (data_name, data_name), category_index)

        data_num = len(data)
        dim_num = len(data[0,])

        # norm
        # data_norm = np.mean(wines, axis=0).reshape(1, -1).repeat(wines_num, 0)
        data_max = np.max(data, axis=0).reshape(1, -1).repeat(data_num, 0)
        data_min = np.min(data, axis=0).reshape(1, -1).repeat(data_num, 0)
        data = (data - data_min) / (data_max - data_min)

        data = data[:4096]
        category = category[:4096]

        # convert to tensor
        data = torch.from_numpy(data).view( -1, 128, dim_num )
        category = torch.from_numpy(category).view( -1, 128, 1 ).expand_as( data )

        self.encoder_input = data
        self.encoder_label = category
        
        self.decoder_input = torch.zeros_like(self.encoder_input[:,:,0:1])
        self.decoder_label = torch.zeros_like(self.encoder_input[:,:,0:1])
        

        self.num_samples = 32

        print('Train dataset!!!')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (input, start_loc)
        return (self.encoder_input[idx], self.encoder_label[idx], self.decoder_input[idx], self.decoder_label)

    def read_data(self, filename, category_dim=0):
        """
        Parameters
        ----
            filename: string
                path of dataset
            category_dim: int
                the dimensino index of category / class
        Returns
        ----
            data: array of data (data_num, dim_num)
                the first dim is category
        """
        data = []
        category = []
        category_set = []
        # read data
        with open(filename, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    continue
                line = line.strip('\n').split(';')
                if line[category_dim] not in category_set:
                    category_set.append(line[category_dim])
                
                category.append( category_set.index(line[category_dim]) )

                data_point = []
                for i in range(len(line)):
                    if i != category_dim:
                        data_point.append( float(line[i]) )
                data.append(data_point)
                    
        data = np.array(data).astype('float32')
        category = np.array(category).astype('float32')
        
        return data, category, category_set



def create_dataset( dim_num, data_num, train_size, valid_size, data_type, label_num, vis_type, standard, seed=None):
    dim_num = int(dim_num)
    # if seed is None:
    #     seed = np.random.randint(123456789)
    # np.random.seed(seed)

    # if label_num == -1:
    #     if rate == 0:
    #         train_dir = './data/rand/train-' + str(dim_num) + 'd-' + str(data_num) + '-' + str(train_size) + '-' + data_type + '-' + str(data_range) + '/'
    #         valid_dir = './data/rand/valid-' + str(dim_num) + 'd-' + str(data_num) + '-' + str(valid_size) + '-' + data_type + '-' + str(data_range) + '/'
    #     else:
    #         train_dir = './data/rand/train-' + str(dim_num) + 'd-' + str(data_num) + '-' + str(train_size) + '-' + data_type + '-' + str(rate) + '-' + str(data_range) + '/'
    #         valid_dir = './data/rand/valid-' + str(dim_num) + 'd-' + str(data_num) + '-' + str(valid_size) + '-' + data_type + '-' + str(rate) + '-' + str(data_range) + '/'
    # else:
    train_dir = './data/'+ vis_type + '/train-' + str(dim_num) + 'd-' + str(label_num) + 'c-' + str(data_num) + 'n-' + str(train_size) + '-' + data_type + '-' + str(standard) + '/'
    valid_dir = './data/'+ vis_type + '/valid-' + str(dim_num) + 'd-' + str(label_num) + 'c-' + str(data_num) + 'n-' + str(valid_size) + '-' + data_type + '-' + str(standard) + '/'


    def arr2str(arr):
        ret = ''
        for i in range(len(arr)-1):
            ret += str(arr[i]) + ' '
        ret += str(arr[-1]) + '\n'
        return ret

    if os.path.exists(train_dir + 'data.txt') and os.path.exists(valid_dir  + 'data.txt'):
        return train_dir, valid_dir
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)

    if label_num == -1:
        label_num = 2 # TODO
    

    def gauss_data(data_num, x=None):
        if x is None:
            x = np.random.normal(size=(data_num))
        y = np.random.normal(size=(data_num))
        x = (x- x.min() ) / (x.max() - x.min())
        y = (y- y.min() ) / (y.max() - y.min())

        return x, y

    def relation_data(data_num, x=None, only_neg=False):
        max_range = 1
        min_range = -1

        if only_neg:
            a = np.random.uniform(min_range, 0)
        else:
            a = np.random.uniform(min_range, max_range)
        b_mean = np.random.uniform(-1, 1)
        b_scale = standard

        b = np.random.normal(loc=b_mean, scale=b_scale, size=data_num)
        if x is None:
            x = np.random.uniform(0, max_range, size=(data_num))
        y = a * x + b

        x = (x- x.min() ) / (x.max() - x.min())
        y = (y- y.min() ) / (y.max() - y.min())
        return x, y, a

    def generate_data_label(data_dir, data_size):

        data_f = open(data_dir + 'data.txt', 'w')
        label_f = open(data_dir + 'label.txt', 'w')
        for _ in tqdm(range(data_size)):
            if data_type == 'uniform':
                sub_data = np.random.rand(data_num, dim_num).flatten()
                sub_labels = np.random.random_integers( 0, label_num-1, (data_num))
                
                data_f.writelines( arr2str( sub_data ))
                label_f.writelines( arr2str( sub_labels ))
                continue
                
            if data_type == 'ran':
                while True:
                    sub_labels = np.random.random_integers( 0, label_num-1, (data_num))
                    # don't generate single data points
                    count = 0
                    for label_id in range(label_num):
                        if (sub_labels == label_id).sum() > 1:
                            count += 1
                    if count == label_num:
                        break
                
            elif data_type == 'sim':
                label_id = np.random.randint( 0, label_num )
                sub_labels = np.ones(data_num) * label_id

            elif data_type == 'dis':
                sub_labels = np.zeros(data_num)
                label_index = 0
                # label_step = int(data_num / label_num)
                label_step = int( np.floor(data_num / label_num) )
                label_id = 0

                # while label_index < data_num:
                while label_id < label_num:
                    sub_labels[ label_index:label_index+label_step ] = label_id
                    label_index += label_step
                    label_id += 1
                np.random.shuffle(sub_labels)

            sub_data = np.zeros((data_num, dim_num))
            for i in range(label_num):

                for label_index in range(label_num):
                    # rand
                    label_mask = (sub_labels == label_index)
                    base_mean = np.random.uniform( 10, 100, size=(dim_num))

                    for dim_index in range(dim_num):
                        mean = base_mean[dim_index]
                        while True:
                            rand_data = np.random.normal( loc=mean, scale= standard * mean, size=(label_mask.sum()) )
                            if vis_type == 'star':
                                if (rand_data > 0).all():
                                    break
                            else:
                                break

                        sub_data[label_mask, dim_index] = rand_data

            sub_data = sub_data.flatten()

            data_f.writelines( arr2str( sub_data ))
            label_f.writelines( arr2str( sub_labels ))

        data_f.close()
        label_f.close()

    def generate_data_relation(data_dir, data_size):

        data_f = open(data_dir + 'data.txt', 'w')
        label_f = open(data_dir + 'label.txt', 'w')
        pair_f = open(data_dir + 'pair.txt', 'w')
        relation_f = open(data_dir + 'rel.txt', 'w')

        for _ in tqdm(range(data_size)):
            labels = np.random.random_integers( 0, label_num-1, (data_num))

            data = np.zeros((data_num, dim_num))

            choice_list = []
            rel_list = []
            
            x_dim = 0
            y_dim = 1
            while y_dim < dim_num:
                
                if x_dim == 0:
                    x, y, a = relation_data(data_num)
                    choice_list.append( x_dim )
                    choice_list.append( y_dim )
                    rel_list.append( a )
                else:
                    # half gauss, half relation
                    if np.random.rand() >= 0.5:
                        x, y, a = relation_data(data_num, data[:, x_dim])
                        choice_list.append( x_dim )
                        choice_list.append( y_dim )
                        rel_list.append( a )
                    else:
                        x, y = gauss_data(data_num, data[:, x_dim])
                data[:, x_dim] = x
                data[:, y_dim] = y
                x_dim = y_dim
                y_dim = x_dim+1
            
            # random_order = [ o for o in range(dim_num) ]
            # random_order = np.array(random_order)
            # np.random.shuffle(random_order)
            # data = data[:, random_order]

            data = data.flatten()
            
            choice_pair = np.zeros(dim_num*2)
            choice_pair[:len(choice_list)] = choice_list
            
            rel_data = np.zeros(dim_num)
            rel_data[:len(rel_list)] = rel_list

            data_f.writelines( arr2str( data ))
            pair_f.writelines( arr2str( choice_pair ))
            relation_f.writelines( arr2str( rel_data ))
            label_f.writelines( arr2str( labels ))

        data_f.close()
        pair_f.close()
        relation_f.close()
        label_f.close()



    if not os.path.exists(train_dir + 'data.txt'):
        if data_type == 'rel':
            generate_data_relation(train_dir, train_size)
        else:
            generate_data_label(train_dir, train_size)
            
    if not os.path.exists(valid_dir + 'data.txt'):
        if data_type == 'rel':
            generate_data_relation(valid_dir, valid_size)
        else:
            generate_data_label(valid_dir, valid_size)

    return train_dir, valid_dir

def create_mix_dataset( train_size, valid_size, data_type, vis_type, dim_num_range, data_num_range, label_num_range, standard, seed=None):

    train_files = []
    valid_files = []

    dim_nums = []
    data_nums = []
    label_nums = []

    for dim_num in dim_num_range:
        for data_num in data_num_range:
            for label_num in label_num_range:
                train, valid = create_dataset(dim_num, data_num, train_size, valid_size, data_type, label_num, vis_type, standard, seed)
                train_files.append(train)
                valid_files.append(valid)

                dim_nums.append(dim_num)
                data_nums.append(data_num)
                label_nums.append(label_num)
        
    return train_files, valid_files, dim_nums, data_nums, label_nums

def update_mask(mask, chosen_idx):
    """Marks the visited city, so it can't be selected a second time."""
    mask.scatter_(1, chosen_idx.unsqueeze(1), 0)
    return mask

def reward(encoder_input, encoder_label, dim_indices, vis_type, reward_type, with_label, evaluate_tool=None, label_num=0):
    """
    Parameters
    ----------
    encoder_input: torch.FloatTensor containing static (e.g. w, h) data (batch_size, sourceL, num_cities)
    dim_indices: torch.IntTensor of size (batch_size, num_cities)

    Returns
    -------
    Total Euclidean distance between neighbor dimensions. of size (batch_size, num_cities)
    """
    
    use_cuda = encoder_input.is_cuda
    
    data_num = encoder_input.shape[1]
    dim_num = encoder_input.shape[2]
    
    # batch_size x dim_num  // dim_indices and expand as(static)
    idx = dim_indices.unsqueeze(1).repeat(1, data_num, 1)

    sample_input = torch.gather(encoder_input.data, 2, idx)
    sample_label = torch.gather(encoder_label.data, 2, idx)
    
    batch_size = sample_input.shape[0]

    scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()
    
    order = None
    if evaluate_tool is None:
        scores = tools.evaluate( sample_input, sample_label, order, vis_type, reward_type, with_label, label_num=label_num )
    else:
        scores = evaluate_tool.evaluate( sample_input, sample_label, order, vis_type, reward_type, with_label, label_num=label_num )
    
    # if reward_type == 'parg_pcc':
    #     scores = abs(scores)
    #     scores = -scores

    return scores

def render(encoder_input, encoder_label, dim_indices, save_path, vis_type, reward_type, with_label, label_num):
    """Plots"""
    batch_size = encoder_input.shape[0]
    data_num = encoder_input.shape[1]
    dim_num = encoder_input.shape[2]
    
    # batch_size x dim_num
    idx = dim_indices.unsqueeze(1).repeat(1, data_num, 1)

    # batch_size x data_num x dim_num
    sample_input = torch.gather(encoder_input.data, 2, idx)
    sample_label = torch.gather(encoder_label.data, 2, idx)

    for batch_index in range(batch_size):
        
        dpi = 400
        # if batch_index >= 10:
        # if batch_index not in [6, 56, 62, 77] or batch_index > 10:
        if batch_index not in [6]:
            continue

        
        data = sample_input[batch_index:batch_index+1,:,:]
        label = sample_label[batch_index:batch_index+1,:,:]
        order = dim_indices[batch_index]

        origin_data = encoder_input.data[ batch_index:batch_index+1, :, : ]
        origin_label = encoder_label.data[ batch_index:batch_index+1, :, : ]
        origin_order = [ i for i in range(dim_num) ]

        if True:
            scores = tools.evaluate(data, label, order, vis_type, reward_type, with_label, label_num=label_num)
            origin_scores = tools.evaluate(origin_data, origin_label, origin_order, vis_type, reward_type, with_label, label_num=label_num)
        
        data = data.cpu().numpy()[0]
        label = label.cpu().numpy()[0,:,0]
        order = order.cpu().numpy()

        origin_data = origin_data.cpu().numpy()[0]
        origin_label = origin_label.cpu().numpy()[0,:,0]
        
        with_category = False
        if vis_type == 'star' or vis_type == 'radviz':
            data = np.column_stack((label, data))
            origin_data = np.column_stack((origin_label, origin_data))
            with_category = True

        scores = abs(scores)
        origin_scores = abs(origin_scores)
        

        if vis_type == 'star':

            # TODO draw image
            tools.draw( 
                data, order,
                vis_type,
                reward_type, 
                save_title="%.3f" % (scores),
                # save_title="CLASS DIST: %.3f\nSIL: %.3f" % (mer, sil),
                # save_name=save_path[:-8] + '-%d' % (batch_index), with_category=with_category, 
                save_name=save_path[:-8] + '-%d_%.3f' % (batch_index, scores), with_category=with_category, 

                dpi=dpi, label_num=label_num )

            tools.draw( 
                origin_data, origin_order,
                vis_type,
                reward_type, 
                save_title="%.3f" % (origin_scores),
                # save_title="CLASS DIST: %.3f\nSIL: %.3f" % (mer_o, sil_o),
                # save_name=save_path[:-8] + '-%d-o' % (batch_index), with_category=with_category, 
                save_name=save_path[:-8] + '-%d-%.3f-o' % (batch_index, origin_scores), with_category=with_category, 

                dpi=dpi, label_num=label_num )
            # print(origin_scores)
            # print(scores)
            # np.savetxt(save_path[:-8] + '-%.3f_%.3f.txt' % (origin_scores, scores), data[0] )
            
        else:
            # TODO draw image
            tools.draw( 
                data, order,
                vis_type,
                reward_type, 
                # save_title="Score: %.3f\n" % (scores),
                save_title='',
                save_name=save_path[:-8] + '-%d_%.3f' % (batch_index, scores), with_category=with_category, 
                dpi=dpi, label_num=label_num )

            tools.draw( 
                origin_data, origin_order,
                vis_type,
                reward_type, 
                # save_title="Score: %.3f\n" % (origin_scores),
                save_title='',
                save_name=save_path[:-8] + '-%d-o-_%.3f' % (batch_index, origin_scores), with_category=with_category, 
                dpi=dpi, label_num=label_num )
            # print(origin_scores)
            # print(scores)
            # np.savetxt(save_path[:-8] + '-%.3f_%.3f.txt' % (origin_scores, scores), data[0] )


    # if vis_type == 'star':
    #     sim, mer, sil = tools.evaluate(sample_input, sample_label, order, vis_type, 'sc_all', with_label, label_num=label_num)
    #     sim_o, mer_o, sil_o = tools.evaluate(encoder_input, encoder_label, origin_order, vis_type, 'sc_all', with_label, label_num=label_num)

    #     sim = abs(sim)
    #     mer = abs(mer)
    #     sil = abs(sil)
    #     sim_o = abs(sim_o)
    #     mer_o = abs(mer_o)
    #     sil_o = abs(sil_o)

    #     if reward_type == 'sc_sim':
    #         scores = sim
    #         origin_scores = sim_o
    #     elif reward_type == 'sc_mer':
    #         scores = mer
    #         origin_scores = mer_o
    #     elif reward_type == 'sc_sil':
    #         scores = sil
    #         origin_scores = sil_o
    #     else:
    #         print('=======> Error in render')
    # else:
    if True:
        order = None
        origin_order = None
        scores = tools.evaluate(sample_input, sample_label, order, vis_type, reward_type, with_label, label_num=label_num)
        origin_scores = tools.evaluate(encoder_input, encoder_label, origin_order, vis_type, reward_type, with_label, label_num=label_num)

    

    # if vis_type == 'star':
    #     np.savetxt(save_path[:-8] + '-sim.txt', sim.cpu().numpy())
    #     np.savetxt(save_path[:-8] + '-mer.txt', mer.cpu().numpy())
    #     np.savetxt(save_path[:-8] + '-sil.txt', sil.cpu().numpy())
    #     np.savetxt(save_path[:-8] + '-sim_o.txt', sim_o.cpu().numpy())
    #     np.savetxt(save_path[:-8] + '-mer_o.txt', mer_o.cpu().numpy())
    #     np.savetxt(save_path[:-8] + '-sil_o.txt', sil_o.cpu().numpy())

    
    ids = dim_indices.cpu().numpy().astype('int')
    np.savetxt(save_path[:-8] + '-score.txt', scores.cpu().numpy())
    np.savetxt(save_path[:-8] + '-score-o.txt', origin_scores.cpu().numpy())
    np.savetxt(save_path[:-8] + '-ids.txt', ids)
