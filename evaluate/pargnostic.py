import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from scipy.stats import pearsonr

def pargnostics_o(data, labels, with_label=True, label_num=2, reward_type='cross', NPG=None):
    """
    Parameters
    ----
        data: Tensor of data (batch_size, data_num, dim_num)
    """
    use_cuda = data.is_cuda

    # data_norm = data.norm(dim=2).unsqueeze(2).expand_as(data)
    # data = data / data_norm


    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]

    data = data.cpu().numpy()
    labels = labels.cpu().numpy()
    
    bin_num = 20
    angle_bin_num = 9

    scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()

    # normalize
    for batch_index in range(batch_size):
        bin_data = bin_coordinate(data[batch_index], bin_num)
        score = 0
        for i in range(dim_num-1):
            if reward_type == 'cross':
                cross, angle = line_cross(data[batch_index], bin_data, i, i+1, bin_num, angle_bin_num)
                score += cross
            elif reward_type == 'angle':
                cross, angle = line_cross(data[batch_index], bin_data, i, i+1, bin_num, angle_bin_num)
                score += angle
            elif reward_type == 'param':
                # cross, angle = line_cross(data[batch_index], bin_data, i, i+1, bin_num, angle_bin_num)
                param = parallelism(bin_data, i, i+1, bin_num)
                score += param
            elif reward_type == 'cap':
                cross, angle = line_cross(data[batch_index], bin_data, i, i+1, bin_num, angle_bin_num)
                param = parallelism(bin_data, i, i+1, bin_num)
                score += (cross + angle + param) / 3
            
            # score += angle
            # score += param
        scores[batch_index] += score / (dim_num-1)

    return scores


def pargnostics(data, labels, with_label=True, label_num=2, reward_type='cross', NPG=None):
    """
    Parameters
    ----
        data: Tensor of data (batch_size, data_num, dim_num)
    """
    use_cuda = data.is_cuda

    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]
    
    bin_num = 50
    angle_bin_num = 90

    # if reward_type == 'all':
    #     if NPG is None:
    #         NPG_c = NetPargnostics( 'cross', bin_num, 128, 1, 0.1, use_cuda)
    #         NPG_a = NetPargnostics( 'angle', bin_num, 128, 1, 0.1, use_cuda)
    #         NPG_p = NetPargnostics( 'parallelism', bin_num, 128, 1, 0.1, use_cuda)
    #     else:
    #         NPG_c, NPG_a, NPG_p = NPG
    #     scores_c = NPG_c.net_parg(data)
    #     scores_a = NPG_a.net_parg(data)
    #     scores_p = NPG_p.net_parg(data)

    #     scores = ( scores_c + scores_a + scores_p ) / 3
    if True:
        if NPG is None:
            NPG = NetPargnostics( reward_type, bin_num, 128, 1, 0.1, use_cuda)
        scores = NPG.net_parg(data)

    # data = data.cpu().numpy()
    # labels = labels.cpu().numpy()

    # scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()

    return scores

def angle_from_pair(a, b, bin_num, dim_num):
    line_gap = bin_num / (dim_num - 1)
    
    a_left = a[0]
    a_right = a[1]
    b_left = b[0]
    b_right = b[1]

    vector_a = ( line_gap, a_right - a_left )
    vector_b = ( line_gap, b_right - b_left )
    
    cos = np.dot(vector_a, vector_b) / ( np.linalg.norm(vector_a) * np.linalg.norm(vector_b) )

    angle = np.arccos(cos)
    # 0 ~ pi
    if angle < 0:
        angle += np.pi
    if angle >= np.pi / 2:
        angle = np.pi - angle
    return angle

def bin_coordinate(data, bin_num):
    data_num = len(data)

    # normalize
    max_data = np.max(data, axis=0).reshape(1, -1).repeat(data_num, 0)
    min_data = np.min(data, axis=0).reshape(1, -1).repeat(data_num, 0)
    norm_data = (data - min_data) / (max_data - min_data)

    # quantize dimension data
    bin_data = np.floor(norm_data * bin_num)
    bin_top_mask = (bin_data == bin_num)
    # the top of the data (1.0) belongs to the final bin
    bin_data[bin_top_mask] = bin_data[bin_top_mask] - 1
    return bin_data

def product(a, b):
    """
    Cartesian product of two array
    ---
    Parameters
    ----
    a: a float array. (data_num)
    b: a float array. (data_num)
    Returns:
    ---
    result: a float array. (data_num, 2)
    """
    data_num = len(a)
    a = a.repeat(data_num)
    b = b.reshape(1, -1).repeat(data_num, 0).flatten()
    return np.column_stack((a,b)).astype('int')

def axis_hist(bin_data, i, bin_num):
    '''
    Returns
    ----
    hist: array of one axis dimension (bin_num)
    '''
    hist = np.zeros(bin_num)

    counter = Counter(bin_data[:,i])
    keys = np.array(list(counter.keys())).astype('int')
    values = np.array(list(counter.values()))
    hist[keys] = values
    return hist

def pair_hist(bin_data, i, j, bin_num):
    """
    The two dimension axis histogram of each axis
    ----------
    Parameters
    ----------
        bin_data: array of size (, data_num, dim_num)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates
    Retruns
    ----
        hist: array (bin_num, bin_num)
    """
    
    # bin_num = len(bin_data)
    hist = np.zeros( (bin_num, bin_num) )
    
    bin_data = bin_data.astype('int')
    # axis_pair = product( bin_data[:,i], bin_data[:,j] )
    for p in bin_data[:, (i,j)]:
        hist[p[0], p[1]] += 1
    return hist

def line_cross(data, bin_data, i, j, bin_num, angle_bin_num):
    
    data_num = data.shape[0]
    dim_num = data.shape[1]

    # 0 ~ np.pi
    # angle_bin_num = 18
    angles = []
    
    total_cross = 0
    for data_i in range(data_num):
        for data_j in range(data_num):
            if data_i >= data_j: continue
            left_i = bin_data[data_i, i]
            left_j = bin_data[data_j, i]
            right_i = bin_data[data_i, j]
            right_j = bin_data[data_j, j]
            
            if (left_i < left_j and right_i > right_j) or (left_i > left_j and right_i < right_j):
                total_cross += 1
                angle = angle_from_pair(bin_data[data_i, (i,j)], bin_data[data_j, (i,j)], bin_num, dim_num)
                angles.append(angle)
                
    angles = np.array(angles)


    # quantize
    # angle_data = np.floor(angles / (np.pi/2 / angle_bin_num)). * (np.pi/2 / angle_bin_num)
    # hist = np.zeros(angle_bin_num)
    # counter = Counter(angle_data)
    # keys = np.array(list(counter.keys())).astype('int')
    # values = np.array(list(counter.values()))
    # hist[keys] = values

    if len(angles) == 0:
        ang = 0
    else:
        angle_data = np.floor(angles / (np.pi/2 / angle_bin_num)) * (np.pi/2 / angle_bin_num)
        ang = np.median(angle_data) / (np.pi/2)
    return 2 * total_cross / (data_num * (data_num-1)) , ang

def line_cross_num(data, bin_data, i, j, bin_num):
    
    data_num = data.shape[0]
    dim_num = data.shape[1]

    total_cross = 0
    for data_i in range(data_num):
        for data_j in range(data_num):
            if data_i >= data_j: continue
            left_i = bin_data[data_i, i]
            left_j = bin_data[data_j, i]
            right_i = bin_data[data_i, j]
            right_j = bin_data[data_j, j]
            
            if (left_i < left_j and right_i > right_j) or (left_i > left_j and right_i < right_j):
                total_cross += 1
                
    return 2 * total_cross / (data_num * (data_num-1))

def distance_hist(bin_data, i, j, bin_num):
    # bin_num = len(bin_data)
    dist_bin_num = bin_num * 2 - 1
    # let min is 0
    dist_data = bin_data[:,i] - bin_data[:,j] + bin_num-1

    hist = np.zeros(dist_bin_num)

    counter = Counter(dist_data)
    keys = np.array(list(counter.keys())).astype('int')
    values = np.array(list(counter.values()))
    hist[keys] = values
    return hist

def parallelism(bin_data, i, j, bin_num):
    # larger is more parallel
    dist_data = abs(bin_data[:,i] - bin_data[:,j])
    # norm to 0~1
    norm_dist_data = dist_data / (bin_num-1)
    # quantile
    q25, q75 =  np.percentile(norm_dist_data, [25, 75])
    # return 1 - abs(q75 - q25)
    return abs(q75 - q25)

def parallel_entropy(bin_data, i, j, bin_num):
    # the larger, the more parallel
    dist_hist = distance_hist(bin_data, i, j, bin_num)

    p_hist = dist_hist / dist_hist.sum()
    p_hist = p_hist[ p_hist != 0 ]    

    entropy = - np.sum( p_hist * np.log2(p_hist) )
    
    return entropy

def pearson_corrcoef(bin_data, i, j, bin_num):
    # r ∈ [-1, 1], Pearson’s correlation coefficient.
    # The closer to -1, the correlation more negative;
    # Other wise the closer to 1, the correlation more positive;
    # If r is around 0, no linear correlation
    
    # p: roughly indicates the probability of an uncorrelated system 
    # producing datasets that have a Pearson correlation at least as 
    # extreme as the one computed from these datasets.
    # p值越小，表示相关系数越显著，一般在500个样本以上时有较高可靠性
    
    r, p = pearsonr(bin_data[:, i], bin_data[:, j])

    return r

def pearson_corrcoef_tensor(bin_data, i, j, bin_num):
    x = torch.tensor(bin_data[:, i])
    y = torch.tensor(bin_data[:, j])
    n = len(x)
    sum_xy = torch.sum(x * y)
    sum_x  = torch.sum(x)
    sum_y  = torch.sum(y)
    sum_x2 = torch.sum(x * x)
    sum_y2 = torch.sum(y * y)
    pcc = ( (n*sum_x2 - sum_x*sum_x) * (n*sum_y2 - sum_y*sum_y) ).float()
    pccs = ( n*sum_xy - sum_x*sum_y ) / torch.sqrt(pcc)
    return pccs

def mutual(bin_data, i, j, bin_num):
    # close to 0 is independent
    p_hist = pair_hist(bin_data, i, j, bin_num) / bin_num
    b_hist_i = axis_hist(bin_data, i, bin_num) / bin_num
    b_hist_j = axis_hist(bin_data, j, bin_num) / bin_num

    cof = 0
    for bin_i in range(bin_num):
        for bin_j in range(bin_num):
            p_ij = p_hist[bin_i, bin_j]
            if p_ij == 0:
                continue
            p_i = b_hist_i[bin_i]
            p_j = b_hist_j[bin_j]
            cof += p_ij * np.log( p_ij / (p_i * p_j) )

    return cof

def convergence(bin_data, i, j, bin_num):
    p_hist = pair_hist(bin_data, i, j, bin_num)

    cov = 0
    for bin_i in range(bin_num):
        for bin_j in range(bin_num):
            if p_hist[bin_j, bin_i] > 0:
                cov += 1    
    return cov / np.max(p_hist)

def divergence(bin_data, i, j, bin_num):
    p_hist = pair_hist(bin_data, i, j, bin_num)

    div = 0
    for bin_i in range(bin_num):
        for bin_j in range(bin_num):
            if p_hist[bin_i, bin_j] > 0:
                div += 1    
    return div / np.max(p_hist)

def overplotting(data, bin_data, i, j, bin_num):
    data_num = data.shape[0]

    p_hist = pair_hist(bin_data, i, j, bin_num)

    over = 0
    for bin_i in range(bin_num):
        for bin_j in range(bin_num):
            if p_hist[bin_i, bin_j] > 1:
                over += p_hist[bin_i, bin_j]
    return 2 * over / ( data_num * (data_num-1) )

def one_dimension_axis_histogram(data, bin_num):
    """
    The one dimension axis histogram of each axis
    ----------
    Parameters
    ----------
        data: array of size (, data_num, dim_num)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates
    Retruns
    ----
        scores: array of evaluation scores (batch_size)
    """

    data_num = data.shape[0]
    dim_num = data.shape[1]

    bin_data = bin_coordinate(data, bin_num)

    hist = np.zeros((dim_num, bin_num))

    for dim_index in range(dim_num):
        counter = Counter(bin_data[:,dim_index])
        keys = np.array(list(counter.keys())).astype('int')
        values = np.array(list(counter.values()))
        hist[dim_index][keys] = values
    
    return hist

def one_dimension_distance_histogram(data, bin_num):
    """
    The one dimension distance histogram of each axis
    ----------
    Parameters
    ----------
        data: array of size (, data_num, dim_num)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates
    Retruns
    ----
        d_hist: array of distance (dim_num, dim_num, dist_bin_num)
    """

    data_num = data.shape[0]
    dim_num = data.shape[1]
    dist_bin_num = bin_num * 2 - 1

    d_hist = np.zeros((dim_num, dim_num, dist_bin_num))
    bin_data = bin_coordinate(data, bin_num)
        
    for i in range(dim_num):
        for j in range(dim_num):
            if i == j: continue
            elif i > j:
                d_hist[i][j] = -d_hist[j][i]
            else:
                d_hist[i][j] = distance_hist(bin_data, i, j, bin_num)
    
    return d_hist

def two_dimension_axis_histogram(data, bin_num):
    """
    The two dimension axis histogram of each axis
    ----------
    Parameters
    ----------
        data: array of size (, data_num, dim_num)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates
    Retruns
    ----
        hist: array (dim_num, dim_num, bin_num, bin_num)
    """
    data_num = data.shape[0]
    dim_num = data.shape[1]
    
    p_hist = np.zeros((dim_num, dim_num, bin_num, bin_num))
    bin_data = bin_coordinate(data, bin_num)

    for i in range(dim_num):
        for j in range(dim_num):
            if i==j: continue
            elif i>j: p_hist[i][j] = p_hist[j][i].transpose()
            else:
                p_hist[i][j] = pair_hist(bin_data, i, j, bin_num)

    return p_hist

def line_crossing(data, bin_num, angle_bin_num):
    data_num = data.shape[0]
    dim_num = data.shape[1]
    
    bin_data = bin_coordinate(data, bin_num)

    a_hist = np.zeros( (dim_num, dim_num, angle_bin_num) )

    total_cross = 0
    total_angles = 0

    for i in range(dim_num):
        for j in range(dim_num):
            cross, angles = line_cross(data, bin_data, i, j, angle_bin_num)
            total_cross += cross
    # norm of cross
    return 2 * total_cross / (data_num *  (data_num-1))


class PargNet(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(PargNet, self).__init__()

        self.gru = nn.GRU( input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.drop_hh = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            # nn.Sigmoid()
        )

    def forward(self, data):
        # batch_size x data_num x 2
        use_cuda = data.is_cuda
        
        batch_size = data.shape[0]
        data_num = data.shape[1]

        last_hh = None
        rnn_out, last_hh = self.gru(data, last_hh)
        
        if self.num_layers == 1:
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)
        
        last_hh = last_hh.squeeze(0)
        output = self.fc(last_hh)
        
        return output

class NetPargnostics(object):
    def __init__(self, evaluate_type, bin_num, hidden_size, num_layers, dropout, use_cuda):
        self.net = PargNet(2, hidden_size, num_layers, dropout)
        # checkpoint = './parg_task/%s/200b-200-note-train-rel-10/checkpoints/99' % evaluate_type
        # checkpoint = './parg_task/%s/50b-200-note-train/checkpoints/299' % evaluate_type
        checkpoint = '/newhome/juzhan/ws/Coor_Opt/parg_task/%s/50b-200-note-train/checkpoints/299' % evaluate_type
        path = os.path.join(checkpoint, 'net.pt')
        self.net.load_state_dict(torch.load(path))
        
        self.bin_num = bin_num
        self.use_cuda = use_cuda
        self.evaluate_type = evaluate_type

        if use_cuda:
            self.net = self.net.cuda()
        self.net.eval()

    def net_parg( self, data ):

        batch_size = data.shape[0]
        data_num = data.shape[1]
        dim_num = data.shape[2]

        # max_data = np.max(data, axis=1).reshape(batch_size, 1, -1).repeat(data_num, 1)
        # min_data = np.min(data, axis=1).reshape(batch_size, 1, -1).repeat(data_num, 1)
        # norm_data = (data - min_data) / (max_data - min_data)

        # quantize dimension data
        bin_data = torch.floor(data * self.bin_num)
        bin_top_mask = (bin_data == self.bin_num)
        # the top of the data (1.0) belongs to the final bin
        bin_data[bin_top_mask] = bin_data[bin_top_mask] - 1

        # bin_data = torch.from_numpy(bin_data)
        if self.use_cuda:
            bin_data = bin_data.cuda()

        with torch.no_grad():
            prediction = 0
            for i in range(dim_num-1):
                net_pre = self.net(bin_data[:,:,i:i+2])
                if self.evaluate_type == 'pccs':
                    net_pre = -abs(net_pre)
                prediction += net_pre
            prediction /= (dim_num-1)

        return prediction





if __name__ == "__main__":

    data_num = 50
    dim_num = 8
    bin_num = 10
    angle_bin_num = 9
    
    data = np.random.rand(data_num, dim_num)



    # data_num = 10
    # dim_num = 6
    # bin_num = 5
    # angle_bin_num = 9
    
    # data = [ i for i in range(10)]
    # data = np.array(data).reshape(-1, 1).repeat(dim_num, -1).astype('float32')
    # # data[:,1] = data[:,1][::-1]
    # # data[:,0] = data[:,0] * 10
    
    # data[1,0] = 1.5
    # data[6,0] = 5.2
    # data[:,1] = [1,1,2,3,8,9,6,0,6,3]
    # data[:,3] = data[:,2] - 1
    # data[0,2] = 1
    # data[0,3] = 0

    # data[:,4] = [ 5 for i in range(data_num)]

    # rand data
    

    # normalize
    # max_data = np.max(data, axis=0).reshape(1, -1).repeat(data_num, 0)
    # min_data = np.min(data, axis=0).reshape(1, -1).repeat(data_num, 0)
    # max_data = np.max(data)
    # min_data = np.min(data)
    # norm_data = (data - min_data) / (max_data - min_data)

    data_num = data.shape[0]
    dim_num = data.shape[1]
    dist_bin_num = bin_num * 2 - 1

    bin_data = bin_coordinate(data, bin_num)

    order = [ i for i in range(dim_num)]
    # draw_parallel(data, order, '', 'para2')

    scores = np.zeros( (dim_num, dim_num) )
    for i in range(dim_num):
        for j in range(dim_num):
            if i == j: continue
            elif i > j: scores[i,j] = scores[j,i]
            else:
                cross, angle = line_cross(data, bin_data, i, j, bin_num, angle_bin_num)
                param = parallelism(bin_data, i, j, bin_num)
                scores[i,j] = (cross + angle + param) / 3

    # over = overplotting(data, bin_data, i, j, bin_num)
    
    # mut = mutual(bin_data, i, j, bin_num)
    # cov = convergence(bin_data, i, j, bin_num)
    # div = divergence(bin_data, i, j, bin_num)


    # entropy stop
    # a = 0