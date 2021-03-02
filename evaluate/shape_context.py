import os
import torch
import torch.nn as nn
import numpy as np
import math
import cv2
from collections import Counter
import copy

def radial_edges(r1, r2, n):
    #return a list of radial edges from an inner (r1) to an outer (r2) radius
    re = np.logspace(np.log10(r1), np.log10(r2), n)
    return re


def euclid_distance(p1,p2):
    return math.sqrt( ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) ** 2 )


def get_angle(p1,p2):
    #compute the angle between points.
    return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))


def normalize(data, data_num):
    return data / data_num
# def normalize(data):
#     data_min = np.min(data)
#     data_max = np.max(data)
#     return (data - data_min) / (data_max - data_min)

class ShapeContext(object):
    """
    Given a point in the image, for all other points that are within a given
    radius, computes the relative angles.
    Radii and angles are stored in a  "shape matrix" with dimensions: radial_bins x angle_bins.
    Each element (i,j) of the matrix contains a counter/integer that corresponds to,
    for a given point, the number of points that fall at that i radius bin and at
    angle bin j. 
    """

    def __init__(self,nbins_r=5,nbins_theta=12,r_inner=0.1250,r_outer=2,wlog=False):
        self.nbins_r        = nbins_r             # number of bins in a radial direction
        self.nbins_theta    = nbins_theta         # number of bins in an angular direction
        self.r_inner        = r_inner             # inner radius
        self.r_outer        = r_outer             # outer radius
        self.nbins          = nbins_theta*nbins_r # total number of bins
        self.wlog           = wlog                # using log10(r) or Normalize with the mean
        # get radius range
        self.radial_range = radial_edges(self.r_inner, self.r_outer, self.nbins_r)
        # get angle range
        self.angle_range = np.linspace(0, 2*np.pi, self.nbins_theta+1)[1:-1]


    def distM(self, x):
        """
        Compute the distance matrix

        Params:
        -------
        x: a list with points tuple(x,y) in an image

        Returns:
        --------
        result: a distance matrix with euclidean distance
        """
        x_len = len(x)
        result = np.zeros((x_len, x_len))
        for i, _x in enumerate(x):
            points = np.repeat([_x], x_len, axis=0)
            result[i] = np.linalg.norm(points - x, axis=1)
        return result

    def angleM(self, x):
        """
        Compute the distance matrix

        Params:
        -------
        x: a list with points tuple(x,y) in an image

        Returns:
        --------
        result: a distance matrix with euclidean distance
        """
        x_len = len(x)
        result = np.zeros((x_len, x_len))
        for i, _x in enumerate(x):
            points = np.repeat([_x], x_len, axis=0)
            residual = x - points
            result[i] = np.arctan2(residual[:,1], residual[:,0])
        return result

    def compute(self,points):

        # distance matrix
        dist_array = self.distM(points)

        # Normalize the distance matrix by the mean distance or use log10
        if self.wlog:
            dist_array = np.log10(dist_array+1)
        else:
            # mean_dist = dist_array.mean()
            # dist_array = dist_array / mean_dist
            median_dist = np.median(dist_array)
            dist_array = dist_array / median_dist

        ang_array = self.angleM(points)
        # let ang 0 ~ 2pi
        ang_array = ang_array + 2*np.pi * (ang_array < 0)
        
        # static place
        points_len = len(points)
        radial_index = np.zeros((points_len, points_len))
        angle_index = np.zeros((points_len, points_len))
        for rad in self.radial_range:
            radial_index += (dist_array >= rad) # from 0 ~ nbins_r-1
        radial_bin = dist_array >= self.radial_range[-1]

        for ang in self.angle_range:
            angle_index += (ang_array >= ang) # from 1 ~ nbin_t

        BH = np.zeros((points_len, self.nbins))
        BH_index = (radial_index * self.nbins_theta + angle_index).astype('int')
        for i in range(points_len):
            counter = Counter(BH_index[i])
            keys = np.array(list(counter.keys()))
            key_mask = keys < self.nbins
            values = np.array(list(counter.values()))
            BH[i][ keys[key_mask] ] = values[key_mask]
            if BH_index[i, i] == 0:
                BH[i][0] -= 1

        # normalize
        for i in range(points_len):
            sm = normalize(BH[i], points_len-1)
            BH[i] = sm
        
        return BH

    def cost_ij(self, b1, b2, i, j):
        '''
        Compute the distribution distance of point i in b1 and point j in b2

        Params:
        -------
        b1: a distribution from compute()
        b2: a distribution from compute()
        i: int, point's index
        j: int, point's index

        Returns:
        --------
        cost: a distance of the distance bewteen i and j's distribution
        '''
        dist_1 = b1[i]
        dist_2 = b2[j]
        dist_sum = dist_1 + dist_2
        return 0.5 * sum( ((dist_1 - dist_2) ** 2) / (dist_sum + (dist_sum == 0)) )

    def cost(self, b1, b2):
        '''
        Compute the shape context of b1 and b2

        Params:
        -------
        b1: a distribution from compute()
        b2: a distribution from compute()

        Returns:
        --------
        cost: a total sum of the cost bewteen related points in each distribution
        '''
        ret = 0
        for i in range(len(b1)):
            ret += self.cost_ij(b1, b2, i, i)
        ret = ret / len(b1)
        return ret

    def cost_list(self, b1, b2):
        '''
        Compute the shape context of b1 and b2

        Params:
        -------
        b1: a distribution from compute()
        b2: a distribution from compute()

        Returns:
        --------
        cost: a total sum of the cost bewteen related points in each distribution
        '''
        ret = []
        for i in range(len(b1)):
            ret.append( self.cost_ij(b1, b2, i, i))
        return ret

def get_shape_old(data, sample_num=10):
    '''
    Compute the shape points of radar data

    Params:
    ----
    data: list or array of radar data [(x,y),...]

    Returns:
    ---
    shapes: list of the shape points of each radar data
    '''
    sourceL = len(data[0])
    angles = np.linspace(0, 2*np.pi, sourceL, endpoint=False)

    cos_sin = np.array( [ (np.cos(ang), (np.sin(ang))) for ang in angles ])

    shapes = []
    positions = []

    # order = [7, 3, 6, 0, 4, 1, 5, 2]
    # order = [7, 4, 3, 6, 2, 5, 1, 0]
    for i in range(len(data)):
        d = np.array(data[i]) #[order]
        
        position = [ np.multiply(d[j], cos_sin[j]) for j in range(sourceL)  ]
        # np.concatenate((position, position[0]))
        position.append(position[0])
        positions.append(position)
        points = []
        for i in range(sourceL):
            # points.extend(np.linspace(position[i], position[(i+1) % sourceL], sample_num)[:-1])
            # points.extend(np.linspace(position[i], position[i+1], sample_num)[:-1])
            points.extend(np.linspace(position[i], position[i+1], sample_num-1, endpoint=False))
        points = np.array(points)
        shapes.append(points)
    return shapes


def get_shape(data, sample_num=80):
    '''
    Compute the shape points of radar data

    Params:
    ----
    data: list or array of radar data [(x,y),...]

    Returns:
    ---
    shapes: list of the shape points of each radar data
    '''
    sourceL = len(data[0])
    angles = np.linspace(0, 2*np.pi, sourceL, endpoint=False)

    cos_sin = np.array( [ (np.cos(ang), (np.sin(ang))) for ang in angles ])

    shapes = []
    positions = []

    for i in range(len(data)):
        d = np.array(data[i])
        
        position = [ np.multiply(d[j], cos_sin[j]) for j in range(sourceL)  ]
        position.append(position[0])
        positions.append(position)
        points = []

        mean_num = math.floor( sample_num / sourceL )
        
        # for i in range(sourceL):
        for i in range(sourceL - 1):
            points.extend(np.linspace(position[i], position[i+1], mean_num, endpoint=False))

        i = sourceL - 1
        final_num = sample_num - mean_num * (sourceL-1)
        points.extend(np.linspace(position[i], position[i+1], final_num, endpoint=False))
        points = np.array(points)
        shapes.append(points)
    
    shapes = np.reshape( shapes, (-1, 2) )
    
    return shapes

class SCNet(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(SCNet, self).__init__()

        self.gru = nn.GRU( input_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)

        self.drop_hh = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size/2)),
            nn.ReLU(),
            nn.Linear( int(hidden_size/2) , 1),
            nn.Sigmoid()
        )
        

    def forward(self, data):
        # batch_size x points_num x 2
        use_cuda = data.is_cuda
        
        batch_size = data.shape[0]
        points_num = data.shape[1]
        data_num = data.shape[2]

        half_points_num = int(points_num / 2)

        vectors = torch.zeros( batch_size, self.hidden_size, data_num )
        if use_cuda:
            vectors = vectors.cuda()

        # batch_size x points_num x data_num
        data_input_1 = data[:, 0:half_points_num  , :]
        data_input_2 = data[:, half_points_num:half_points_num*2  , :]
        data_input = torch.cat( (data_input_1, data_input_2), dim=-1 )
        
        last_hh = None
        rnn_out, last_hh = self.gru(data_input, last_hh)
        
        if self.num_layers == 1:
            last_hh = self.drop_hh(last_hh)

        output = last_hh.squeeze(0)
        output = self.fc(output)
        return output


class NetShapeContext(object):
    def __init__(self, hidden_size, num_layers, dropout, use_cuda):
        self.net = SCNet(4, hidden_size, num_layers, dropout)    
        checkpoint = './shape_task/shape/24d-fcn2/checkpoints/99'
        
        path = os.path.join(checkpoint, 'net.pt')
        self.net.load_state_dict(torch.load(path))
        
        self.use_cuda = use_cuda

        if use_cuda:
            self.net = self.net.cuda()
        self.net.eval()

    def net_sc(self, points1, points2, data ):

        batch_size = data.shape[0]
        # data_num = data.shape[1]
        # dim_num = data.shape[2]

        points1 = np.reshape(points1, (-1, batch_size, 2)).astype('float32')
        points2 = np.reshape(points2, (-1, batch_size, 2)).astype('float32')
        
        points1 = torch.from_numpy(points1).transpose(1, 0)
        points2 = torch.from_numpy(points2).transpose(1, 0)

        if self.use_cuda:
            points1 = points1.cuda()
            points2 = points2.cuda()

        points = torch.cat( (points1, points2), dim=1 )
        with torch.no_grad():
            prediction = self.net(points)

        return prediction


def shape_context_distance(data, labels, with_label=True, label_num=-1, avg_type='sil', NSC=None):
    """
    Parameters
    ----
        data: Tensor of data (batch_size, data_num, dim_num)
    """
    use_cuda = data.device.type == 'cuda'

    data_norm = data.norm(dim=2).unsqueeze(2).expand_as(data)
    data = data / data_norm
    
    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]

    if NSC is None:
        hidden_size = 128
        NSC = NetShapeContext(hidden_size, 1, 0.1, use_cuda)

    data = data.cpu().numpy()
    # labels = labels.cpu().numpy()

    # data_mean = data.mean(axis=2).reshape(batch_size, data_num, 1).repeat(dim_num, 2)
    # data = data / data_mean

    scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()

    # get label num
    if label_num == -1:
        label_num = 2
    
    sample_num = 80

    angles = np.linspace(0, 2*np.pi, dim_num+1)[:-1]
    cos_sin = np.array( [ (np.cos(ang), (np.sin(ang))) for ang in angles ])

    points = []
    for data_index in range(data_num):
        # dim_num x batch_size x 2
        pos = [ data[:,data_index,i].reshape(-1, 1) * cos_sin[i] for i in range(dim_num) ]
        # dim_num x sample_num x batch_size x 2

        # 1
        # point = [ np.linspace(pos[i], pos[ (i+1)%dim_num ], sample_num+1)[:-1] for i in range(dim_num) ]
        # point = np.reshape(point, (-1, batch_size, 2)).astype('float32')
        
        # 2
        point = []
        mean_num = math.floor( sample_num / dim_num )
        for i in range(dim_num - 1):
            point.extend(np.linspace(pos[i], pos[i+1], mean_num, endpoint=False))
        i = dim_num - 1
        final_num = sample_num - mean_num * (dim_num-1)
        point.extend(np.linspace(pos[i], pos[0], final_num, endpoint=False))

        # ------------

        point = np.reshape(point, (-1, batch_size, 2)).astype('float32')
        points.append(
            point
        )



    my_score = torch.zeros((batch_size, data_num, data_num))
    my_label = torch.zeros((batch_size, data_num, data_num))
    if use_cuda:
        my_score = my_score.cuda()
        my_label = my_label.cuda()
    
    for data_i in range(data_num):
        for data_j in range(data_num):
            if data_i >= data_j:
                if avg_type == 'sil' or avg_type == 'mean' or avg_type == 'sil_list':
                    my_score[:, data_i, data_j] = my_score[:, data_j, data_i]
                continue
            points_i = points[data_i]
            points_j = points[data_j]
            my_score[:, data_i, data_j] = NSC.net_sc(points_i, points_j, data).squeeze(-1)

            # batch_size x data_num x dim_num
            diff_label_mask = (labels[:,data_i,0] != labels[:,data_j,0])
            # same_label_mask = (labels[:,data_i,0] == labels[:,data_j,0])

            if avg_type == 'simple':
                my_score[diff_label_mask, data_i, data_j] = 1 - my_score[diff_label_mask, data_i, data_j]
            # my_label[diff_label_mask, data_i, data_j] = -1
            # my_label[same_label_mask, data_i, data_j] = labels[same_label_mask,data_i,0] + 1


    if avg_type == 'simple':
        scores = torch.sum(my_score, dim=(1,2)) / ( data_num * (data_num-1) / 2 ) 
    elif avg_type == 'mean':
        
        # get between class
        total_labels = labels[:,:,0].clone()

        s_mean = torch.zeros(batch_size)

        score_in_class = []
        score_between_class = []
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = torch.zeros(batch_size)
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            # TODO 
            # if len(mask_i) == 1:
            #     s_mean = torch.zeros(batch_size)
            if mask_i.sum() == 0:
                s_mean = torch.zeros(batch_size)
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    i_len = mask_i.sum(dim=1).float()
                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                        a_i = a_i.sum(dim=-1) / ( i_len[:,0] )

                        score_in_class.append(a_i)
                    else:
                        b_i = score.sum(dim=-1) / j_len
                        b_i = b_i.sum(dim=-1) / j_len[:,0]
                        
                        score_between_class.append(b_i)

        in_score = 0
        for score_in in score_in_class:
            in_score += score_in
        in_score = in_score / len(score_in_class)
        between_score = 0
        for score_between in score_between_class:
            between_score += score_between
        between_score = between_score / len(score_between_class)

        scores = -(between_score - in_score)

    elif avg_type == 'sil':
        # get between class
        total_labels = labels[:,:,0].clone()

        s_mean = torch.zeros(batch_size)
        if use_cuda:
            s_mean = s_mean.cuda()
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = torch.zeros(batch_size)
                if use_cuda:
                    s_mean = s_mean.cuda()
                
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            i_len = mask_i.sum(dim=1).float()
            
            # TODO 
            if mask_i.sum() == 0:
                s_mean = torch.zeros(batch_size)
                if use_cuda:
                    s_mean = s_mean.cuda()
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                    else:
                        if b_i is None:
                            b_i = score.sum(dim=-1) / j_len
                        else:
                            b_i = torch.min( b_i, score.sum(dim=-1) / j_len )
                # print(b_i.mean(), a_i.mean())
                s_i = (b_i - a_i) / torch.max( a_i, b_i )
                # s_i = s_i.mean(dim=-1)
                s_i[ torch.isnan(s_i) ] = 0
                s_i = s_i.sum(dim=-1) / i_len[:,0]
                s_mean = torch.max( s_i, s_mean )


        # s_mean[i_len[:,0] == 1] = 0
        scores = s_mean

        scores = -scores

    elif avg_type == 'sil_list':
        # get between class
        total_labels = labels[:,:,0].clone()

        # s_mean = torch.zeros(batch_size)
        # if use_cuda:
        #     s_mean = s_mean.cuda()
        s_mean = []

        for label_i in range(label_num):
            # if label_num == 1:
            #     s_mean = torch.zeros(batch_size)
            #     if use_cuda:
            #         s_mean = s_mean.cuda()
            #     continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            i_len = mask_i.sum(dim=1).float()
            
            # TODO 
            # if mask_i.sum() == 0:
            #     s_mean = torch.zeros(batch_size)
            #     if use_cuda:
            #         s_mean = s_mean.cuda()
            # else:
            if True:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                    else:
                        if b_i is None:
                            b_i = score.sum(dim=-1) / j_len
                        else:
                            b_i = torch.min( b_i, score.sum(dim=-1) / j_len )
                # print(b_i.mean(), a_i.mean())
                s_i = (b_i - a_i) / torch.max( a_i, b_i )
                # s_i = s_i.mean(dim=-1)
                s_i[ torch.isnan(s_i) ] = 0
                s_i = s_i.sum(dim=-1) / i_len[:,0]
                # s_mean = torch.max( s_i, s_mean )
                
                s_mean.append( s_i )

        return s_mean


    
    
    else:
        print('====> Error in shape context distance')

    scores = scores.detach()
    if use_cuda:
        scores = scores.cuda()  

    return scores


def orignal_space_distance(data, labels, with_label=True, label_num=-1, avg_type='sil'):
    """
    Parameters
    ----
        data: Tensor of data (batch_size, data_num, dim_num)
    """
    use_cuda = data.device.type == 'cuda'
    
    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]


    scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()

    # get label num
    if label_num == -1:
        label_num = 2
    

    my_score = torch.zeros((batch_size, data_num, data_num))
    if use_cuda:
        my_score = my_score.cuda()
    
    for data_i in range(data_num):
        for data_j in range(data_num):
            if data_i >= data_j:
                my_score[:, data_i, data_j] = my_score[:, data_j, data_i]                    
                continue
            # calc distance
            my_score[:, data_i, data_j] = torch.norm( (data[:, data_i]-data[:, data_j]), dim=-1 )

        
    if avg_type == 'mean':
        
        # get between class
        total_labels = labels[:,:,0].clone()

        s_mean = torch.zeros(batch_size)

        score_in_class = []
        score_between_class = []
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = torch.zeros(batch_size)
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            # TODO 
            # if len(mask_i) == 1:
            #     s_mean = torch.zeros(batch_size)
            if mask_i.sum() == 0:
                s_mean = torch.zeros(batch_size)
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    i_len = mask_i.sum(dim=1).float()
                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                        a_i = a_i.sum(dim=-1) / ( i_len[:,0] )

                        score_in_class.append(a_i)
                    else:
                        b_i = score.sum(dim=-1) / j_len
                        b_i = b_i.sum(dim=-1) / j_len[:,0]
                        
                        score_between_class.append(b_i)

        in_score = 0
        for score_in in score_in_class:
            in_score += score_in
        in_score = in_score / len(score_in_class)
        between_score = 0
        for score_between in score_between_class:
            between_score += score_between
        between_score = between_score / len(score_between_class)

        scores = -(between_score - in_score)

    elif avg_type == 'sil':
        # get between class
        total_labels = labels[:,:,0].clone()

        s_mean = torch.zeros(batch_size)
        if use_cuda:
            s_mean = s_mean.cuda()
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = torch.zeros(batch_size)
                if use_cuda:
                    s_mean = s_mean.cuda()
                
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            i_len = mask_i.sum(dim=1).float()
            
            # TODO 
            if mask_i.sum() == 0:
                s_mean = torch.zeros(batch_size)
                if use_cuda:
                    s_mean = s_mean.cuda()
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                    else:
                        if b_i is None:
                            b_i = score.sum(dim=-1) / j_len
                        else:
                            b_i = torch.min( b_i, score.sum(dim=-1) / j_len )

                s_i = (b_i - a_i) / torch.max( a_i, b_i )
                s_i[ torch.isnan(s_i) ] = 0
                s_i = s_i.sum(dim=-1) / i_len[:,0]
                s_mean = torch.max( s_i, s_mean )

        scores = -s_mean

    else:
        print('====> Error in original space distance')

    scores = scores.detach()
    if use_cuda:
        scores = scores.cuda()  

    return scores

def silhouette_rate(data, labels, with_label=True, label_num=-1, avg_type='sil', NSC=None):
    silhouette_2d = shape_context_distance(data, labels, with_label, label_num, avg_type, NSC)
    silhouette_nd = orignal_space_distance(data, labels, with_label, label_num, avg_type)
    return silhouette_nd / silhouette_2d


def shape_context_distance_local(data, labels, with_label=True, label_num=-1, avg_type='sil', NSC=None):
    """
    Parameters
    ----
        data: array of data (1, data_num, dim_num)
    """
    if type(data) == torch.Tensor:
        data = data.cpu().numpy()
    if type(labels) == torch.Tensor:
        labels = labels.cpu().numpy()

    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]

    data_norm = np.linalg.norm( data, axis=2).reshape( (batch_size, data_num, 1) ).repeat( dim_num, 2 )
    data = data / data_norm
    
    scores = np.zeros(batch_size)

    # get label num
    if label_num == -1:
        label_num = 2
    
    sample_num = 80

    angles = np.linspace(0, 2*np.pi, dim_num+1)[:-1]
    cos_sin = np.array( [ (np.cos(ang), (np.sin(ang))) for ang in angles ])

    points = []
    for data_index in range(data_num):
        # dim_num x batch_size x 2
        pos = [ data[:,data_index,i].reshape(-1, 1) * cos_sin[i] for i in range(dim_num) ]
        # dim_num x sample_num x batch_size x 2

        # 2
        point = []
        mean_num = math.floor( sample_num / dim_num )
        for i in range(dim_num - 1):
            point.extend(np.linspace(pos[i], pos[i+1], mean_num, endpoint=False))
        i = dim_num - 1
        final_num = sample_num - mean_num * (dim_num-1)
        point.extend(np.linspace(pos[i], pos[0], final_num, endpoint=False))

        # ------------

        point = np.reshape(point, (-1, 2)).astype('float32')
        points.append(
            point
        )

    sc = ShapeContext()

    my_score = np.zeros((batch_size, data_num, data_num))
    my_label = np.zeros((batch_size, data_num, data_num))
    
    
    for data_i in range(data_num):
        for data_j in range(data_num):
            if data_i >= data_j:
                if avg_type == 'sil' or avg_type == 'sil_list':
                    my_score[:, data_i, data_j] = my_score[:, data_j, data_i]                    
                continue
            points_i = points[data_i]
            points_j = points[data_j]
            # my_score[:, data_i, data_j] = NSC.net_sc(points_i, points_j, data).squeeze(-1)
            if True:
                h1 = sc.compute(points_i)
                h2 = sc.compute(points_j)
                my_score[:, data_i, data_j] = sc.cost(h1, h2)

            # for batch_index in range(batch_size):
            #     my_score[batch_index, data_i, data_j] 
            
            # if with_label:
            # batch_size x data_num x dim_num
            diff_label_mask = (labels[:,data_i,0] != labels[:,data_j,0])
            same_label_mask = (labels[:,data_i,0] == labels[:,data_j,0])

            if avg_type == 'simple':
                my_score[diff_label_mask, data_i, data_j] = 1 - my_score[diff_label_mask, data_i, data_j]
            my_label[diff_label_mask, data_i, data_j] = -1
            my_label[same_label_mask, data_i, data_j] = labels[same_label_mask,data_i,0] + 1

            # else:
            #     my_label[:, data_i, data_j] = 1

    if avg_type == 'simple':
        scores = np.sum(my_score, axis=(1,2)) / ( data_num * (data_num-1) / 2 )
    elif avg_type == 'sil':
        # get between class
        total_labels = copy.copy( labels[:,:,0] )

        s_mean = np.zeros(batch_size)
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = np.zeros(batch_size)
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.reshape( (batch_size, data_num, -1) ).repeat( data_num, 2 )
            
            # TODO 
            if mask_i.sum() == 0:
                s_mean = np.zeros(batch_size)
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
        
                    mask_j = mask_j.reshape( (batch_size, 1, data_num) ).repeat( data_num, 1 )
        
                    mask_ij = mask_i * mask_j

                    i_len = mask_i.sum(axis=1)
                    j_len = mask_j.sum(axis=-1)

                    score = np.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(axis=-1) / ( i_len - 1 )
                    else:
                        if b_i is None:
                            b_i = score.sum(axis=-1) / j_len
                        else:
                            b_i = np.minimum( b_i, score.sum(axis=-1) / j_len )

                max_mask = (np.maximum( a_i, b_i ) == 0)
                s_i = (b_i - a_i) / np.maximum( a_i, b_i )
                s_i[ max_mask ] = 0
                # s_i = s_i.mean(dim=-1)
                s_i = s_i.sum(axis=-1) / i_len[:,0]
                # if s_i > s_mean:
                #     s_mean = s_i
                s_mean = np.maximum( s_i, s_mean )


        # s_mean[i_len[:,0] == 1] = 0
        scores = s_mean

        scores = -scores

    else:
        print('====> Error in shape context distance')

    return scores[0]


def shape_context_distance_tradition(data, labels, with_label=True, label_num=-1, avg_type='sil', NSC=None):
    """
    Parameters
    ----
        data: Tensor of data (batch_size, data_num, dim_num)
    """
    use_cuda = data.device.type == 'cuda'

    data_norm = data.norm(dim=2).unsqueeze(2).expand_as(data)
    data = data / data_norm
    
    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]

    if NSC is None:
        hidden_size = 128
        NSC = NetShapeContext(hidden_size, 1, 0.1, use_cuda)

    data = data.cpu().numpy()
    # labels = labels.cpu().numpy()

    # data_mean = data.mean(axis=2).reshape(batch_size, data_num, 1).repeat(dim_num, 2)
    # data = data / data_mean

    scores = torch.zeros(batch_size).detach()
    if use_cuda:
        scores = scores.cuda()

    # get label num
    if label_num == -1:
        label_num = 2
    
    sample_num = 80

    angles = np.linspace(0, 2*np.pi, dim_num+1)[:-1]
    cos_sin = np.array( [ (np.cos(ang), (np.sin(ang))) for ang in angles ])

    points = []
    for data_index in range(data_num):
        # dim_num x batch_size x 2
        pos = [ data[:,data_index,i].reshape(-1, 1) * cos_sin[i] for i in range(dim_num) ]
        # dim_num x sample_num x batch_size x 2

        # 1
        # point = [ np.linspace(pos[i], pos[ (i+1)%dim_num ], sample_num+1)[:-1] for i in range(dim_num) ]
        # point = np.reshape(point, (-1, batch_size, 2)).astype('float32')
        
        # 2
        point = []
        mean_num = math.floor( sample_num / dim_num )
        for i in range(dim_num - 1):
            point.extend(np.linspace(pos[i], pos[i+1], mean_num, endpoint=False))
        i = dim_num - 1
        final_num = sample_num - mean_num * (dim_num-1)
        point.extend(np.linspace(pos[i], pos[0], final_num, endpoint=False))

        # ------------

        point = np.reshape(point, (-1, batch_size, 2)).astype('float32')
        points.append(
            point
        )


    shape_context = ShapeContext()

    my_score = torch.zeros((batch_size, data_num, data_num))
    my_label = torch.zeros((batch_size, data_num, data_num))
    if use_cuda:
        my_score = my_score.cuda()
        my_label = my_label.cuda()
    
    for data_i in range(data_num):
        for data_j in range(data_num):
            if data_i >= data_j:
                if avg_type == 'sil' or avg_type == 'mean' or avg_type == 'sil_list':
                    my_score[:, data_i, data_j] = my_score[:, data_j, data_i]
                continue
            points_i = points[data_i]
            points_j = points[data_j]
            my_score[:, data_i, data_j] = shape_context.cost(
                shape_context.compute( points_i.reshape(sample_num, -1) ),
                shape_context.compute( points_j.reshape(sample_num, -1) )
            )
            # NSC.net_sc(points_i, points_j, data).squeeze(-1)

            # batch_size x data_num x dim_num
            diff_label_mask = (labels[:,data_i,0] != labels[:,data_j,0])
            # same_label_mask = (labels[:,data_i,0] == labels[:,data_j,0])

            if avg_type == 'simple':
                my_score[diff_label_mask, data_i, data_j] = 1 - my_score[diff_label_mask, data_i, data_j]
            # my_label[diff_label_mask, data_i, data_j] = -1
            # my_label[same_label_mask, data_i, data_j] = labels[same_label_mask,data_i,0] + 1


    if avg_type == 'simple':
        scores = torch.sum(my_score, dim=(1,2)) / ( data_num * (data_num-1) / 2 ) 
    elif avg_type == 'mean':
        
        # get between class
        total_labels = labels[:,:,0].clone()

        s_mean = torch.zeros(batch_size)

        score_in_class = []
        score_between_class = []
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = torch.zeros(batch_size)
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            # TODO 
            # if len(mask_i) == 1:
            #     s_mean = torch.zeros(batch_size)
            if mask_i.sum() == 0:
                s_mean = torch.zeros(batch_size)
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    i_len = mask_i.sum(dim=1).float()
                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                        a_i = a_i.sum(dim=-1) / ( i_len[:,0] )

                        score_in_class.append(a_i)
                    else:
                        b_i = score.sum(dim=-1) / j_len
                        b_i = b_i.sum(dim=-1) / j_len[:,0]
                        
                        score_between_class.append(b_i)

        in_score = 0
        for score_in in score_in_class:
            in_score += score_in
        in_score = in_score / len(score_in_class)
        between_score = 0
        for score_between in score_between_class:
            between_score += score_between
        between_score = between_score / len(score_between_class)

        scores = -(between_score - in_score)

    elif avg_type == 'sil':
        # get between class
        total_labels = labels[:,:,0].clone()

        s_mean = torch.zeros(batch_size)
        if use_cuda:
            s_mean = s_mean.cuda()
        
        for label_i in range(label_num):
            if label_num == 1:
                s_mean = torch.zeros(batch_size)
                if use_cuda:
                    s_mean = s_mean.cuda()
                
                continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            i_len = mask_i.sum(dim=1).float()
            
            # TODO 
            if mask_i.sum() == 0:
                s_mean = torch.zeros(batch_size)
                if use_cuda:
                    s_mean = s_mean.cuda()
            else:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                    else:
                        if b_i is None:
                            b_i = score.sum(dim=-1) / j_len
                        else:
                            b_i = torch.min( b_i, score.sum(dim=-1) / j_len )
                # print(b_i.mean(), a_i.mean())

                s_i = (b_i - a_i) / torch.max( a_i, b_i )
                # s_i = s_i.mean(dim=-1)
                s_i[ torch.isnan(s_i) ] = 0
                s_i = s_i.sum(dim=-1) / i_len[:,0]
                s_mean = torch.max( s_i, s_mean )


        # s_mean[i_len[:,0] == 1] = 0
        scores = s_mean

        scores = -scores


    elif avg_type == 'sil_list':
        # get between class
        total_labels = labels[:,:,0].clone()

        # s_mean = torch.zeros(batch_size)
        # if use_cuda:
        #     s_mean = s_mean.cuda()
        s_mean = []
        a_list = []
        b_list = []
        for label_i in range(label_num):
            # if label_num == 1:
            #     s_mean = torch.zeros(batch_size)
            #     if use_cuda:
            #         s_mean = s_mean.cuda()
            #     continue
            a_i = None
            b_i = None
            # batch_size x data_num
            mask_i = (total_labels == label_i)

            mask_i = mask_i.unsqueeze(-1).expand_as(my_score)
            
            i_len = mask_i.sum(dim=1).float()
            
            # TODO 
            # if mask_i.sum() == 0:
            #     s_mean = torch.zeros(batch_size)
            #     if use_cuda:
            #         s_mean = s_mean.cuda()
            # else:
            if True:
                for label_j in range(label_num):
                    # get the data of label_i to label_j
                    mask_j = (total_labels == label_j)
                    mask_j = mask_j.unsqueeze(1).expand_as(my_score)
                    
                    mask_ij = mask_i * mask_j

                    j_len = mask_j.sum(dim=-1).float()

                    score = torch.zeros_like(my_score)
                    score[mask_ij] = my_score[mask_ij]
                    
                    if label_i == label_j:
                        a_i = score.sum(dim=-1) / ( i_len - 1 )
                        a_i[ i_len==1 ] = 0
                    else:
                        if b_i is None:
                            b_i = score.sum(dim=-1) / j_len
                        else:
                            b_i = torch.min( b_i, score.sum(dim=-1) / j_len )
                # print(b_i.mean(), a_i.mean())

                s_i = (b_i - a_i) / torch.max( a_i, b_i )
                # s_i = s_i.mean(dim=-1)
                s_i[ torch.isnan(s_i) ] = 0
                s_i = s_i.sum(dim=-1) / i_len[:,0]
                # s_mean = torch.max( s_i, s_mean )
                s_mean.append( s_i )
                a_list.append(a_i)
                b_list.append(b_i)


        # s_mean[i_len[:,0] == 1] = 0
        return s_mean, a_list, b_list, my_score.cpu().numpy(), total_labels.cpu().numpy()


    else:
        print('====> Error in shape context distance')

    scores = scores.detach()
    if use_cuda:
        scores = scores.cuda()  

    return scores

