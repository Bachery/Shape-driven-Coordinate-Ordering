import torch
import numpy as np


def davies_bouldin_index(data, labels, reward_type, label_num=-1):
    """
    Evaluate the visulization result unber some order
    ----------
    Parameters
    ----------
        data: array of size (batch_size, data_num, dim_num)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates
        labels: labels of data (batch_size, data_num, dim_num)
            Defines the labels id of data
        label_num: int
    Retruns
    ----
        scores: array of evaluation scores (batch_size)
    """

    points = calc_radviz(data)
    dbi_2d = DBi(points, labels[:, :, 0], label_num)
    if reward_type == 'dbr':
        dbi_nd = DBi( data, labels[:, :, 0], label_num )
        return - dbi_nd / dbi_2d
    else:
        return dbi_2d
    return 


def calc_radviz(data):
    '''
    Paramters:
    ---
        data: tensor, (batch_size x data_num x dim_num)

    Returns:
    ---
        points: tensor, (batch_size x data_num x 2)
    '''
    
    # batch_size x data_num x dim_num
    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]

    dim_angles = torch.linspace( 0, np.pi*2, dim_num+1)[:-1]
    dim_x = torch.cos(dim_angles)
    dim_y = torch.sin(dim_angles)

    if data.is_cuda:
        dim_angles = dim_angles.cuda()
        dim_x = dim_x.cuda()
        dim_y = dim_y.cuda()        

    # normalization
    # data_min = data.min(dim=1).view(( batch_size, 1, -1 )).repeat( data_num, 1 )
    # data_max = data.max(dim=1).view(( batch_size, 1, -1 )).repeat( data_num, 1 )
    # data = (data-data_min) / (data_max - data_min)

    # calculate data points
    data_sum = data.sum(dim=-1)
    # avoid some case when sum == 0
    # data_sum[ abs(data_sum) < 1e-6 ] = 1
    
    point_x = (data * dim_x).sum(dim=-1) / data_sum
    point_y = (data * dim_y).sum(dim=-1) / data_sum

    point_x[ abs(data_sum) < 1e-6 ] = 0
    point_y[ abs(data_sum) < 1e-6 ] = 0

    # batch_size x data_num x 2
    points = torch.stack( (point_x, point_y), dim=-1 )
    
    return points


def DBi(points, labels, label_num):
    # batch_size x data_num x 2
    batch_size, data_num, _ = points.shape    

    # labels: batch_size x data_num
    
    # 计算每个类的中心
    class_center = []
    class_distance = []
    for label_i in range(label_num):
        no_label_i_mask = (labels != label_i)
        label_i_mask = (labels == label_i)

        class_i_points = points.clone()
        class_i_points[no_label_i_mask] = 0
        
        class_i_len = label_i_mask.sum(dim=1).float()

        sum_i_points = class_i_points.sum(dim=1)
        class_i_center = sum_i_points / class_i_len.unsqueeze(-1).expand_as(sum_i_points)
        
        class_center.append( class_i_center.clone() )

        class_i_center = class_i_center.unsqueeze(1).repeat(1, data_num, 1)
        class_i_center[no_label_i_mask] = 0
        # 类内距离
        S_i = torch.sqrt(  torch.sum(
            torch.pow( (class_i_points - class_i_center), 2 ), dim=(1,2)
             ) / class_i_len )

        class_distance.append( S_i.clone() )

    # 类间距离
    M = torch.zeros( (batch_size, label_num, label_num) )
    D = torch.zeros( (batch_size, label_num) )

    if points.is_cuda:
        M = M.cuda()
        D = D.cuda()

    for label_i in range(label_num):
        for label_j in range(label_num):
            if label_i == label_j:
                continue
            elif label_i > label_j:
                M[:, label_i, label_j] = M[:, label_j, label_i]
            # M[label_i, label_j]  = np.linalg.norm( class_center[label_i] - class_center[label_j], axis=-1 )
            else:
                M[:, label_i, label_j]  = torch.norm( class_center[label_i] - class_center[label_j], dim=-1 )

            R_ij = ( class_distance[label_i] + class_distance[label_j] ) / M[:, label_i, label_j]

            D[:,label_i] = torch.max( D[:, label_i], R_ij )
            
    # DB index
    DB = torch.mean(D, dim=-1)
    return DB



def silhouette_coefficient(data, labels, label_num):
    """
    Parameters
    ----
        points: Tensor of data (batch_size, data_num, dim_num)
    """
    use_cuda = data.device.type == 'cuda'

    points = calc_radviz(data)    

    batch_size = points.shape[0]
    data_num = points.shape[1]

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
            points_i = points[:,data_i]
            points_j = points[:,data_j]
            my_score[:, data_i, data_j] = torch.norm( points_i-points_j, dim=-1)

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
                # inner dis
                if label_i == label_j:
                    a_i = score.sum(dim=-1) / ( i_len - 1 )
                    a_i[ i_len==1 ] = 0
                # outer dis
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
    scores = scores.detach()
    if use_cuda:
        scores = scores.cuda()  

    return scores
