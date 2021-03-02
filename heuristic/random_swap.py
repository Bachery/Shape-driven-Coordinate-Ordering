import time
import torch
import numpy as np
import os
import sys
import copy
sys.path.append('../')
import tools
from tqdm import tqdm
import evaluate.shape_context as SC


def load_data(data_file, dim_num, num_samples, vis_type):

    data = np.loadtxt(data_file + 'data.txt').astype('float32')
    if vis_type == 'star':
        data = data.reshape((num_samples, -1, dim_num))

    else:
        data = data.reshape((num_samples, -1, dim_num))
        data_num = data.shape[1]
        # normalize
        max_data = np.max(data, axis=1).reshape(num_samples, 1, -1).repeat(data_num, 1)
        min_data = np.min(data, axis=1).reshape(num_samples, 1, -1).repeat(data_num, 1)
        data = (data - min_data) / (max_data - min_data)            

    # data = torch.from_numpy(data)
    data_num = data.shape[1]


    labels = np.loadtxt(data_file + 'label.txt').astype('float32')
    labels = labels.reshape((num_samples, -1, 1)).repeat( dim_num, -1 )

    return data, labels

def calc_score_tradition(data, labels):
    """
    data: float array, (data_num, dim_num)
    labels: float array, (data_num, dim_num)
    """
    data = np.reshape( data, (1, data_num, dim_num))
    labels = np.reshape( labels, (1, data_num, dim_num))
    
    sil = SC.shape_context_distance_local(data, labels, with_label, label_num, 'sil')
    # print('ho')
    return abs(sil)



os.environ["CUDA_VISIBLE_DEVICES"] = '2'

use_cuda = True

# vis_type = 'radviz'
# reward_type = 'rad_dbr'
# data_type = 'ran'

vis_type = 'star'
reward_type = 'sc_sil'
data_type = 'dis'

with_label = True

# 数据集的
num_samples = 1000
label_num = 6 #2
data_num = 32 #8
dim_num = 16  #8

# star_dim = 16
# end_dim = 16

# max_keep_time的范围
start_time = 5
end_time = 5
max_loop_num = 1000

evaluate_tool = tools.EvaluateTools(use_cuda)

# 结果保存在vis_valid/star/xxxxxx

# for dim_num in range(star_dim, end_dim+1):
for max_keep_time in range(start_time, end_time+1, 5):
    data_file = '../data/%s/valid-%dd-%dc-%dn-%d-%s-0.1/' % ( vis_type, dim_num, label_num, data_num, num_samples, data_type )
    # data_file = './data/%s/valid-8d-%dc-%dn-%d-%s-0.1/' % ( vis_type, label_num, data_num, num_samples, data_type )
    # data_file = './data/%s/valid-8d-%dc-32n-%d-%s-0.1/' % ( vis_type, label_num, num_samples, data_type )


    # max_keep_time = 10
    folder = 'curve_alg_%d_%d' % (max_loop_num, max_keep_time)
    print(folder)



    def calc_score(data, labels):
        """
        data: float array, (data_num, dim_num)
        labels: float array, (data_num, dim_num)
        """
        data = np.reshape( data, (1, data_num, dim_num))
        labels = np.reshape( labels, (1, data_num, dim_num))
        
        data = torch.from_numpy(data)
        labels = torch.from_numpy(labels)

        if use_cuda:
            data = data.cuda()
            labels = labels.cuda()
        
        sil = evaluate_tool.evaluate(data, labels, order, vis_type, reward_type, with_label, label_num)
        sil = sil.cpu().numpy()[0]
        return abs(sil)

    # save_dir = os.path.join( './vis_valid', vis_type,  folder, 
    save_dir = os.path.join( '../vis_valid', vis_type, 'random',  
        str(max_loop_num) + '-' + str(max_keep_time) + '-' +
        reward_type + '-' + data_type + '-' + 
        str(dim_num) + 'd-' + str(data_num) + 'n-' 
        + str(label_num) + 'c'  )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/vis_valid')


    # num_samples x data_num x dim_num
    total_data, total_labels = load_data( data_file, dim_num, num_samples, vis_type )

    total_orders = []
    total_scores = []
    total_origin_scores = []


    t1 = time.time()

    for batch_index in tqdm(range(num_samples)):
    # for batch_index in tqdm(range(2)):
        # if batch_index > 40:
        #     break
        # if batch_index not in [75]:
        # if batch_index not in [6, 56, 62, 77]:
            # continue
        # initial order
        order_index = [ i for i in range(dim_num) ]
        order = [ i for i in range(dim_num) ]
        order = np.array(order)

        best_order = None
        best_score = 0
        keep_time = 0
        
        data = total_data[batch_index]
        labels = total_labels[batch_index]

        origin_score = calc_score( data[:, order], labels[:, order] )
        # origin_score = calc_score_tradition( data[:, order], labels[:, order] )
        total_origin_scores.append( origin_score )

        loop_time = 0
        
        for _ in range(max_loop_num):
            loop_time += 1
            
            current_score = calc_score( data[:, order], labels[:, order] )
            # current_score = calc_score_tradition( data[:, order], labels[:, order] )
            if current_score > best_score:
                best_score = current_score
                best_order = copy.copy(order)
                keep_time = 0
            else:
                if keep_time >= max_keep_time:
                    break
                keep_time += 1
            # swap two index
            i,j = np.random.choice(order_index, 2, False)
            order = copy.copy(best_order)
            order[i], order[j] = order[j], order[i]
        
        # tools.draw_star_mul( data[:, best_order], best_order, labels=labels[:, 0].astype('int'), save_title='', save_name=save_dir + '/vis_valid/%d-net-%.3f-r' % (batch_index, best_score), label_num=label_num, dpi=400 )
        # tools.draw_star_mul( data, order_index, labels=labels[:, 0].astype('int'), save_title='', save_name=save_dir + '/vis_valid/%d-net-%.3f-0' % (batch_index, origin_score ), label_num=label_num, dpi=400 )
        # print('\n', loop_time, best_score)
        total_scores.append( best_score )
        total_orders.append( best_order )

    t2 = time.time()

    total_scores = np.array(total_scores)
    total_orders = np.array(total_orders)
    total_origin_scores = np.array(total_origin_scores)


    a = np.array(t2-t1).reshape(1,1)
    np.savetxt(save_dir + '/alg-time.txt', a)
    np.savetxt(save_dir + '/alg-score.txt', total_scores)
    np.savetxt(save_dir + '/alg-score-o.txt', total_origin_scores)
    np.savetxt(save_dir + '/alg-ids.txt', total_orders)
