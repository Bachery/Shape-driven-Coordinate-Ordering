import numpy as np
import torch
import sys
import os
sys.path.append('../')
import tools
from tqdm import tqdm

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

    data = torch.from_numpy(data)

    data_num = data.shape[1]

    labels = np.loadtxt(data_file + 'label.txt').astype('float32')
    labels = torch.from_numpy(labels)
    labels = labels.view((num_samples, -1, 1)).expand_as( data )


    return data, labels

def load_result(net_file, dim_num, num_samples, vis_type):
    ids = np.loadtxt(net_file).astype('int')
    return ids

vis_type = 'star'
reward_type = 'sc_sil'
data_type = 'dis'

# note = '8d-8n-2c-SSS-label'
# note = '16d-32n-4c-SSM'
note = '16d-8n-2c-SSS-tmp'
# note = 'valid-16d-8n-2c-SSS-label'
# note = 'curve_alg_200_100_dis'

dim_num = 16
data_num = 16
num_samples = 10000
label_num = 2
standard = 0.1

draw_num = 3

# data_path = './data/rand/valid-%dd-%d-%d-dis-5.0-5/' % ( dim_num, data_num, num_samples )
data_path = './data/%s/valid-%dd-%dc-%dn-%d-%s-0.1/' % ( vis_type, dim_num, label_num, data_num, num_samples, data_type )
# data_path = './data/%s/valid-%dd-%dc-16n-%d-%s-0.1/' % ( vis_type, dim_num, label_num, num_samples, data_type )


# 网络训练结果的命名改变了些，现在类似这样的，表示 16到16维，8到16数量，2到4类的训练数据，使用距离中心center作为类别信息
# sc_sil-rn2-dis-True-True-[16-16]d-[8-16]n-[2-4]c-center-0.1-note-debug-2020-10-29-21-44

# net_name = '%s-rnn-dis-True-%dd-%dn-%dc-SSS-note-valid-8n' % (reward_type, dim_num, data_num, label_num)
net_name = '%s-rn2-dis-True-True-%dd-%dn-%dc-SSS-center-0.1-note-valid-%s' % (reward_type, dim_num, data_num, label_num, note)
# net_name = '%s-rn2-dis-True-True-%dd-%dn-%dc-SSS-label-0.1-note-%s' % (reward_type, dim_num, data_num, label_num, note)
#           sc_sil-rn2-dis-True-True-8d-8n-2c-SSS-center-0.1-note-8d-8n-2c-SSS
# net_name = 'sc_sil-dis-%dd-8n-2c' % dim_num

# net_path = './vis/%s/16/valid-16d-8n-2c-SMS/%s/render/0/net-ids.txt' % ( vis_type, net_name )
net_path = './vis_valid/%s/16/%s/render/0/net-ids.txt' % ( vis_type, net_name )
# net_path = './vis_valid/%s/curve_k_10000/%s/render/0/net-ids.txt' % ( vis_type, net_name )
# net_path = './img/%s/curve_alg_100_10/sc_sil-dis-16d-8n-2c/alg-ids.txt' % ( vis_type )
# net_path = './img/%s/curve_alg_200_100_dis/sc_sil-dis-16d-8n-2c/alg-ids.txt' % ( vis_type )
# net_path = './vis_valid/%s/curve_alg_100_10/sc_sil-dis-16d-8n-2c/alg-ids.txt' % ( vis_type )
# net_path = './vis/%s/8/valid-8d-8n-2c-SSS/%s/render/0/net-ids.txt' % ( vis_type, net_name )
# net_path = './vis/%s/8/valid-alg-8d-8n-2c/%s/render/0/net-ids.txt' % ( vis_type, net_name )
# net_path = './user_study_copy/data/net/100-10-sc_sil-dis-16d-16n-2c/alg-ids.txt'

datas, labels = load_data(data_path, dim_num, num_samples, vis_type)

ids = load_result(net_path, dim_num, num_samples, vis_type)

# convert to np
net_datas = datas.clone().cpu().numpy()
net_labels = labels.clone().cpu().numpy()


for batch_index in tqdm(range(draw_num)):
    for data_index in range(data_num):
        net_datas[batch_index, data_index] = datas[batch_index, data_index][ ids[batch_index] ]
        net_labels[batch_index, data_index] = labels[batch_index, data_index][ ids[batch_index] ]



net_datas = torch.from_numpy(net_datas)
net_labels = torch.from_numpy(net_labels)

datas = datas.cuda()
labels = labels.cuda()
net_datas = net_datas.cuda()
net_labels = net_labels.cuda()

init_scores = tools.evaluate(datas[:draw_num], labels[:draw_num], None, vis_type, 'sc_sil', False)
net_scores = tools.evaluate(net_datas[:draw_num], net_labels[:draw_num], None, vis_type, 'sc_sil', False)


datas = datas.cpu().numpy()
labels = labels.cpu().numpy()[:,:,0]

net_datas = net_datas.cpu().numpy()
net_labels = net_labels.cpu().numpy()[:,:,0]

# for batch_index in range(num_samples):
save_dir = os.path.join( './img', vis_type, '%d' % dim_num, 
    reward_type + '-' + data_type + '-' + 
    str(dim_num) + 'd-' + str(data_num) + 'n-' 
    + str(label_num) + 'c-note-' + note)
print(save_dir)
    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for batch_index in range(draw_num):

    data = datas[batch_index]
    net_data = net_datas[batch_index]
    
    label = labels[batch_index]
    net_label = net_labels[batch_index]
    
    init_order = [ i for i in range(dim_num) ]
    net_order = ids[batch_index]

    init_score  = init_scores[batch_index]
    net_score  = net_scores[batch_index]
    # init_sims, init_weis, init_sils  = init_scores
    # init_sim = init_sims[batch_index]
    # init_mer = init_weis[batch_index]
    # init_sil = init_sils[batch_index]

    # net_sims, net_weis, net_sils  = net_scores
    # net_sim = net_sims[batch_index]
    # net_mer = net_weis[batch_index]
    # net_sil = net_sils[batch_index]

    
    # net_score = net_scores[batch_index]

    dpi = 400

    data = np.column_stack((label, data))
    net_data = np.column_stack((net_label, net_data))

    if reward_type == 'sc_sil':
        tools.draw_star_mul(data, init_order, 
            # 'Class dist: %.3f\nSIL: %.3f' % (abs(init_mer), abs(init_sil)), 
            '%.3f' % ( abs(init_score)), 
            save_dir + '/%d-net-%.3f-o' % (batch_index, abs(init_score)), 
            label_num=label_num,
            with_category=True, dpi=dpi )
            # './img/%s-%d-%d-%d-o' % (vis_type, dim_num, data_num, batch_index), with_category=True )

    tools.draw_star_mul(net_data, net_order, 
        # 'Class dist: %.3f\nSIL: %.3f' % (abs(net_mer), abs(net_sil)), 
        '%.3f' % ( abs(net_score)), 
        # save_dir + '/%d-net-%s' % (batch_index, reward_type), 
        save_dir + '/%d-net-%.3f' % (batch_index, abs(net_score)), 
        label_num=label_num,
        with_category=True, dpi=dpi )