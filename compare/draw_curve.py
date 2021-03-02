
import numpy as np
import networkx as nx
import prettytable as pt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib.gridspec as gs
from matplotlib.patches import Polygon 

range_color = 'hotpink'
colors = [ 'cornflowerblue',  '#a0c8c0',  '#dd9286',  'peachpuff', #'#91aec6',   #'#f5f4c2', 
        'salmon','royalblue', 'silver', 'khaki', 'lime', 'coral', 'peachpuff', 
        'yellowgreen', 'lightblue',  'aqua', 'tan', 'violet', 'lavender' ]


def read_data(file_path, is_origin=False, is_alg=False):
    if is_alg:
        if is_origin:
            score = np.loadtxt(file_path + '/alg-score-o.txt')
        else:
            score = np.loadtxt(file_path + '/alg-score.txt')
    else:
        if is_origin:
            score = np.loadtxt(file_path + '/net-score-o.txt')
        else:
            score = np.loadtxt(file_path + '/net-score.txt')
    score = abs(score)
    return score

# dim_num = 8
# data_num = 2
# valid_size = 9

vis_type = 'star'
# vis_type = 'radviz'

star_dim = 16
end_dim = 32

# star_dim = 2
# end_dim = 8

def load_result(net_type, dim_num, is_origin, label_type):
    
    # if net_type == '16d-32n-2c-SSS-label':
    #     # net_type = '16d-32n-2c-SSS'
    #     valid_path = '../vis_valid/star/curve_l_10000/sc_sil-rn2-dis-True-True-16d-32n-%dc-SSS-label-0.1-note-%s/render/0' % (dim_num, net_type)
    # else:
    #     valid_path = '../vis_valid/star/curve_l_10000/sc_sil-rn2-dis-True-True-16d-32n-%dc-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)

    if net_type == 'alg':
        # valid_path = '../img/star/curve_alg_100_10/sc_sil-dis-8d-%dn-2c' % dim_num
        # valid_path = '../img/star/curve_alg_100_10/sc_sil-dis-8d-32n-%dc' % dim_num
        # valid_path = '../img/star_8/curve_alg_100_20/sc_sil-dis-%dd-8n-2c' % dim_num
        valid_path = '../vis_valid/star/curve_alg_100_10/sc_sil-dis-%dd-8n-2c' % dim_num
        is_alg = True
    elif net_type == 'max':
        valid_path = '../vis_valid/star/curve_alg_1000_1000_dis_max/sc_sil-dis-%dd-8n-2c' % dim_num
        is_alg = True
    else:
        # valid_path = '../vis_valid/star/curve_m/sc_sil-rn2-dis-True-True-8d-%dn-2c-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)
        # valid_path = '../vis_valid/star/curve_k_10000/sc_sil-rn2-dis-True-True-8d-32n-%dc-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)
        # valid_path = '../vis_valid/star/curve_n_1000/sc_sil-rn2-dis-True-True-%dd-8n-2c-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)

        if "Label" in label_type:
            valid_path = '../vis_valid/star/curve_d_10000/sc_sil-rn2-dis-True-True-%dd-8n-2c-SSS-label-0.1-note-%s/render/0' % (dim_num, net_type)
        else:
            valid_path = '../vis_valid/star/curve_d_10000/sc_sil-rn2-dis-True-True-%dd-8n-2c-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)
            # valid_path = '../vis_valid/star/curve_std_10000/sc_sil-rn2-dis-True-True-%dd-8n-2c-SSS-center-0.2-note-%s/render/0' % (dim_num, net_type)

        # valid_path = '../vis_valid/star/curve_k_10000/sc_sil-rn2-dis-True-True-16d-32n-%dc-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)
        # valid_path = '../vis_valid/star/curve_m_10000/sc_sil-rn2-dis-True-True-16d-%dn-2c-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)
        # valid_path = '../vis_valid/star/curve_n_10000/sc_sil-rn2-dis-True-True-%dd-8n-2c-SSS-center-0.1-note-%s/render/0' % (dim_num, net_type)
        
        is_alg = False

    score = read_data(valid_path, is_origin, is_alg)
    return np.mean(score), score



lw = 2

net_types = [ '[16-16]d-[8-8]n-[2-2]c', '[24-24]d-[8-8]n-[2-2]c', '[16-24]d-[8-8]n-[2-2]c', '[16-16]d-[8-8]n-[2-2]c' ]
label_types = [ 'Trained on n=16', 'Trained on n=24', 'Trained on n=16~24', 'Input order' ]

net_types = [ '[16-16]d-[8-8]n-[2-2]c', '[16-16]d-[16-16]n-[2-2]c', '[16-16]d-[8-16]n-[2-2]c', '[16-16]d-[8-8]n-[2-2]c' ]
label_types = [ 'Trained on m=8', 'Trained on m=16', 'Trained on m=8~16', 'Input order' ]

net_types = [ '[16-16]d-[32-32]n-[2-2]c', '[16-16]d-[32-32]n-[4-4]c', '[16-16]d-[32-32]n-[2-4]c', '[16-16]d-[32-32]n-[2-2]c' ]
label_types = [ 'Trained on k=2', 'Trained on k=4', 'Trained on k=2~4', 'Input order' ]


net_types = [ '[16-16]d-[8-8]n-[2-2]c', 'alg', 'max', '16d-100n-4c-SSS-dbr' ]
label_types = [ 'network', 'random swapping', 'random swapping', 'input order' ]


table = pt.PrettyTable()
table.field_names = [ 'type', 'mean', 'std', ]


alg_score = None
max_score = None
net_score = None

plt.figure(figsize=(8,4))

for i in range(len(net_types)):
    # 这里 == 什么 取决于label_types的数量，如果你绘制3个曲线，那么第4个曲线就会绘制 input_order 的数值，就 i == 3
    if i == 2:
        is_origin = True
        color = '#c0bed3'
    else:
        is_origin = False
        color = colors[i]
    scores = []
    net_type = net_types[i]
    for dim_num in range(star_dim, end_dim+1):
        mean_score, score = load_result(net_type, dim_num, is_origin, label_types[i])
        
        if net_type == 'alg':
            alg_score = np.array(score)
        elif net_type == 'max':
            max_score = np.array(score)
        elif label_types[i] == 'network':
            net_score = np.array(score)


        table.add_row( [net_type, '%.3f' % mean_score, '%.3f' % np.std(score)] )
        
        scores.append( mean_score )

    plt.plot( range(end_dim-star_dim+1),  scores, c=color, lw=lw, label= label_types[i] )

fontsize = 15

plt.legend()
# plt.legend(loc='right', bbox_to_anchor=(1, 0.4))

# xticks = [ i-star_dim for i in range(star_dim, end_dim+1, 2) ]
# xlabels = [ str(i) for i in range(star_dim, end_dim+1, 2) ]
# plt.axvline(x=16-star_dim, alpha=0.75, lw=1, linestyle='--', c='k')
 
xticks = [ i-star_dim for i in range(star_dim, end_dim+1, 1) ]
xlabels = [ str(i) for i in range(star_dim, end_dim+1, 1) ]
plt.axvline(x=24-star_dim, alpha=0.75, lw=1, linestyle='--', c='k')

plt.xticks(xticks, xlabels)

plt.xlabel('n', fontsize=fontsize)
plt.ylabel('Silhouette coefficient', fontsize=fontsize)
# label_types = [ 'ratio of DB index', 'silhouette coefficient', 'input order' ]

plt.savefig(os.path.abspath('./') + '/curve/star_n_curve.png',bbox_inches='tight', dpi=400)
plt.savefig(os.path.abspath('./') + '/curve/star_n_curve.pdf',bbox_inches='tight', dpi=400)
# plt.savefig(os.path.abspath('./') + '/curve/star_n_curve_16n.png',bbox_inches='tight', dpi=400)
# plt.savefig(os.path.abspath('./') + '/curve/star_n_curve_16n.pdf',bbox_inches='tight', dpi=400)
