import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import matplotlib.colors as col
import matplotlib.cm as cm
import matplotlib.gridspec as gs
from matplotlib.patches import Polygon 


range_color = 'hotpink'
colors = [ 'cornflowerblue', '#a0c8c0', '#dd9286', '#c0bed3', '#91aec6',  '#f5f4c2',   
        'peachpuff', 'lightgreen', 'royalblue', 'silver', 'khaki', 'lime', 'coral',
        'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender' ]

def read_data(file_path):
    score = np.loadtxt(file_path + '/net-loss.txt')
    gt = np.loadtxt(file_path + '/net-gt.txt')
    return score, gt

vis_type = 'star'

star_dim = 16
end_dim = 32


def load_result(net_type, dim_num):
    valid_path = '../shape/curve/%dd-%s-S-note-valid-curve/render/0' % (dim_num, net_type)
    score, gt = read_data(valid_path)
    return np.mean(score), np.mean(gt)

plt.figure(figsize=(8,3))

lw = 2

net_types = [ 'fcn2', 'fc' ]
# label_types = [ 'points data', 'dims data' ]
label_types = [ 'SAMPLE', 'ORIG' ]


for i in range(len(net_types)):
    scores = []
    gts = []
    net_type = net_types[i]
    for dim_num in range(star_dim, end_dim+1):
        score, gt = load_result(net_type, dim_num)
        scores.append( score )
        gts.append( gt )

    color = colors[i]
    plt.plot( range(end_dim-star_dim+1),  scores, c=color, lw=lw, label= label_types[i] )
    
    # if i == 0:
    #     plt.plot( range(end_dim-star_dim+1),  gts, c=range_color, lw=lw, label="GT" )
    # for dim_num in range(star_dim, end_dim):
    #     if dim_num >=8 and dim_num <= 16:
    #         dim_num = dim_num - star_dim
    #         plt.plot( [dim_num, dim_num+1],  [  scores[dim_num], scores[dim_num+1]  ], c=range_color, lw=lw)

plt.axvline(x=24-star_dim, alpha=0.75, lw=1, linestyle='--', c='k')

# plt.legend()
plt.legend(loc='upper left')


fontsize = 15

xticks = [ i-star_dim for i in range(star_dim, end_dim+1, 1) ]
xlabels = [ str(i) for i in range(star_dim, end_dim+1, 1) ]
plt.xticks(xticks, xlabels)

plt.xlabel('n', fontsize=fontsize)
plt.ylabel('MSE', fontsize=fontsize)

plt.savefig(os.path.abspath('./') + '/curve/curve_16n.png', bbox_inches='tight' , dpi=400)
plt.savefig(os.path.abspath('./') + '/curve/curve_16n.pdf', bbox_inches='tight' , dpi=400)
