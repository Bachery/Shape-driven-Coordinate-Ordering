import tqdm
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from evaluate import shape_context, pargnostic, DBi


# colors = [ 'black', 'hotpink', 'cornflowerblue', '#a0c8c0', '#dd9286', '#c0bed3', '#91aec6',  '#f5f4c2',   
#         'peachpuff', 'lightgreen', 'royalblue', 'silver', 'khaki', 'lime', 'coral',
#         'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender' ]

# range_color = 'hotpink'
colors = [ 'hotpink','cornflowerblue',  'salmon', 'yellowgreen', 'red', 'violet', #'#91aec6',  # '#dd9286',    #'#f5f4c2', '#a0c8c0', 
        'royalblue', 'silver', 'khaki', 'lime', 'coral', 'peachpuff', 
        'yellowgreen', 'lightblue',  'aqua', 'tan', 'violet', 'lavender' ]



class EvaluateTools(object):
    def __init__(self, use_cuda=True):
        bin_num = 50
        # use_cuda = True

        self.parg_c = pargnostic.NetPargnostics('cross', bin_num, 128, 1, 0.1, use_cuda)
        self.parg_e = pargnostic.NetPargnostics('par_en', bin_num, 128, 1, 0.1, use_cuda)
        self.parg_p = pargnostic.NetPargnostics('pccs', bin_num, 128, 1, 0.1, use_cuda)
        self.parg_m = pargnostic.NetPargnostics('mut', bin_num, 128, 1, 0.1, use_cuda)
        # self.parg_a = pargnostic.NetPargnostics('angle', bin_num, 128, 1, 0.1, use_cuda)
        # self.parg_p = pargnostic.NetPargnostics('parallelism', bin_num, 128, 1, 0.1, use_cuda)
        self.nsc = shape_context.NetShapeContext(128, 1, 0.1, use_cuda)
    
        
    def evaluate(self, data, labels, order, vis_type, reward_type, with_label, label_num=-1):
        """
        Evaluate the visulization result unber some order
        ----------
        Parameters
        ----------
            data: array of size (batch_size, data_num, dim_num)
                Defines the elements to consider as static. For the TSP, this could be
                things like the (x, y) coordinates
            order: order of dimensions (batch_size, dim_num)
                Defines the dimension order of data
            vis_type: string
            reward_type: string
        Retruns
        ----
            scores: array of evaluation scores (batch_size)
        """
        if reward_type == 'sim':
            if vis_type == 'parallel':
                return similarity(data, False)
            elif vis_type == 'star' or vis_type == 'radviz':
                return similarity(data, True)
            else:
                print("====>  Error in Evaluate")
        elif reward_type == 'sc_sim':
            return shape_context.shape_context_distance(data, labels, with_label, avg_type='simple', NSC=self.nsc, label_num=label_num)
        elif reward_type == 'sc_mer':
            return shape_context.shape_context_distance(data, labels, with_label, avg_type='mean', NSC=self.nsc, label_num=label_num)
        elif reward_type == 'sc_sil':
            return shape_context.shape_context_distance(data, labels, with_label, avg_type='sil', NSC=self.nsc, label_num=label_num)
        elif reward_type == 'sc_sil_list':
            return shape_context.shape_context_distance_tradition(data, labels, with_label, avg_type='sil_list', NSC=self.nsc, label_num=label_num)
        elif reward_type == 'sc_slr':
            return shape_context.silhouette_rate(data, labels, with_label, avg_type='sil', NSC=self.nsc, label_num=label_num)
        elif reward_type == 'sc_all':
            sim = shape_context.shape_context_distance(data, labels, with_label, avg_type='simple', NSC=self.nsc, label_num=label_num)
            mer = shape_context.shape_context_distance(data, labels, with_label, avg_type='mean', NSC=self.nsc, label_num=label_num)
            sil = shape_context.shape_context_distance(data, labels, with_label, avg_type='sil', NSC=self.nsc, label_num=label_num)
            return [ sim, mer, sil ]
        elif reward_type == 'scd':
            return -1 * shape_context.shape_context_distance(data, labels, with_label, avg_type='simple', NSC=self.nsc, label_num=label_num)
        elif reward_type == 'parg_cro':
            return pargnostic.pargnostics(data, labels, with_label, reward_type='cross', NPG=self.parg_c, label_num=label_num)
        elif reward_type == 'parg_enp':
            return pargnostic.pargnostics(data, labels, with_label, reward_type='par_en', NPG=self.parg_e, label_num=label_num)
        elif reward_type == 'parg_pcc':
            return pargnostic.pargnostics(data, labels, with_label, reward_type='pccs', NPG=self.parg_p, label_num=label_num)
        elif reward_type == 'parg_mut':
            return pargnostic.pargnostics(data, labels, with_label, reward_type='mut', NPG=self.parg_p, label_num=label_num)
        elif reward_type == 'rad_dbi':
            return DBi.davies_bouldin_index(data, labels, 'dbi', label_num)
        elif reward_type == 'rad_dbr':
            return DBi.davies_bouldin_index(data, labels, 'dbr', label_num)
        elif reward_type == 'rad_sil':
            return DBi.silhouette_coefficient(data, labels, label_num)
        else:
            print("====>  Error in Evaluate")

# NOTE evaluate

def evaluate(data, labels, order, vis_type, reward_type, with_label, nsc=None, label_num=-1):
    """
    Evaluate the visulization result unber some order
    ----------
    Parameters
    ----------
        data: array of size (batch_size, data_num, dim_num)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates
        order: order of dimensions (batch_size, dim_num)
            Defines the dimension order of data
        vis_type: string
        reward_type: string
    Retruns
    ----
        scores: array of evaluation scores (batch_size)
    """
    if reward_type == 'sim':
        if vis_type == 'parallel':
            return similarity(data, False)
        elif vis_type == 'star' or vis_type == 'radviz':
            return similarity(data, True)
        else:
            print("====>  Error in Evaluate")
    elif reward_type == 'sc_sim':
        return shape_context.shape_context_distance(data, labels, with_label, avg_type='simple', NSC=nsc, label_num=label_num)
    elif reward_type == 'sc_mer':
        return shape_context.shape_context_distance(data, labels, with_label, avg_type='mean', NSC=nsc, label_num=label_num)
    elif reward_type == 'sc_sil':
        return shape_context.shape_context_distance(data, labels, with_label, avg_type='sil', NSC=nsc, label_num=label_num)
    elif reward_type == 'sc_slr':
        return shape_context.silhouette_rate(data, labels, with_label, avg_type='sil', NSC=nsc, label_num=label_num)
    elif reward_type == 'sc_all':
        sim = shape_context.shape_context_distance(data, labels, with_label, avg_type='simple', NSC=nsc, label_num=label_num)
        mer = shape_context.shape_context_distance(data, labels, with_label, avg_type='mean', NSC=nsc, label_num=label_num)
        sil = shape_context.shape_context_distance(data, labels, with_label, avg_type='sil', NSC=nsc, label_num=label_num)
        return [ sim, mer, sil ]
    elif reward_type == 'sc':
        return shape_context.shape_context_distance(data, labels, with_label, NSC=nsc, label_num=label_num)
    elif reward_type == 'scd':
        return -1 * shape_context.shape_context_distance(data, labels, with_label, NSC=nsc, label_num=label_num)
    elif reward_type == 'parg_cro':
        return pargnostic.pargnostics(data, labels, with_label, reward_type='cross', label_num=label_num)
    elif reward_type == 'parg_enp':
        return pargnostic.pargnostics(data, labels, with_label, reward_type='par_en', label_num=label_num)
    elif reward_type == 'parg_pcc':
        return pargnostic.pargnostics(data, labels, with_label, reward_type='pccs', label_num=label_num)
    elif reward_type == 'parg_mut':
        return pargnostic.pargnostics(data, labels, with_label, reward_type='mut', label_num=label_num)
    # elif reward_type == 'parg_all':
    #     return pargnostic.pargnostics(data, labels, with_label, reward_type='all', label_num=label_num)
    elif reward_type == 'rad_dbi':
        return DBi.davies_bouldin_index(data, labels, 'dbi', label_num)
    elif reward_type == 'rad_dbr':
        return DBi.davies_bouldin_index(data, labels, 'dbr', label_num)
    elif reward_type == 'rad_sil':
        return DBi.silhouette_coefficient(data, labels, label_num)
    else:
        print("====>  Error in Evaluate")

def similarity(data, circle=False):
    """
    Parameters
    ----
        data: Tensor of data (batch_size, data_num, dim_num)
    """
    batch_size = data.shape[0]
    data_num = data.shape[1]
    dim_num = data.shape[2]
    
    # 全体tensor
    max_data = data.max(dim=1)[0].view(batch_size, 1, dim_num).repeat(1, data_num, 1)
    min_data = data.min(dim=1)[0].view(batch_size, 1, dim_num).repeat(1, data_num, 1)
    
    # normalize
    data = ( data - min_data ) / ( max_data - min_data )
    
    # calculate neighbor similarity
    if circle == False:
        loop_num = dim_num - 1
    else:
        loop_num = dim_num

    scores = 0
    for i in range(loop_num):
        scores += torch.norm(  data[:,:, i % dim_num ] - data[:,:, (i+1) % dim_num ], dim=1 )

    return scores

# NOTE draw

def draw( data, order, vis_type, reward_type, save_title, save_name, with_category=False, labels=None, dpi=50, label_num=-1):
    """
    Parameters
    ----
        data: Array of data (data_num, dim_num)
    """
    if order is None:
        if with_category:
            order = [ i for i in range(len(data[0,:]) - 1) ]
        else:
            order = [ i for i in range(len(data[0,:])) ]

    if vis_type == 'parallel':
        draw_parallel(data, order, save_title, save_name, with_category, dpi=dpi, label_num=label_num)
    elif vis_type == 'star':
        if data.shape[0] > 2:
            draw_star_mul(data, order, save_title, save_name, with_category, labels, dpi=dpi, label_num=label_num)
        else:
            draw_star(data, order, save_title, save_name, with_category, labels, dpi=dpi, label_num=label_num)
    elif vis_type == 'radviz':
        draw_radviz(data, order, save_title, save_name, with_category, dpi=dpi, label_num=label_num)
    else:
        print("====>  Error in Draw")

def draw_parallel(data, order, save_title, save_name, with_category=False, dpi=50, label_num=-1):
    """
    Parameters
    ----
        data: Array of data (data_num, dim_num)
        order: Array of dim order (dim_num)
    """
    if with_category:
        df = data
    else:
        class_name = np.zeros_like(data[:,0])
        df = np.column_stack((class_name, data))

    df = pd.DataFrame(df)

    columns = {}
    for index, o in enumerate(order):
        columns[index+1] = o+1
    
    df = df.rename(columns=columns)

    plt.close('all')
    plt.figure()
    
    # pd.plotting.parallel_coordinates(df, 0, lw=0.2, color=colors, alpha=0.25)
    pd.plotting.parallel_coordinates(df, 0, lw=0.5, color=colors, alpha=0.75)
            # color=('#556270', '#4ECDC4', '#C7F464'))
    plt.gca().get_legend().remove()
    plt.title(save_title, fontsize=15)
    plt.savefig(save_name + '.png', bbox_inches='tight', dpi=dpi)
    plt.cla()

def draw_star_mul(data, order, save_title, save_name, with_category=False, labels=None, dpi=50, label_num=-1):
    """
    Parameters
    ----
        data: Array of data (data_num, dim_num)
        order: Array of dim order (dim_num)
    """
    colors = [ 'black', 'hotpink', 'cornflowerblue',  '#a0c8c0',  '#dd9286', 'cornflowerblue',  'salmon', 'yellowgreen', 'red', 'violet', '#c0bed3', '#91aec6',  '#f5f4c2',   
        'peachpuff', 'lightgreen', 'royalblue', 'silver', 'khaki', 'lime', 'coral',
        'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender' ]

    if with_category:
        labels = data[:,0].astype('int')
        data = data[:,1:]
    
    data_num = data.shape[0]
    dim_num = data.shape[1]

    plt.close('all')
    plt.figure()

    if label_num == -1:
        label_num = 2
        row = 4
        col = 2
    elif label_num == 8:
        row = 8
        col = 4
    else:
        row = int(data_num / label_num)
        col = label_num

    max_row = 8
    if row > max_row:
        row = max_row
        col = int(data_num / row)

    plt.figure(figsize=(row, col))

    dim = []
    for o in order:
        dim.append( '' )
        # dim.append( str(o) )

    real_labels = np.zeros_like(labels)
    label_index = 0
    # label_step = int(data_num / label_num)
    label_id = 0

    while label_id < label_num:
        tmp = np.where(labels == label_id)[0][:int(data_num / label_num)]
        label_step = len(tmp)
        real_labels[ label_index:label_index+label_step ] = tmp
        label_index += label_step
        label_id += 1
        
    lw = 1.5
    
    for i in range(col):
        for j in range(row):
            data_index = row*i+j
            data_index = real_labels[data_index]
            if data_index >= data_num:
                break
            
            ax = plt.subplot( col, row, row*i+j+1, projection = 'polar')      #构建图例
            ax.set_theta_zero_location('N')         #设置极轴方向

            theta = np.linspace(0, 2*np.pi, len(dim), endpoint=False)    #将圆根据标签的个数等比分
            ax.set_thetagrids(theta*180/np.pi, dim)         #替换标签
            theta = np.concatenate((theta,[theta[0]]))  #闭合

            plt.gca().axes.spines['polar'].set_visible(False)
            plt.gca().axes.get_yaxis().set_visible(False)
            ax.grid(True, linestyle='--', color='k', linewidth=0.5, alpha=0.5)

            ylabels = [ '' ]
            ax.set_yticklabels(ylabels)

            value = np.concatenate((data[data_index],[data[data_index][0]]))  #闭合
            if labels is None:
                ax.plot(theta,value,'m-',lw=lw,alpha =1, c=colors[0])    #绘图
            else:
                ax.plot(theta,value,'m-',lw=lw,alpha =1, c=colors[ labels[data_index] + 1 ])    #绘图

    plt.suptitle(save_title, fontsize=15, y=-0.01)
    plt.savefig(save_name + '.pdf', bbox_inches='tight', dpi=dpi)
    plt.savefig(save_name + '.png', bbox_inches='tight', dpi=dpi)
    # plt.savefig(save_name + '.svg', bbox_inches='tight', format='svg')
    plt.cla()

def draw_star(data, order, save_title, save_name, with_category=False, labels=None, dpi=50, label_num=-1, c=None):
    """
    Parameters
    ----
        data: Array of data (data_num, dim_num)
        order: Array of dim order (dim_num)
    """
    colors = [ 'black', 'hotpink', 'cornflowerblue', '#a0c8c0', '#dd9286', '#c0bed3', '#91aec6',  '#f5f4c2',   
            'peachpuff', 'lightgreen', 'royalblue', 'silver', 'khaki', 'lime', 'coral',
            'yellowgreen', 'lightblue', 'salmon', 'aqua', 'tan', 'violet', 'lavender' ]

    if with_category:
        labels = data[:,0].astype('int')
        data = data[:,1:]

    data_num = data.shape[0]
    dim_num = data.shape[1]

    plt.close('all')
    plt.figure()
    ax = plt.subplot(111,projection = 'polar')      #构建图例
    plt.gca().axes.spines['polar'].set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)

    ax.set_theta_zero_location('N')         #设置极轴方向
    
    dim = []
    for o in order:
        dim.append( str(o) )

    theta = np.linspace(0, 2*np.pi, len(dim), endpoint=False)    #将圆根据标签的个数等比分
    theta = np.concatenate((theta,[theta[0]]))  #闭合
    ax.set_thetagrids(theta*180/np.pi, dim, fontsize=30)         #替换标签

    # ylabels = [ '' ]
    # ax.set_yticklabels(ylabels)

    for i, d in enumerate(data):
        value = np.concatenate((d,[d[0]]))  #闭合
        if c is None:
            ax.plot(theta,value,'m-',lw=2, alpha = 1, c=colors[i] )    #绘图
        else:
            ax.plot(theta,value,'m-',lw=2, alpha = 1, c=c )    #绘图

    ax.grid(True, linestyle='--', color='k', linewidth=0.75, alpha=0.75)
    # plt.gca().get_legend().remove()
    # plt.title(save_title, fontsize=15)
    plt.savefig(save_name + '.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(save_name + '.svg', bbox_inches='tight', format='svg')

    plt.cla()

def draw_radviz(data, order, save_title, save_name, with_category=False, dpi=50, label_num=-1):
    """
    Parameters
    ----
        data: Array of data (data_num, dim_num)
        order: Array of dim order (dim_num)
    """

    colors = [ 'hotpink','cornflowerblue',  'salmon', 'yellowgreen', 'red', 'violet', #'#91aec6',  # '#dd9286',    #'#f5f4c2', '#a0c8c0', 
            'royalblue', 'silver', 'khaki', 'lime', 'coral', 'peachpuff', 
            'yellowgreen', 'lightblue',  'aqua', 'tan', 'violet', 'lavender' ]

    if with_category:
        df = data
    else:
        class_name = np.zeros_like(data[:,0])
        df = np.column_stack((class_name, data))

    df = pd.DataFrame(df)
    columns = {}
    for index, o in enumerate(order):
        columns[index+1] = o+1
    # 替换维度的标签
    df = df.rename(columns=columns)

    plt.close('all')
    plt.figure()
    
    # ax = plt.gca(xlim=[-0.5, 0.8], ylim=[-0.1, 0.5])
    # ax.tick_params(size=30)
    plt.rc('font', size=20)
    rad = pd.plotting.radviz(df, 0, color=colors, s=6)#, marker='s')
    # circle = plt.Circle( (0,0), 1, color='black', linewidth=2, fill=False )
    # plt.gca().add_patch(circle)
    xy = np.zeros((len(order), 2))
    theta = np.linspace(0, 2*np.pi, len(order), endpoint=False)
    for index, ang in enumerate(theta):
        xy[index][0] = np.cos(ang) * 1.0
        xy[index][1] = np.sin(ang) * 1.0
    polygon = plt.Polygon( xy, color='black', fill=False, lw=2)
    plt.gca().add_patch(polygon)

    plt.gca().get_legend().remove()
    plt.axis('off')
    plt.axis('equal')
    
    plt.title(save_title, fontsize=10)
    plt.savefig(save_name + '.png', bbox_inches='tight', dpi=dpi)
    plt.savefig(save_name + '.pdf', bbox_inches='tight', dpi=dpi)
    plt.cla()

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

# NOTE load data

def read_data(filename, category_dim=0):
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
    title = []
    category = []
    category_set = []
    # read data
    with open(filename, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            if len(title) == 0:
                line = line.strip('\n').split(',')
                for l in line:
                    title.append(l)
            else:
                line = line.strip('\n').split(',')
                if line[category_dim] not in category_set:
                    category_set.append(line[category_dim])
                
                category.append( category_set.index(line[category_dim]) )

                data_point = []
                for i in range(len(line)):
                    if i != category_dim:
                        data_point.append( float(line[i]) )
                data.append(data_point)
                
    data = np.array(data)
    
    return np.column_stack((category, data)), category_set
