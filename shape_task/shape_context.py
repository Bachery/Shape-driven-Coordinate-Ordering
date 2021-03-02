import cv2
import numpy as np
import math
import time
from collections import Counter

def radial_edges(r1, r2, n):
    #return a list of radial edges from an inner (r1) to an outer (r2) radius
    # re = [0]
    # re.extend( np.logspace(np.log10(r1), np.log10(r2), n))
    re = np.logspace(np.log10(r1), np.log10(r2), n)
    # re = [ r1* ( (r2/r1)**(k /(n - 1.) ) ) for k in range(0, n)]
    return re


def euclid_distance(p1,p2):
    return math.sqrt( ( p2[0] - p1[0] ) ** 2 + ( p2[1] - p1[1] ) ** 2 )


def get_angle(p1,p2):
    #compute the angle between points.
    return math.atan2((p2[1] - p1[1]),(p2[0] - p1[0]))

def normalize(data, data_num):
    return data / data_num
    # data_min = np.min(data)
    # data_max = np.max(data)
    # return (data - data_min) / (data_max - data_min)

class ShapeContext(object):
    """
    Given a point in the image, for all other points that are within a given
    radius, computes the relative angles.
    Radii and angles are stored in a  "shape matrix" with dimensions: radial_bins x angle_bins.
    Each element (i,j) of the matrix contains a counter/integer that corresponds to,
    for a given point, the number of points that fall at that i radius bin and at
    angle bin j. 
    """

    # def __init__(self,nbins_r=6,nbins_theta=12,r_inner=0.1250,r_outer=2.5,wlog=False):
    def __init__(self,nbins_r=5,nbins_theta=12,r_inner=0.1250,r_outer=2,wlog=False):

        self.nbins_r        = nbins_r             # number of bins in a radial direction
        self.nbins_theta    = nbins_theta         # number of bins in an angular direction
        self.r_inner        = r_inner             # inner radius
        self.r_outer        = r_outer             # outer radius
        self.nbins          = nbins_theta*nbins_r # total number of bins
        self.wlog           = wlog                # using log10(r) or Normalize with the median
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
            # points = np.repeat([_x], x_len, axis=0)
            # residual = x - points
            residual = x - _x
            result[i] = np.arctan2(residual[:,1], residual[:,0])

        # import IPython
        # IPython.embed()
        # result[-1.0]
        return result

    def compute(self,points):

        t1 = time.time()
        
        # distance matrix
        dist_array = self.distM(points)

        t2 = time.time()
        # print('distM: ', t2 - t1)

        # Normalize the distance matrix by the median distance or use log10
        if self.wlog:
            dist_array = np.log10(dist_array+1)
        else:
            median_dist = np.median(dist_array)
            dist_array = dist_array / median_dist

        ang_array = self.angleM(points)
        # let ang 0 ~ 2pi
        ang_array = ang_array + 2*np.pi * (ang_array < 0)

        t3 = time.time()
        # print('angleM: ', t3 - t2)
        
        # static place
        points_len = len(points)
        radial_index = np.zeros((points_len, points_len))
        angle_index = np.zeros((points_len, points_len))
        for rad in self.radial_range:
            radial_index += (dist_array >= rad) # from 0 ~ nbins_r-1
        radial_bin = dist_array >= self.radial_range[-1]

        for ang in self.angle_range:
            angle_index += (ang_array >= ang) # from 1 ~ nbin_t

        t4 = time.time()
        # print('Static: ', t4-t3)

        # BH_index = (radial_index * self.nbins_theta + (angle_index)).astype('int')
        # for i in range(points_len):
        #     # radial_index.diagonal() + angle_index.diagonal()
        #     pass
        # self.BH_index = BH_index

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

            
            # for j in range(points_len):
            #     if i == j or radial_bin[i][j] == True:
            #         continue
            #     rad = radial_index[i,j]
            #     ang = angle_index[i,j]
            #     index = int(rad * self.nbins_theta + ang)
            #     index = BH_index[i, j]
            #     BH[i, index] += 1

        # import IPython
        # IPython.embed()
        # normalize        
        for i in range(points_len):
            sm = normalize(BH[i], points_len-1)
            BH[i] = sm
        
        t5 = time.time()
        # print('HIS: ', t5-t4)
        # print()
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

    for i in range(len(data)):
        d = np.array(data[i])
        
        position = [ np.multiply(d[j], cos_sin[j]) for j in range(sourceL)  ]
        position.append(position[0])
        positions.append(position)
        points = []

        mean_num = math.floor( sample_num / sourceL )
        
        for i in range(sourceL):
            points.extend(np.linspace(position[i], position[i+1], sample_num, endpoint=False))

        points = np.array(points)
        shapes.append(points)
    
    shapes = np.reshape( shapes, (-1, 2) )
    
    return shapes




def get_best(data, sample_num=5):
    from itertools import permutations
    from tqdm import tqdm
    import copy
    order = [ i for i in range(len(data[0]))]
    min_order = None
    max_order = None
    min_score = float('inf')
    max_score = float('-inf')

    sc = ShapeContext()
    for p in permutations(order):
        my_data = copy.copy(data)
        my_data = np.array(my_data)
        my_data[0] = my_data[0][list(p)]
        my_data[1] = my_data[1][list(p)]
        shapes = get_shape(my_data, sample_num)
        b1 = sc.compute(shapes[0])
        b2 = sc.compute(shapes[1])

        score = sc.cost(b1,b2)
        if score > max_score:
            max_score = score
            max_order = copy.copy(p)
        if score < min_score:
            min_score = score
            min_order = copy.copy(p)

    print("Min score is : %.3f, order: %s" % (min_score, min_order))
    print("Max score is : %.3f, order: %s" % (max_score, max_order))
    return min_score, max_score, min_order, max_order

# 2 repeat
def angleM_repeat(x):
    x_len = len(x)
    result = np.zeros((x_len, x_len))
    for i, _x in enumerate(x):
        points = np.repeat([_x], x_len, axis=0)
        residual = x - points
        result[i] = np.arctan2(residual[:,1], residual[:,0])
    return result

def angleM(x):
    result = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            result[i,j] = get_angle(x[i],x[j])
    return result
