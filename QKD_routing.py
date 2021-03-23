import networkx as nx
import numpy as np
import numpy.core.multiarray
import scipy as sp
import pandas as pd
from copy import deepcopy

import random
import sys


def init(max_key_rate, nodes):
    """Generate initial and target topologies for demonstration of
    routing algorithm.
 
    Args:
        max_key_rate: set maximum possible key rate on link
        nodes: number of nodes in the network
    Returns:
        rate_matrix: current topology presented by matrix with key rates on 
            each node
        target_matrix: target topology of the network
    """


    #give initial key rates on each link
    rate_matrix = nx.complete_graph(nodes)
    for (u,v,w) in rate_matrix.edges(data=True):
        w['weight'] = random.uniform(0,max_key_rate)
        
    pos = nx.spring_layout(rate_matrix)
    nx.draw_networkx(rate_matrix,pos)
    labels = nx.get_edge_attributes(rate_matrix,'weight')
    for link in labels:
        labels[link] = float('{:.2f}'.format(labels[link]))
    nx.draw_networkx_edge_labels(rate_matrix,pos, edge_labels=labels)
    
    rate_matrix = nx.adjacency_matrix(rate_matrix)
    rate_matrix = rate_matrix.toarray()
    rate_matrix = rate_matrix.astype(float)

    
    #give target key rates on each link
    target_matrix = nx.complete_graph(nodes)
    
    for (u,v,w) in target_matrix.edges(data=True):
        w['weight'] = random.randint(0, max_key_rate)

    target_matrix = nx.adjacency_matrix(target_matrix)
    target_matrix = target_matrix.toarray()
    target_matrix = target_matrix.astype(float)
    
    return rate_matrix, target_matrix


def dif(rate_matrix,
        target_matrix):
    """Calculate discrepency and sorting it;
    the aim is to get link with the smallest key rate.

    Args:
        rate_matrix: current topology presented by matrix with 
            key rates on each node
        target_matrix: target topology of the network
    Returns:
        diff_vec: vector with full discrepansy information
    """
    
    diff = rate_matrix - target_matrix
    diff_max = diff.max()
    diff_min = diff.min()
    
    diff_min_index = np.where(diff == diff.min())
    
    #get indexes of sender and reciver nodes
    n = 0
    snd = diff_min_index[0][n]
    rcv = diff_min_index[1][n]
        
    return diff, diff_max, diff_min, snd, rcv


def paths_fnd(rate_matrix,
              diff_vec):
    """Find all ways between source and target nodes.

    Args:
        rate_matrix: current topology presented by matrix with key rates on 
            each node
        diff_vec: vector with full discrepansy information
    Returns:
        paths: all possible ways between source and target nodes
    """

    
    paths = []
    g = nx.Graph(rate_matrix)
    for path in nx.all_simple_paths(g, source=diff_vec[3], target=diff_vec[4]):
        paths.append(path)
    
    return paths


def path_slc(diff_vec,
             paths):
    """Select donor path to provide choosed link with key.

    Args:
        diff_vec: vector with full discrepansy information
        paths: all possible ways between source and target nodes
    Returns:
        cur_path: current or choosed donor path
        biggest_min: minimum key rate on current path
    """

    #consider minimum elements on each path in possible paths
    min_elem = []
    
    for path in paths:
        i = 0
        min_on_i  = diff_vec[1]
        while(i < len(path) - 1):
            path_s = path[i]
            path_t = path[i+1]
            cur_state = diff_vec[0][path_s][path_t]
            if cur_state < min_on_i:
                min_on_i = cur_state
            i += 1
        min_elem.append(min_on_i)
        
    #find path with biggest minimum element
    biggest_min = np.amax(min_elem)
    cur_path_ind = np.where(min_elem == biggest_min)
    
    cur_path = paths[cur_path_ind[0][0]]
    
    return cur_path, biggest_min


def boost(rate_matrix,
          cur_path,
          biggest_min,
          diff_vec,
          dlt):
    """Change current topology.
    Args:
        rate_matrix,
        cur_path,
        biggest_min,
        diff_vec,
        dlt: value of changing
    Returns:
        paths: all possible ways between source and target nodes
    """
    
    first = 0
    second = 0
    end = 0
    eps = dlt * 0.000000001
    
    g = nx.Graph(rate_matrix)
    srs = cur_path[0]
    n = len(cur_path) - 1
    trg = cur_path[n]
    
    
    #overshoot catch
    a = diff_vec[2] + dlt #minimal element of D + dlt
    b = biggest_min - dlt #minimal element of current donor path - dlt
    print("the situation before running", abs(a-b))
    
    
    if (abs(a-b) < eps):
        
        print("last step")
        print("dlt", dlt)
        
        #take key from each link on current path
        for path in nx.all_simple_paths(G, source=srs, target=trg):
            if path == cur_path:
                i = 0
                while(i<len(path)-1):
                    path_s = path[i]
                    path_t = path[i+1]
                    R[path_s][path_t] -= dlt
                    R[path_t][path_s] -= dlt
                    i += 1
                
        #provide the poorest link with key
        rate_matrix[srs][trg] += dlt
        rate_matrix[trg][srs] += dlt
        rate_matrix_min = np.amin(rate_matrix)
        rate_matrix_max = np.amax(rate_matrix)
        
        end = 1
        return end
        
    #normal situation: boost does not change roles
    
    if a < b:
        print("dlt", dlt)
        
        #take key from each link on current path
        for path in nx.all_simple_paths(g, source=srs, target=trg):
            if path == cur_path:
                i = 0
                while (i < len(path)-1):
                    path_s = path[i]
                    path_t = path[i+1]
                    rate_matrix[path_s][path_t] -= dlt
                    rate_matrix[path_t][path_s] -= dlt
                    i += 1
                
        #provide the poorest link with key
        rate_matrix[srs][trg] += dlt
        rate_matrix[trg][srs] += dlt
        rate_matrix_min = np.amin(rate_matrix)
        rate_matrix_max = np.amax(rate_matrix)
        
        first = 1
        
        
    #after boost poorest link skip donor rate. need to equalize them
    
    if a > b:
        print('last step. equalize')
        dlt_final = abs(biggest_min-diff_vec[2])/2
        print("dlt to equalize:", dlt_final)
        
        #take key from each link on current path
        for path in nx.all_simple_paths(g, source=srs, target=trg):
            if path == cur_path:
                i = 0
                while(i<len(path)-1):
                    path_s = path[i]
                    path_t = path[i+1]
                    rate_matrix[path_s][path_t] -= dlt_final
                    rate_matrix[path_t][path_s] -= dlt_final
                    i += 1
                    
        #provide the poorest link with key
        rate_matrix[srs][trg] += dlt_final
        rate_matrix[trg][srs] += dlt_final
        rate_matrix_min = np.amin(rate_matrix)
        rate_matrix_max = np.amax(rate_matrix)
        
        second = 1
        
        if dlt_final < eps:
            end = 1
        
    return end


def main(init, dlt):
    """Runs the algorithm of routing.

    Args:
        init: parametrs to initialize our network
        dlt: step for each iteration

    Returns:
        d_max: vector of maximum differences at each step
        d_min: vector of minimum differences at each step
        step: the number of steps from start to stop of the algorithm
        average_key: the sum by the weights of the current topology minus 
            the sum by the target, related to the sum by the current
        plight: sum of all differences
        dlt: the size of the delta to which we change the topology at each 
            step
        donor_paths: a vector containing all the paths from which 
            we occupied the delta, we need to then write the routing script
    """
    
    rate_matrix = init[0]
    target_matrix = init[1]
    
    donor_paths = [] #collect all pathes that give key
    d_max = []
    d_min = []
    key_volume_crnt = 0
    t_crit_ind = 0
    
    # N is number of steps, we run algoritm until all links are satisfyed or 
    # until it's clear that there is
    # not enough key to satisfy each link
    
    step = 0
    i = 1
    
    while i == 1:
        print(f"-------------------------step {step}-------------------------")
        
        #PRELEMINARIES
        
        diff_vec = []
        diff_vec = dif(rate_matrix, target_matrix)
        
        average_key = (sum(rate_matrix)-sum(target_matrix))/sum(rate_matrix)
        d_max.append(diff_vec[1])
        d_min.append(diff_vec[2])
        
        print("sender, reciever: ", diff_vec[3], diff_vec[4])
        
        paths = paths_fnd(rate_matrix, diff_vec)
        print("all the paths: ", paths)
        
        if len(paths) == 0:
            t_crit_ind = 1
            print("end")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            return (d_max, d_min, step, average_key, plight, dlt, donor_paths, 
                t_crit_ind)

        ps = []
        ps = path_slc(diff_vec, paths) 
        #ps=[current donor path, biggest minimal]
        
        cur_path = ps[0]
        biggest_min = ps[1]
        donor_paths.append(cur_path)
        print("choosed path: ", cur_path)
        
        #EXECUTION
        
        #catch problems:
        if diff_vec[2] == 0:
            print("complete")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            t_crit_ind = 1
            return (d_max, d_min, step, average_key, plight, dlt, donor_paths,
                t_crit_ind)
        
        if len(cur_path) == 2:
            print("can not be complete")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            return (d_max, d_min, step, average_key, plight, dlt, donor_paths,
                t_crit_ind)
        
        # if no problems go to boost
        end = boost(rate_matrix, cur_path, biggest_min, diff_vec, dlt)
        print("R:\n", rate_matrix)
        
        # cath problems after boost: last step indicator or minimum key 
        # rate decay
        if end == 1:
            print("end")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            return (d_max, d_min, step, average_key, plight, dlt, donor_paths, 
                t_crit_ind)
        
        if len(d_min) > 3:
            n = len(d_max) - 1
            if d_min[n] < d_min[n-1]:
                print("can not be complete. decay")
                p = 0
                for k in range(1,len(diff_vec[0])):
                    for p in range(k):
                        p += diff_vec[0][k][p]
                plight=p
                return (d_max, d_min, step, average_key, plight, dlt, 
                    donor_paths, t_crit_ind)
        
        #renew our discrepancy
        diff_vec = dif(rate_matrix, target_matrix)
        print("D:\n", diff_vec[0])
        
        d_max[step] = diff_vec[1]
        d_min[step] = diff_vec[2]

        key_volume_crnt = rate_matrix.sum()/2 #amount of key in network
        
        step += 1



# ROUTING WITH THE SECRET KEY SHARING

def path_slc_sksh(diff_vec, paths, privacy):
    """ Choosing path for routing within given set of paths.
    
    Args:
        diff_vec: vector of descrepancies
        paths: set of paths
        privacy: number of nodes for the secret sharing

    Returns:
        cur_path: the chosen path
    """

    #consider minimum elements on each path in possible paths
    min_elem = []
    cur_path = []
    
    for path in paths:
        i = 0
        min_on_i  = diff_vec[1]
        while(i<len(path)-1):
            path_s = path[i]
            path_t = path[i+1]
            cur_state = diff_vec[0][path_s][path_t]
            if cur_state < min_on_i:
                min_on_i = cur_state
            i += 1
        min_elem.append(min_on_i)
        
    
    min_elem_sort = sorted(min_elem, reverse=True)
    
    for i in range(privacy):
        m = min_elem_sort[i]
        cur_path_ind = np.where(min_elem == m)
        cur_path.append(paths[cur_path_ind[0][0]])
    
    return cur_path


def main_sksh(init, dlt, privacy):
    """Runs routing algorithm with secret sharing.

    Args:
        init: parametrs to initialize network
        dlt: step for each iteration
        privacy: number of nodes for the secret sharing 

    Returns:
        d_max: vector of maximum differences at each step
        d_min: vector of minimum differences at each step
        step: the number of steps from start to stop of the algorithm
        average_key: the sum by the weights of the current topology minus 
            the sum by the target, related to the sum by the current
        plight: sum of all differences
        dlt: the size of the delta to which we change the topology at each 
            step
        donor_paths: a vector containing all the paths from which 
            we occupied the delta, we need to then write the routing script
        t_crit_ind: critical index
    """
    
    rate_matrix = init[0]
    target_matrix = init[1]
    
    donor_paths = [] #collect all pathes that give key
    d_max = []
    d_min = []
    sum_key = []
    plight = [] #illustrates how poor or jaded our network is
    key_volume_crnt = 0
    t_crit_ind = 0
    
    diff_vec = []
    diff_vec = dif(rate_matrix, target_matrix)
        
    d_max.append(diff_vec[1])
    d_min.append(diff_vec[2])
    
    # N is number of steps, we run algoritm until all links are satisfyed or 
    # until it's clear that there is
    # not enough key to satisfy each link
    
    step = 0
    i = 1
    
    while i == 1:
        
        print(f"-------------------------step {step}-------------------------")
        
        #PRELEMINARIES
        
        print("sender, reciever: ", diff_vec[3], diff_vec[4])
        
        paths = paths_fnd(rate_matrix, diff_vec)
        print("all the paths: ", paths)
        
        if len(paths) == 0:
            t_crit_ind = 1
            print("end")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            return (d_max, d_min, step, sum_key, plight, dlt, donor_paths, 
                t_crit_ind)
        
        #privacy = init[1][diff_vec[3]][diff_vec[4]]
        
        cur_path = path_slc_sksh(diff_vec, paths, privacy)
        donor_paths.append(cur_path)
        print("choosed paths: ", cur_path)
        
        #EXECUTION
        
        #catch problems:
        
        if diff_vec[2] == 0:
            print("complete")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            t_crit_ind = 1
            return (d_max, d_min, step, sum_key, plight, dlt, donor_paths, 
                t_crit_ind)
        
        if len(cur_path[0]) == 2:
            print("can not be complete")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            return (d_max, d_min, step, sum_key, plight, dlt, donor_paths, 
                t_crit_ind)
        
        #if no problems go to boost
        
        if privacy > 1:
            for n in range(privacy):
                biggest_min = np.amax(cur_path[n])
                end = boost(rate_matrix, cur_path[n], biggest_min, 
                    diff_vec, dlt)
                print("R:\n", rate_matrix, "\n", "privacy:\n", n)
                
        if privacy == 1:
            biggest_min = np.amax(cur_path[0])
            end = boost(rate_matrix, cur_path[0], biggest_min, diff_vec, dlt)
        
        # cath problems after boost: last step indicator or minimum key rate 
        # decay
            
        if end == 1:
            print("end")
            p = 0
            for k in range(1,len(diff_vec[0])):
                for p in range(k):
                    p += diff_vec[0][k][p]
            plight=p
            return (d_max, d_min, step, sum_key, plight, dlt, donor_paths, 
                t_crit_ind)
        
        if len(d_min) > 3:
            n = len(d_max) - 1
            if d_min[n] < d_min[n-1]:
                print("can not be complete. decay")
                p = 0
                for k in range(1,len(diff_vec[0])):
                    for p in range(k):
                        p += diff_vec[0][k][p]
                plight=p
                return (d_max, d_min, step, sum_key, plight, dlt, donor_paths, 
                    t_crit_ind)
        
        #renew our discrepancy
        diff_vec = dif(rate_matrix, target_matrix)
        print("D:\n", diff_vec[0])
        
        d_max.append(diff_vec[1])
        d_min.append(diff_vec[2])

        key_volume_crnt = rate_matrix.sum()/2 #amount of key in network
        sum_key.append(key_volume_crnt)
        
        step += 1