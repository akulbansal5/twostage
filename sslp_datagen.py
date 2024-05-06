"""
generate data for sslp instances

    lp file containing the global problem
    one is a dictionary containing the stochastic information

"""

import cplex as cpx
import pickle as pk
import os
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/stofiles/"
lp_path = dir_path + "/lpfiles/"
pickle_path = dir_path + "/pk_files/"


def generate_data(servers, clients, scens):


    """
    servers (int): number of servers in the instance
    clients (int): number of clients in the instance
    scens (int): number of scenarios in the instance
    """


    c_low = 40
    c_high = 80

    d_low = 0
    d_high = 25

    q_fixed = 1000
    v = servers
    r = 1.5


    #cost of locating a server at a location j: c_j
    fcosts = np.random.randint(c_low, c_high, size = servers)


    #revenue from client i being served by server at location j: q_ij
    d_sample = np.random.randint(d_low, d_high, size = (clients, servers))


    #Client i resource demand from server at location j: dij    
    q = {}
    # the revenue and the cost
    for i in range(1, clients + 1):
        q[i] = {}
        for j in range(1, servers + 1):
            q[i][j] = d_sample[i-1][j-1]



    total = sum([max(q[i].values()) for i in range(1, clients + 1)])
    u = int((r/v)*total)
    rhs_matrix = np.random.binomial(1, 0.5, size = (scens, clients))

    
    return fcosts, q, q_fixed, v, u, rhs_matrix


def create_global_problem(fcosts, q, q_fixed, v, u, servers, clients, lplocation, instName):

    """
    
    """

    model = cpx.Cplex()

    
    mNames = [f'x_{i}' for i in range(1, servers+1)]
    fcosts_new = [elem.item() for elem in fcosts]
    model.variables.add(ub = [1]*servers, names = mNames, types = ['B']*servers, obj = list(fcosts_new))


    sNames = [f'y_{i}_{j}' for i in range(1, clients+1) for j in range(1, servers+1)]
    sObj   = [-q[i][j].item() for i in range(1, clients+1) for j in range(1, servers+1)]
    sCount = servers*clients
    model.variables.add(ub = [1]*sCount, names = sNames, types = ['B']*sCount, obj = sObj)


    sNames2 = [f'x_{i}_{0}' for i in range(1, servers+1)]
    sObj2 = [q_fixed]*servers
    model.variables.add(names = sNames2, obj = sObj2)


    ind = mNames.copy()
    val = [-1]*servers

    index = 1
    model.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind, val = val)], rhs = [-v], senses = ['G'], names = [f'c{index}'])
    index += 1

    for j in range(1, servers+1):
        ind = [f'x_{j}'] + [f'y_{i}_{j}' for i in range(1, clients+1)] + [f'x_{j}_{0}']
        val = [u] + [-q[i][j].item() for i in range(1, clients+1)] + [1]
        model.linear_constraints.add(lin_expr= [cpx.SparsePair(ind = ind, val = val)], senses = ['G'], rhs = [0], names = [f'c_{index}'])
        index += 1


    start = index
    for i in range(1, clients+1):
        ind = [f'y_{i}_{j}' for j in range(1, servers+1)]
        val = [1]*len(ind)
        model.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind, val = val)], senses = ['E'], rhs = [1], names = [f'c_{index}'])
        index += 1
    end = index

    eqInd = [f'c_{index}' for index in range(start, end)]
    
    model.write(lplocation + instName + ".lp")

    return eqInd



def create_scenario_data(rhs_matrix, scens, eqInd):

    """
    create scenario data
    """

    rhs_map = {}
    for s in range(scens):
        
        rhs_map[s] = dict(zip(eqInd, rhs_matrix[s]))
    
    return rhs_map
        

def gen_inst_data(servers, clients, scens, lplocation, pickle_path, instName):

    fcosts, q, q_fixed, v, u, rhs_matrix = generate_data(servers, clients, scens)
    
    eqInd = create_global_problem(fcosts, q, q_fixed, v, u, servers, clients, lplocation, instName)
    
    rhs_map = create_scenario_data(rhs_matrix, scens, eqInd)

    pk.dump(rhs_map, open(pickle_path + "_" + instName + "_rhsmap.p", 'wb'))


gen_inst_data(15, 100, 100, lp_path, pickle_path, 'sslp_new_')


    






    
    





