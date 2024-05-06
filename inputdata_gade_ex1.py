"""

load the data from the example given by Gade


"""


import os
import copy
import cplex as cpx
import numpy as np
from primitives import Subproblem

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/stofiles"
lp_path = dir_path + "/lpfiles/"


def master_generation_gade():

    """
    returns: the master problem (cplex model)
    """

    master = cpx.Cplex()
    master.variables.add(obj = [-1.5, -4.0, 1.0], types = ['B']*2 + ['C'], lb = [0, 0, -90], names = ['x_1', 'x_2', 't'])


    return master


def def_sub_gade(rhs_const):

    """
    here we define the integer program related to the subproblem
    def_sub gives the deterministic equivalent formulation for the subproblem
    """

    sub_prob = cpx.Cplex()
    sub_varNames = ['y0', 'y_1', 'y_2', 'y_3', 'y_4', 'R', 'x_1', 'x_2']
    sub_varTypes = ['I'] + ['B']*4 + ['I'] + ['B']*2
    lbs = [-cpx.infinity] + [0]*4 +[0] +  [0]*2
    objs = [1] + [0]*7
    
    sub_prob.variables.add(obj = objs, types = sub_varTypes, lb = lbs, names = sub_varNames)

    row1 = cpx.SparsePair(ind = ['y0', 'y_1', 'y_2', 'y_3', 'y_4', 'R'], val = [1, 16, 19, 23, 28, -100])
    row2 = cpx.SparsePair(ind = ['y_1', 'y_2', 'y_3', 'y_4', 'R', 'x_1'], val = [2, 3, 4, 5, -1, 1])
    row3 = cpx.SparsePair(ind = ['y_1', 'y_2', 'y_3', 'y_4', 'R', 'x_2'], val = [6, 1, 3, 2, -1, 1])
    boundRows = [cpx.SparsePair(ind = [f'y_{i}'], val = [1]) for i in range(1,5)]


    rhs_col = [0] + [0,0] + [1]*4                                   #note we need to update this for each scenario
    row_names = [f'c_{i}' for i in range(2, 2+7)]                   #name of constraints 

    sub_prob.linear_constraints.add(lin_expr = [row1, row2, row3] + boundRows, senses = ['E'] + ['L']*6, rhs = rhs_col, names = row_names)
    cons = sub_prob.linear_constraints.get_num()

    sub_prob.linear_constraints.set_rhs([(i, float(rhs_const[i])) for i in range(cons)])

    return sub_prob


def subproblem_fixed_recourse_gade_ver0(master):

    """
    declare subproblem with fixed recourse
    ver0: note setting the variable to be continuous types produces the variables
    """

    cpx_prob = def_sub_gade()
    cpx_prob.variables.delete(master.variables.get_names()[:-1])
    for var_name in cpx_prob.variables.get_names():
        if var_name == 'y0' or var_name == 'R':
            cpx_prob.variables.set_types(var_name, 'C')
        else:
            cpx_prob.variables.set_types(var_name, 'C')
    
    return cpx_prob


def subproblem_fixed_recourse_gade(rhs_const):

    """
    here we define the integer program related to the subproblem
    def_sub gives the deterministic equivalent formulation for the subproblem
    """

    sub_prob = cpx.Cplex()
    sub_varNames = ['y0', 'y_1', 'y_2', 'y_3', 'y_4', 'R']
    
    lbs = [-cpx.infinity] + [0]*4 +[0]
    objs = [1] + [0]*5
    
    sub_prob.variables.add(obj = objs, lb = lbs, names = sub_varNames)

    row1 = cpx.SparsePair(ind = ['y0', 'y_1', 'y_2', 'y_3', 'y_4', 'R'], val = [1, 16, 19, 23, 28, -100])
    row2 = cpx.SparsePair(ind = ['y_1', 'y_2', 'y_3', 'y_4', 'R'], val = [2, 3, 4, 5, -1])
    row3 = cpx.SparsePair(ind = ['y_1', 'y_2', 'y_3', 'y_4', 'R'], val = [6, 1, 3, 2, -1])
    boundRows = [cpx.SparsePair(ind = [f'y_{i}'], val = [1]) for i in range(1,5)]


    rhs_col = [0] + [0,0] + [1]*4                                   #note we need to update this for each scenario
    row_names = [f'c_{i}' for i in range(2, 2+7)]                   #name of constraints 

    sub_prob.linear_constraints.add(lin_expr = [row1, row2, row3] + boundRows, senses = ['E'] + ['L']*6, rhs = rhs_col, names = row_names)
    
    cons = sub_prob.linear_constraints.get_num()

    rhs_map = [(i, float(rhs_const[i])) for i in range(cons)]
    sub_prob.linear_constraints.set_rhs(rhs_map)

    return sub_prob
        

master_gade = master_generation_gade()

# master_gade.write("gades_example.lp")



sub0 = Subproblem(0)
sub0.rhs_const = np.array([0, 5, 2] + [1]*4)
sub0.cpx_model = subproblem_fixed_recourse_gade(sub0.rhs_const)
sub0.probability = 0.5

sub0_cons_count = sub0.cpx_model.linear_constraints.get_num()

master_main_vars = master_gade.variables.get_num()-1

sub0_rhs_tech_row = np.zeros((sub0_cons_count, master_main_vars))
sub0_rhs_tech_row[0, :] = np.array([0, 0])
sub0_rhs_tech_row[1, :] = np.array([-1, 0])
sub0_rhs_tech_row[2, :] = np.array([0, -1])
sub0.rhs_tech_row = sub0_rhs_tech_row


sub1 = Subproblem(1)
sub1.rhs_const    = np.array([0, 10, 3] + [1]*4)
sub1.cpx_model    = subproblem_fixed_recourse_gade(sub1.rhs_const)
sub1.probability  = 0.5
sub1.rhs_tech_row = copy.deepcopy(sub0_rhs_tech_row)


sub0.cpx_def       = def_sub_gade(sub0.rhs_const)
sub0.cpx_def_copy  = def_sub_gade(sub0.rhs_const)


sub1.cpx_def      = def_sub_gade(sub1.rhs_const)                            
sub1.cpx_def_copy = def_sub_gade(sub1.rhs_const)                          #what is the cpx_def that we are getting?


sub_dict_gade = {}
sub_dict_gade[0] = sub0
sub_dict_gade[1] = sub1











