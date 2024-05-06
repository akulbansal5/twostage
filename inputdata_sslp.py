"""

File for loading the master problem, subproblems,
random right hand sides and technology matrix for 
SSLP problem instances.

For these instances technology matrix
and recourse matrix is fixed.


this seems be older version of sslpInstance.py
"""


import os
import copy
import typing
import cplex as cpx
from load_smps import loadScenarios, loadScenarios_rui
from primitives import GlobalModel, Subproblem, turnLogOff
import numpy as np
import pickle as pk

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/stofiles/"
lp_path = dir_path + "/lpfiles/"
pickle_path = dir_path + "/pk_files/"


def getObjAsMatrix(global_model, Nn, Nfv):
    """

    """

    qMtr = np.zeros((Nn, Nfv))

    for var in global_model.secondStageBinVars:

        iInd = int(var.split("_")[1])  # client
        jInd = int(var.split("_")[2])  # server location

        qMtr[iInd-1, jInd -
             1] = abs(global_model.cpx_model.objective.get_linear(var))

    return qMtr


def getRuisDMatrix(global_model, Nn, Nfv):
    """
    get the dMatrix as desired in the model


    """

    dMtr = np.zeros((Nn, Nfv))

    for j in range(Nfv):
        for i in range(Nn):
            dMtr[i, j] = abs(global_model.cpx_model.linear_constraints.get_coefficients(
                j+1, f'y_{i+1}_{j+1}'))

    return dMtr


def getRandomScenMatr(global_model, Nn, Nfv, Nscen):
    """
    get the probability vector also from the problem
    """

    hMtr = np.zeros((Nn, Nscen))
    pVec = np.zeros((Nscen))
    scen_ids = global_model.scenario_dict.keys()

    for s in scen_ids:
        pVec[s] = global_model.scenario_dict[s].probability

        rcost = np.array(
            list(global_model.scenario_dict[s].constraintMap.values()))
        for i in range(len(rcost)):

            hMtr[i, s] = rcost[i]
            # print("--")

    return hMtr, pVec


def convertToRui(global_model):
    """
    Input:
        global_model: global_model should be have scenario dictionary loaded into it

    Take our file and convert to data structures required in Rui
    Check the read_data function in Rui for more info


    #convert directly to Rui's data structures




        self.Nfv = fline[0]								# number of first stage variables (or server locations) 
                self.Nn = fline[1]								# number of of clients 
                self.Nscen = fline[2] 							# number of scenarios
                # Read cVec
                self.cVec = self._readVec(x[1], np.float32)     # cost vector for first stage problem
                # Read qMtr
                self.qMtr = self._readMtr(x[2])					# captures the objective coefficients q_ij corresponding to variables y_ij
                # Read q0Vec
                self.q0Vec = self._readVec(x[3], np.float32)    # objective function corresponding to over flow variables in the second stage
                # Read dMtr
                self.dMtr = self._readMtr(x[4])					# dMatrix is same as in problem statement (authors taking dMatrix to be same as qMatrix)
                # Read Nu
                self.Nu = float(x[5]) 							# u value in the problem formulation (server capacity -> defined to be a large value)
                #Read hMtr
                self.hMtr = self._readMtr(x[6])					# this is the random rhs side h(i, s) for client i and scenario s


                # print("hMtr shape: ", self.hMtr.shape)			

                #Read pVec										
                self.pVec = self._readVec(x[7], np.float32)		#probability of each scenario
                for s in range(self.Nscen):						#information for each scenario
                        self.cutlist[s] = []
                        self.coeflist[s] = []
                        self.thetaCutList[s] = []

    """

    # number of first stage variables
    Nfv = len(global_model.original_masterVars)
    Nn = len(global_model.secondStageBinVars)//Nfv  # number of servers
    Nscen = len(global_model.scenario_dict)  # number of scenarios
    cVec = global_model.cpx_model.objective.get_linear(
        global_model.original_masterVars)  # cost of first stage decision variable
    qMtr = getObjAsMatrix(global_model, Nn, Nfv)
    dMtr = getRuisDMatrix(global_model, Nn, Nfv)
    q0Vec = list(map(abs, global_model.cpx_model.objective.get_linear(
        global_model.secondStageContVars)))
    Nu = abs(global_model.cpx_model.linear_constraints.get_coefficients(1, f'x_1'))
    xBound = abs(global_model.cpx_model.linear_constraints.get_rhs(0))
    hMtr, pVec = getRandomScenMatr(global_model, Nn, Nfv, Nscen)

    return Nfv, Nn, Nscen, xBound, cVec, qMtr, dMtr, q0Vec, Nu, hMtr, pVec


def pickleDump(instance_name, Nfv, Nn, Nscen, xBound, cVec, qMtr, dMtr, q0Vec, Nu, hMtr, pVec):
    """

    """

    inst_info = [Nfv, Nn, Nscen, Nu, xBound]

    pk.dump(inst_info, open(pickle_path + instance_name + f"_info.p", 'wb'))
    pk.dump(cVec, open(pickle_path + instance_name + f"_cVec.p", 'wb'))
    pk.dump(qMtr, open(pickle_path + instance_name + f"_qMtr.p", 'wb'))
    pk.dump(dMtr, open(pickle_path + instance_name + f"_dMtr.p", 'wb'))
    pk.dump(q0Vec, open(pickle_path + instance_name + f"_q0Vec.p", 'wb'))
    pk.dump(hMtr, open(pickle_path + instance_name + f"_hMtr.p", 'wb'))
    pk.dump(pVec, open(pickle_path + instance_name + f"_pVec.p", 'wb'))


def master_generation(global_model: GlobalModel):
    """
    returns : the master problem (cplex model)

    """

    master = cpx.Cplex()
    master.variables.add(obj=global_model.cpx_model.objective.get_linear(
        global_model.masterVars), types=["B"]*global_model.masterVarsLen, names=global_model.masterVars)

    # only one constraint appears in the master problem
    master.linear_constraints.add(lin_expr=[global_model.cpx_model.linear_constraints.get_rows("c1")], senses=[
                                  global_model.cpx_model.linear_constraints.get_senses("c1")], rhs=[global_model.cpx_model.linear_constraints.get_rhs("c1")], names=["c1"])

    # declare a variable t and obtain a suitable lowerbound
    t_lb = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageBinVars))
    master.variables.add(obj=[1.0], lb=[t_lb], types=["C"], names=['t'])

    global_model.masterVars.append('t')
    global_model.masterVarsLen += 1

    # master problem is declared over here
    global_model.master = master

    # master.parameters.preprocessing.presolve	= 0
    turnLogOff(master)

    return master


def master_generation_ver2(global_model: GlobalModel):
    """
    returns : the master problem (cplex model)
    ver2:     this is multi-cut version instead of single-cut version
    """

    master = cpx.Cplex()
    master.variables.add(obj=global_model.cpx_model.objective.get_linear(
        global_model.masterVars), types=["B"]*global_model.masterVarsLen, names=global_model.masterVars)

    # only one constraint appears in the master problem
    master.linear_constraints.add(lin_expr=[global_model.cpx_model.linear_constraints.get_rows("c1")], senses=[
                                  global_model.cpx_model.linear_constraints.get_senses("c1")], rhs=[global_model.cpx_model.linear_constraints.get_rhs("c1")], names=["c1"])

    # declare a variable t and obtain a suitable lowerbound
    # TODO: change the bound on variable t

    t_lb = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageBinVars))

    for idx in global_model.scenario_dict:
        sub = global_model.scenario_dict[idx]
        master.variables.add(obj=[sub.probability], lb=[
                             t_lb], types=["C"], names=[f't_{idx}'])
        global_model.masterVars.append(f't_{idx}')
        global_model.masterVarsLen += 1

    # master problem is declared over here
    global_model.master = master

    # master.parameters.preprocessing.presolve	= 0
    turnLogOff(master)

    return master


def updateConstraintRowsIndex(rows):


    constrs = []
    for sparseObj in rows:

        ind, val = sparseObj.unpack()

        new_ind = [int(i) for i in np.array(ind) + np.ones(len(ind))]

        constrs.append(cpx.SparsePair(ind=new_ind, val=val))


    return constrs


def updateConstraintRowsIndex_ver2(rows, var_name, coeff):


    constrs = []
    for sparseObj in rows:

        ind, val = sparseObj.unpack()


        new_ind_arr = np.array(ind) + np.ones(len(ind))

        new_ind = [int(i) for i in new_ind_arr]

        val = list(val)

        new_ind.append(var_name)
        val.append(coeff)
        constrs.append(cpx.SparsePair(ind=new_ind, val=val))

    return constrs


def updateConstraintDefn(master):
    """
    update the constraint definition where indices are integers to variable names
    """

    var_names = master.variables.get_names()

    cons = master.linear_constraints.get_rows()

    new_cons = []
    for sparseObj in cons:
        ind, val = sparseObj.unpack()
        for i in range(len(ind)):
            ind[i] = var_names[ind[i]]
        new_cons.append(cpx.SparsePair(ind=ind, val=val))

    return new_cons


def updateLowerBoundForSubproblems(global_model: GlobalModel):
    """
    updates the lower bound for the subproblems

    """

    global_model.global_lower_bd = 0

    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        sub = global_model.scenario_dict[idx]
        lb = 0
        for conName in sub.constraintMap:

            rhs = sub.constraintMap[conName]
            sparseObj = sub.cpx_model.linear_constraints.get_rows(conName)
            indices, values = sparseObj.unpack()
            coeff_pairs = [('c0', i) for i in indices]

            lb = lb - \
                max(sub.cpx_model.linear_constraints.get_coefficients(coeff_pairs))*rhs
        sub.lower_bound = lb
        global_model.global_lower_bd += sub.probability*sub.lower_bound


def def_sub_sslp(sub: Subproblem, master: cpx.Cplex):
    """
    given a Subproblem object we write the cpx model for its deterministic equaivalent formulation
    assumes that the given scenario happens with probability 1
    returns a cplex object with problem definition

    """

    sub_def = cpx.Cplex()
    sub_def_names = sub.cpx_model.variables.get_names()
    sub_def.variables.add(names=sub_def_names, lb=sub.cpx_model.variables.get_lower_bounds(
        sub_def_names), ub=sub.cpx_model.variables.get_upper_bounds(sub_def_names))

    for var_name in sub_def_names:
        if var_name == "y0":
            sub_def.variables.set_types(
                var_name, sub_def.variables.type.integer)
        elif "y" in var_name:
            sub_def.variables.set_types(
                var_name, sub_def.variables.type.binary)
        elif "x" in var_name:
            sub_def.variables.set_types(
                var_name, sub_def.variables.type.integer)

    # remove column that represents the variable t
    cols = master.variables.get_num() - 1
    sub_def_x = master.variables.get_names()[:-1]

    sub_def.variables.add(names=sub_def_x, types=['B']*cols)

    sub_def.linear_constraints.add(lin_expr=sub.cpx_model.linear_constraints.get_rows(), senses=sub.cpx_model.linear_constraints.get_senses(
    ), rhs=sub.cpx_model.linear_constraints.get_rhs(), names=sub.cpx_model.linear_constraints.get_names())

    # update the constraints to include x variables and rhs
    cons_count = sub_def.linear_constraints.get_num()
    for j in range(cons_count):
        sub_def.linear_constraints.set_linear_components(
            j, cpx.SparsePair(ind=sub_def_x, val=list(-sub.rhs_tech_row[j])))
        sub_def.linear_constraints.set_rhs(j, float(sub.rhs_const[j]))

    # also update the constraints to include the master constraints
    newConstrs = updateConstraintDefn(master)

    # add linear constraints from master to the def problem
    sub_def.linear_constraints.add(lin_expr=newConstrs, senses=master.linear_constraints.get_senses(
    ), rhs=master.linear_constraints.get_rhs(), names=master.linear_constraints.get_names())

    # sub_def.write(lp_path +  f"subdef_{sub.idx}.lp")

    return sub_def


def subproblem_sslp(global_model: GlobalModel):
    """
    returns: subproblem for sslp problem, has fixed recourse
    """

    # sslp subproblem
    sub = cpx.Cplex()

    # add all variables

    # we add y0 variable
    lower_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageBinVars))

    # upper bound attained by solving
    upper_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageContVars))

    # add y0 variable for the objective function
    
    sub.variables.add(obj=[1.0], names=["y0"], lb=[-cpx.infinity])

    # x variables in the first stage (x variables)
    # here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
    sub.variables.add(names=global_model.original_masterVars)

    # y variables
    sub.variables.add(names=global_model.secondStageBinVars, lb=[
                      0]*len(global_model.secondStageBinVars))

    # x variables in second stage (y_{0, \omega} variables)
    sub.variables.add(names=global_model.secondStageContVars)

    # also add the linear constraint corresponding to the objective function

    ind1 = global_model.secondStageBinVars.copy()
    val1 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageBinVars)))

    ind2 = global_model.secondStageContVars.copy()
    val2 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageContVars)))

    ind0 = ["y0"]
    val0 = [1]

    # here the constraints are added by variable names
    sub.linear_constraints.add(lin_expr=[cpx.SparsePair(
        ind=ind0 + ind1 + ind2, val=val0 + val1 + val2)], senses=["E"], rhs=[0], names=['c0'])

    y0_index = sub.variables.get_num()-1
    sub.y0_index = y0_index

    # add linear constraints (other than the first constraint which belongs to the master problem)
    # first constraint is removed because it belongs to the master problem
    newRows = updateConstraintRowsIndex(
        global_model.cpx_model.linear_constraints.get_rows(global_model.constrs[1:]))

    sub.linear_constraints.add(newRows, senses=global_model.cpx_model.linear_constraints.get_senses(
        global_model.constrs[1:]), rhs=global_model.cpx_model.linear_constraints.get_rhs(global_model.constrs[1:]), names=global_model.constrs[1:])

    yBoundConstrs = [cpx.SparsePair(ind=[yVar], val=[1])
                     for yVar in global_model.secondStageBinVars]

    yBoundConstr_names = [f"c{i}" for i in range(
        len(global_model.constrs)+1, len(global_model.constrs)+1 + len(yBoundConstrs))]

    sub.linear_constraints.add(yBoundConstrs, senses=[
                               'L']*len(global_model.secondStageBinVars), rhs=[1]*len(global_model.secondStageBinVars))

    sub.parameters.lpmethod.set(2)

    turnLogOff(sub)

    return sub


def subproblem_sslp_equals(global_model: GlobalModel):
    """
    returns: subproblem for sslp problem, has fixed recourse
    """

    # sslp subproblem
    sub = cpx.Cplex()

    # add all variables

    # we add y0 variable
    lower_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageBinVars))

    # upper bound attained by solving
    upper_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageContVars))

    # add y0 variable for the objective function
    # sub.variables.add(obj  = [1.0], names = ["y0"], lb = [lower_bd])
    sub.variables.add(obj=[1.0], names=["y0"], lb=[-cpx.infinity])

    # x variables in the first stage (x variables)
    # here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
    sub.variables.add(names=global_model.original_masterVars)

    # y variables
    # sub.variables.add(ub = [1]*len(global_model.secondStageBinVars), names= global_model.secondStageBinVars)

    sub.variables.add(names=global_model.secondStageBinVars, lb=[
                      0]*len(global_model.secondStageBinVars))

    # x variables in second stage (y_{0, \omega} variables)
    sub.variables.add(names=global_model.secondStageContVars)

    # also add the linear constraint corresponding to the objective function

    ind1 = global_model.secondStageBinVars.copy()
    val1 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageBinVars)))

    ind2 = global_model.secondStageContVars.copy()
    val2 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageContVars)))

    ind0 = ["y0"]
    val0 = [1]

    sub.linear_constraints.add(lin_expr=[cpx.SparsePair(
        ind=ind0 + ind1 + ind2, val=val0 + val1 + val2)], senses=["E"], rhs=[0], names=['c0'])

    y0_index = sub.variables.get_num()-1
    sub.y0_index = y0_index

    # add linear constraints (other than the first constraint which belongs to the master problem)
    # first constraint is removed because it belongs to the master problem
    # newRows = updateConstraintRowsIndex(global_model.cpx_model.linear_constraints.get_rows(global_model.constrs[1:]))

    cons_count = len(global_model.constrs)
    # hard coding done here in the position where the index begins from
    for i in range(1, cons_count):
        cons_sense = global_model.cpx_model.linear_constraints.get_senses(i)
        cons_name = global_model.cpx_model.linear_constraints.get_names(i)

        if cons_sense != 'E':

            var_name = "s_" + cons_name
            sub.variables.add(names=["s_" + cons_name], lb=[0])
            cons_rows = [global_model.cpx_model.linear_constraints.get_rows(i)]

            if cons_sense == 'G':
                new_rows = updateConstraintRowsIndex_ver2(
                    cons_rows, var_name, -1)
            elif cons_sense == 'L':
                new_rows = updateConstraintRowsIndex_ver2(
                    cons_rows, var_name, 1)

            sub.linear_constraints.add(lin_expr=new_rows, senses=['E'], rhs=[
                                       global_model.cpx_model.linear_constraints.get_rhs(i)], names=[cons_name])

        else:
            cons_rows = [global_model.cpx_model.linear_constraints.get_rows(i)]
            new_rows = updateConstraintRowsIndex(cons_rows)
            sub.linear_constraints.add(lin_expr=new_rows, senses=['E'], rhs=[
                                       global_model.cpx_model.linear_constraints.get_rhs(i)], names=[cons_name])


    for yVar in global_model.secondStageBinVars:

        sub.variables.add(lb=[0.0], names=["s_" + yVar])
        yBoundConstr = [cpx.SparsePair(ind=[yVar, "s_" + yVar], val=[1, 1])]
        sub.linear_constraints.add(
            lin_expr=yBoundConstr, senses=['E'], rhs=[1])

    sub.parameters.lpmethod.set(2)

    turnLogOff(sub)

    return sub


def subproblem_sslp_mip(global_model: GlobalModel):
    """
    returns: subproblem for sslp problem, has fixed recourse
    """

    # sslp subproblem
    sub = cpx.Cplex()

    # we add y0 variable
    lower_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageBinVars))

    # upper bound attained by solving
    upper_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageContVars))

    # add y0 variable for the objective function
    sub.variables.add(obj=[1.0], names=["y0"], lb=[-cpx.infinity], types=['I'])

    # x variables in the first stage (x variables)
    # here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
    sub.variables.add(names=global_model.original_masterVars)

    # y variables
    sub.variables.add(names=global_model.secondStageBinVars, lb=[
                      0]*len(global_model.secondStageBinVars), types=['B']*len(global_model.secondStageBinVars))

    # x variables in second stage (y_{0, \omega} variables)

    sub.variables.add(names=global_model.secondStageContVars, types=[
                      'I']*len(global_model.secondStageContVars))

    # also add the linear constraint corresponding to the objective function

    ind1 = global_model.secondStageBinVars.copy()
    val1 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageBinVars)))

    ind2 = global_model.secondStageContVars.copy()
    val2 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageContVars)))

    ind0 = ["y0"]
    val0 = [1]

    sub.linear_constraints.add(lin_expr=[cpx.SparsePair(
        ind=ind0 + ind1 + ind2, val=val0 + val1 + val2)], senses=["E"], rhs=[0], names=['c0'])

    y0_index = sub.variables.get_num()-1
    sub.y0_index = y0_index

    # add linear constraints (other than the first constraint which belongs to the master problem)
    # first constraint is removed because it belongs to the master problem
    newRows = updateConstraintRowsIndex(
        global_model.cpx_model.linear_constraints.get_rows(global_model.constrs[1:]))

    sub.linear_constraints.add(newRows, senses=global_model.cpx_model.linear_constraints.get_senses(
        global_model.constrs[1:]), rhs=global_model.cpx_model.linear_constraints.get_rhs(global_model.constrs[1:]), names=global_model.constrs[1:])

    yBoundConstrs = [cpx.SparsePair(ind=[yVar], val=[1])
                     for yVar in global_model.secondStageBinVars]

    yBoundConstr_names = [f"c{i}" for i in range(
        len(global_model.constrs)+1, len(global_model.constrs)+1 + len(yBoundConstrs))]

    sub.linear_constraints.add(yBoundConstrs, senses=[
                               'L']*len(global_model.secondStageBinVars), rhs=[1]*len(global_model.secondStageBinVars))

    turnLogOff(sub)

    return sub


def upperBoundCalculation(sub: cpx.Cplex, global_model: GlobalModel):
    """
    computes the upper bound for the subproblem
    """

    master_vars_names = global_model.original_masterVars
    constrs = list(range(1, len(master_vars_names)+1))
    int_vars = global_model.secondStageContVars
    y_var_names = global_model.secondStageBinVars

    obj_coeffs = global_model.cpx_model.objective.get_linear(int_vars)
    bound = 0

    for i in range(len(constrs)):
        tupleList = [(constrs[i], y_name) for y_name in y_var_names]
        tupleCoeff = global_model.cpx_model.linear_constraints.get_coefficients(
            tupleList)
        x_coeff = global_model.cpx_model.linear_constraints.get_coefficients(
            constrs[i], master_vars_names[i])
        max_x = max(-np.sum(tupleCoeff) - x_coeff, 0)
        bound += max_x*obj_coeffs[i]

    return bound


def subproblem_sslp_sddip(global_model: GlobalModel):
    """
    returns: subproblem for sslp problem, has fixed recourse
    tailored for sddip
    """

    # sslp subproblem
    sub = cpx.Cplex()

    # add all variables

    # we add y0 variable
    lower_bd = np.sum(global_model.cpx_model.objective.get_linear(
        global_model.secondStageBinVars))

    # compute the upper bound we are getting from there
    upper_bd = upperBoundCalculation(sub, global_model)

    # add y0 variable for the objective function
    # sub.variables.add(obj  = [1.0], names = ["y0"], lb = [lower_bd])
    sub.variables.add(obj=[1.0], names=["y0"], lb=[-cpx.infinity])

    # x variables in the first stage (x variables)
    # here the objective is 0 because in subproblem the objective is 0 corresponding to these problems

    sub.variables.add(names=global_model.original_masterVars, ub=[
                      1]*len(global_model.original_masterVars))

    # y variables
    # sub.variables.add(ub = [1]*len(global_model.secondStageBinVars), names= global_model.secondStageBinVars)
    sub.variables.add(names=global_model.secondStageBinVars, types=[
                      'B']*len(global_model.secondStageBinVars))

    # x variables in second stage (y_{0, \omega} variables)
    sub.variables.add(names=global_model.secondStageContVars, types=[
                      'I']*len(global_model.secondStageContVars))

    # also add the linear constraint corresponding to the objective function

    ind1 = global_model.secondStageBinVars.copy()
    val1 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageBinVars)))

    ind2 = global_model.secondStageContVars.copy()
    val2 = list(-np.array(global_model.cpx_model.objective.get_linear(global_model.secondStageContVars)))

    ind0 = ["y0"]
    val0 = [1]

    sub.linear_constraints.add(lin_expr=[cpx.SparsePair(
        ind=ind0 + ind1 + ind2, val=val0 + val1 + val2)], senses=["E"], rhs=[0], names=['c0'])

    y0_index = sub.variables.get_num()-1
    sub.y0_index = y0_index

    # add linear constraints (other than the first constraint which belongs to the master problem)
    # first constraint is removed because it belongs to the master problem
    newRows = updateConstraintRowsIndex(
        global_model.cpx_model.linear_constraints.get_rows(global_model.constrs[1:]))

    sub.linear_constraints.add(newRows, senses=global_model.cpx_model.linear_constraints.get_senses(
        global_model.constrs[1:]), rhs=global_model.cpx_model.linear_constraints.get_rhs(global_model.constrs[1:]), names=global_model.constrs[1:])


    return sub


def lhs_technology_matrix(sub: cpx.Cplex, global_model: GlobalModel):
    """
    Input: sub (cplex model) that we will parameterize
    """

    # create the tech matrix
    sparseTech = sub.variables.get_cols(global_model.original_masterVars)
    sparseTechLen = len(sparseTech)

    techMatrix = np.zeros((sub.linear_constraints.get_num(), sparseTechLen))

    for j in range(sparseTechLen):
        sparseCol = sparseTech[j]
        ind, val = sparseCol.unpack()
        for i in range(len(ind)):
            rowi = ind[i]
            techMatrix[rowi, j] = val[i]

    return techMatrix


def constr_name_to_index_map(sub: cpx.Cplex, global_model: GlobalModel):
    """
    scroll through constraints to determine what name corresponds to what index
    """

    name_list = sub.linear_constraints.get_names()
    out_dict = {}

    for i in range(len(name_list)):
        out_dict[name_list[i]] = i

    return out_dict


def delete_master_vars(sub: cpx.Cplex, global_model: GlobalModel):
    """
    delete the master variables from the original problem

    """

    sub.variables.delete(global_model.original_masterVars)

    return sub


def create_all_subproblems(global_model: GlobalModel):
    """
    scenario_dict: keys are scenario indices
    values are objects of subproblem class defined in primitives.py
    """
    # later we have to update these to delete master variables from the subproblem
    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        global_model.scenario_dict[idx].cpx_model = subproblem_sslp(
            global_model)
        global_model.scenario_dict[idx].cpx_model_mip = subproblem_sslp_mip(
            global_model)
        global_model.scenario_dict[idx].constr_count = global_model.scenario_dict[idx].cpx_model.linear_constraints.get_num(
        )
        global_model.scenario_dict[idx].constr_count_mip = global_model.scenario_dict[idx].constr_count
        global_model.scenario_dict[idx].secondStageBinVars = global_model.secondStageBinVars
        global_model.scenario_dict[idx].secondStageIntVars = global_model.secondStageContVars


def create_all_subproblems_ver2(global_model: GlobalModel):
    """
    scenario_dict: keys are scenario indices
    values are objects of subproblem class defined in primitives.py
    """
    # later we have to update these to delete master variables from the subproblem
    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        global_model.scenario_dict[idx].cpx_model = subproblem_sslp_equals(
            global_model)
        global_model.scenario_dict[idx].cpx_model_mip = subproblem_sslp_mip(
            global_model)
        global_model.scenario_dict[idx].constr_count = global_model.scenario_dict[idx].cpx_model.linear_constraints.get_num(
        )
        global_model.scenario_dict[idx].constr_count_mip = global_model.scenario_dict[idx].constr_count

        global_model.scenario_dict[idx].secondStageBinVars = global_model.secondStageBinVars
        global_model.scenario_dict[idx].secondStageIntVars = global_model.secondStageContVars


def create_scenario_defs(global_model: GlobalModel, master: cpx.Cplex):
    """
    creates def specific to the scenario
    no objective function is set
    scenario def is used to determine if the gfc cut enforced is valid
    includes the x variables also
    why did we create two versions of the problems 

    """

    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        global_model.scenario_dict[idx].cpx_def = def_sub_sslp(
            global_model.scenario_dict[idx], master)
        global_model.scenario_dict[idx].cpx_def_copy = def_sub_sslp(
            global_model.scenario_dict[idx], master)


def delete_master_vars_all_sub(global_model: GlobalModel):
    """
    delete master variables from all subproblems
    """

    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        delete_master_vars(
            global_model.scenario_dict[idx].cpx_model, global_model)
        delete_master_vars(
            global_model.scenario_dict[idx].cpx_model_mip, global_model)


def rhs_technology_sslp(sub_cons_count, master_vars_count):
    """
    Input:
        cons: number of constraints in the subproblems
        vars: number of variables in the master problem

    creates the fixed technology matrix of dimension:
    number of constraints in the subproblem * number of variables in master problem
    """

    rhs_tech_row = np.zeros((sub_cons_count, master_vars_count))


def rhs_random_constant(global_model: GlobalModel, nameToIndexMap: dict):
    """
    creates a dictionary of arrays
    key: scenario id, val: array indicating the rhs in that problem
    """

    scen_ids = global_model.scenario_dict.keys()
    rhs_dict = {}

    for id in scen_ids:
        idsub = global_model.scenario_dict[id]

        rhs_array = np.zeros(idsub.cpx_model.linear_constraints.get_num())
        cnames = idsub.cpx_model.linear_constraints.get_names()
        for cname in cnames:
            idx = nameToIndexMap[cname]
            if cname in idsub.constraintMap:
                rhs_array[idx] = idsub.constraintMap[cname]

            else:
                rhs_array[idx] = idsub.cpx_model.linear_constraints.get_rhs(
                    cname)

        idsub.rhs_const = rhs_array
        idsub.rhs_const_mip = copy.deepcopy(rhs_array)


    return "updated"


def updateSub_withRhs(sub):
    """
    updates the subproblem with incumbent solution
    """
    id = sub.idx
    cons = sub.cpx_model.linear_constraints.get_num()

    sub.cpx_model.linear_constraints.set_rhs(
        [(i, sub.rhs_const[i]) for i in range(cons)])


def turnIntoLessThanConstraints(global_model: GlobalModel):
    """
    global_model (object of GlobalModel class): converts the >= constraints into less than equal to constraints
    """

    if len(global_model.sub_constr_indices) == 0:
        global_model.filterConstrs_sslp()

    for cId in global_model.sub_constr_indices:

        if global_model.cpx_model.linear_constraints.get_senses(cId) == 'G':
            ind, val = global_model.cpx_model.linear_constraints.get_rows(
                cId).unpack()
            val = list(-np.array(val))
            new = cpx.SparsePair(ind=ind, val=val)
            global_model.cpx_model.linear_constraints.set_linear_components(
                cId, new)

            old_rhs = global_model.cpx_model.linear_constraints.get_rhs(cId)
            global_model.cpx_model.linear_constraints.set_rhs(cId, -old_rhs)
            global_model.cpx_model.linear_constraints.set_senses(cId, 'L')


def main(name, isRuiInst=False, ver=1, mVer=1):
    """
    returns:
        rhs_tech_row (matrix): fixed technology matrix
        rhs_const(dictionary of arrays): random rhs constant vector
        sub_dict(dictionary of Subproblems): subproblems corresponding to each scenario
        master(cpx model): cpx model for the master problem
    """

    if not isRuiInst:
        scenario_dict, prob_vector = loadScenarios(f"{name}.sto")
    else:
        map = pk.load(open(pickle_path + name + "_map.p", 'rb'))
        prob = pk.load(open(pickle_path + name + "_prob.p", 'rb'))
        scenario_dict, prob_vector = loadScenarios_rui(map, prob)
        name = data_path + name

    global_model = GlobalModel(name, scenario_dict)

    turnIntoLessThanConstraints(global_model)



    if mVer != 1:
        master = master_generation_ver2(global_model)
    else:
        master = master_generation(global_model)

    # create all subproblems of the model
    if ver != 1:
        create_all_subproblems_ver2(global_model)  # equality constraints
    else:
        # also include inequality constraints
        create_all_subproblems(global_model)

    sub0 = global_model.scenario_dict[0].cpx_model
    global_model.scenario_dict[0].prob_vector = prob_vector

    # technology matrix on lhs
    lhs_tech_matrix = lhs_technology_matrix(sub0, global_model)

    # delete the master variable constraints from the subproblems
    delete_master_vars_all_sub(global_model)

    # update lower bound for all subproblems
    updateLowerBoundForSubproblems(global_model)

    # map for converting name to index
    nameToIndexMap = constr_name_to_index_map(sub0, global_model)

    # load rhs vector for each scenario
    rhs_random_constant(global_model, nameToIndexMap)

    scen_count = len(global_model.scenario_dict)

    for idx in range(scen_count):
        global_model.scenario_dict[idx].rhs_tech_row = copy.deepcopy(
            -lhs_tech_matrix)
        global_model.scenario_dict[idx].rhs_tech_row_mip = copy.deepcopy(
            -lhs_tech_matrix)

    return master, global_model.scenario_dict, global_model


def create_all_langrangian_subs(global_model: GlobalModel):
    """
    scenario_dict: keys are scenario indices
    values are objects of subproblem class defined in primitives.py
    """
    # later we have to update these to delete master variables from the subproblem
    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        global_model.scenario_dict[idx].cpx_model = subproblem_sslp_sddip(
            global_model)
        global_model.scenario_dict[idx].upper_bd = upperBoundCalculation(
            global_model.scenario_dict[idx], global_model)
        global_model.scenario_dict[idx].cpx_model_copy = subproblem_sslp_sddip(
            global_model)


def update_all_subs_with_rhs(global_model: GlobalModel):
    """
    update the subproblems with rhs    
    """

    # later we have to update these to delete master variables from the subproblem
    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        updateSub_withRhs(global_model.scenario_dict[idx])


def delete_master_vars_all_langr(global_model: GlobalModel):
    """
    delete master variables from all lagrangian subproblems
    """

    scen_count = len(global_model.scenario_dict)
    for idx in range(scen_count):
        delete_master_vars(
            global_model.scenario_dict[idx].cpx_model_copy, global_model)


def main_sddip(name):
    """
    """

    scenario_dict, prob_vector = loadScenarios(f"{name}.sto")
    global_model = GlobalModel(name, scenario_dict)
    master = master_generation(global_model)

    create_all_langrangian_subs(global_model)

    sub0 = global_model.scenario_dict[0].cpx_model
    sub0_copy = global_model.scenario_dict[0].cpx_model_copy

    lhs_tech_matrix = lhs_technology_matrix(sub0, global_model)


    # delete master variables from the copy version
    delete_master_vars_all_langr(global_model)


    # map for converting name to index
    nameToIndexMap = constr_name_to_index_map(sub0, global_model)

    # load rhs vector for each scenario
    rhs_random_constant(global_model, nameToIndexMap)

    update_all_subs_with_rhs(global_model)

    scen_count = len(global_model.scenario_dict)

    for idx in range(scen_count):
        global_model.scenario_dict[idx].rhs_tech_row = copy.deepcopy(
            -lhs_tech_matrix)

    return master, global_model.scenario_dict


nameList = ['sslp_5_25_50', 'sslp_5_25_100', 'sslp_10_50_50',
            'sslp_10_50_100', 'sslp_10_50_500', 'sslp_10_50_1000', 'sslp_10_50_2000']
nameList_rem = ['sslp_15_45_5', 'sslp_15_45_10', 'sslp_15_45_15']


# for name in nameList_rem:

#     print("publishing data for: ", name)

#     master, scenario_dict, global_model = main(data_path + name, isRuiInst=False)

#     Nfv, Nn, Nscen, xBound, cVec, qMtr, dMtr, q0Vec, Nu, hMtr, pVec = convertToRui(global_model)

#     pickleDump(name, Nfv, Nn, Nscen, xBound, cVec, qMtr, dMtr, q0Vec, Nu, hMtr, pVec)


# gm = GlobalModel(name = data_path + name2, scenario_dict = {})
# gm.cpx_model.write("rui_file_before.lp")


# turnIntoLessThanConstraints(gm)
# # master_gm = master_generation(gm)


# gm.cpx_model.write("rui_file.lp")


# print(f"type of master problem: {type(master)}")
# print(f"number of scenarios: {len(scenario_dict)}")
# print(f"rhs constant vector: {rhs_const}")
# print(f"technology matrix: {rhs_tech_matrix}")
