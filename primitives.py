"""
functions here do not import from user created files
"""



import cplex as cpx
import csv
import numpy as np
import math

class Subproblem:

    #subproblem and scenario means the same thing

    def __init__(self, idx):
        """
        idx: scenario id
        constraintMap (dict): constraint name and corresponding rhs (#hardcoding for sslp dataset)    
        prob (float): probability of scenario
        """

        
        self.idx = idx
        self.constraintMap = None
        self.varObjMap = None           #objective coefficient of the variable
        self.probability = None         
        self.slackVars = None
        self.coeffs = None
        self.ld_obj = 0
        self.rhs_const = None
        self.rhs_tech_row = None        #technology matrix specific to the subproblem
        self.cpx_model = None           #contains the cplex model
        self.dual = None                # contains the dual vector info used to create the cut
        self.cut = None                 # benders cut used for adding to the masters
        self.gfcs = 0                   # number of gfc cuts added to this subproblem
        self.obj_beforeGade = None      #objective function before Gade's gfc cut is applied
        self.obj_afterGade = None       #objective function after Gade's gfc cut is applied
        self.isGFCenforced = 0          #bool (0 = False, 1 = True) indicating if the gfc is enforced
        self.secondStageIntVars = None  #Continuous variables in the second stage problem (these are actually integer)
        self.secondStageBinVars = None  #Binary variables in the second stage subproblem
        self.y_ans = None               #solution to the subproblem
        self.constr_count = 0           #number of constraints in the model
        self.prob_vector  = None        #probability vector for all subproblems
        self.coeffs = None              #these are dual coefficients obtained during benders procedure
        self.rhs_constant = None        #this is benders constant obtained during the benders procedure
        self.ben_obj = None             #this is benders objective obtained during the benders procedure
        self.lower_bound = None         #lower bound of the subproblem
        self.curr_obj = None            #current objective of the sub-problem
        self.isGFCrequired = False      #bool updated in each iteration to determine if GFC is required in that iteration
        self.sols       = None          #solutions of the subproblem in each iteration
        self.prev_sols = {}         #solutions from previous solves that have been stored


    def get_cons_names(self):

        return self.cpx_model.linear_constraints.get_names()

class Master:
    
    # master class for declaring the master problem and relevant information

    def __init__(self, name, master):
        
        self.name = name
        self.master = master
        self.cpx_model = None # contains the cplex model
        self.optCuts = 0 # records the number of constraints in the Master problem
        self.xVarsInd = None
        self.pVarsInd = None
        self.xVars_ind = None   #x variables indices in the problem
        self.pVars_ind = None   #p variables indices in the problem
        self.xVars_names = None #x variables names 
        self.pVars_names = None #name of p variables in the problem




        

class GlobalModel:



    def __init__(self, name, scenario_dict):
        
        """
        #loads the data from the name.mps file 
        #contains info regarding both master and their subproblems

        """


        self.mps_name = name + ".mps"
        self.cpx_model = cpx.Cplex(self.mps_name)
        

        self.vars    = self.cpx_model.variables.get_names()
        self.constrs = self.cpx_model.linear_constraints.get_names()

        self.original_masterVars = None
        self.masterVars          = None

        self.secondStageContVars  = None
        self.secondStageBinVars   = None
        self.masterVarsLen        = None

        self.master_constr_indices  = [] #index of constraints that belong to master problems only
        self.sub_constr_indices     = [] #index of constraints that belong to the subproblems

        self.filterVars()
        self.master        = None #to store the cpx model of the master formulation
        self.scenario_dict = scenario_dict # to store the information regarding each scenario
    
    def filterVars(self): #hardcoding done here to identify the variables
    
        """
        identifies variables that belong only to the master problem
        also variables which are continuous in the second stage
        and variables with 1 as upper bound in the second stage
        """
        
        allvars = self.vars

        masterVars = list()
        secondStageCont = list()
        secondStageBin = list()

        for var in allvars:
            if var.count("x") == 1:
                if var.count("_") == 1:
                    masterVars.append(var)
                else:
                    secondStageCont.append(var)
            else:
                secondStageBin.append(var)

        #variables and their types in different stages     
        self.masterVars = masterVars.copy()
        self.original_masterVars = masterVars.copy()
        self.secondStageBinVars = secondStageBin.copy()
        self.secondStageContVars = secondStageCont.copy()
        self.masterVarsLen = len(self.masterVars)
    
    def filterConstrs_sslp(self): 
        
        """
        filter constraints for SSLP problems
        """

        constrs = self.cpx_model.linear_constraints.get_rows()
        for i in range(len(constrs)):
            constr = constrs[i]
            ind, val = constr.unpack()
            if self.areIndSecStage(ind) == True:
                self.sub_constr_indices.append(i)
            else:
                self.master_constr_indices.append(i)

    def areIndSecStage(self, ind):

        """
        determines if the indices are second stage
        """    

        flag = False
        for j in ind:
            if self.vars[j] not in self.original_masterVars:
                flag = True
                break
        
        return flag

class GlobalModel_smkp:

    """
    Data structure specific for Stochastic multi knapsack problem
    """

    def __init__(self, name, scenario_dict):
        
        """
        #loads the data from the name.mps file 
        #contains info regarding both master and their subproblems

        """


        self.mps_name = name + ".mps"
        self.cpx_model = cpx.Cplex(self.mps_name)
        
        

        self.vars    = self.cpx_model.variables.get_names()
        self.constrs = self.cpx_model.linear_constraints.get_names()

        self.original_masterVars = None
        self.masterVars          = None

        self.secondStageContVars  = None #no continuous variables in smkp problem
        self.secondStageBinVars   = None
        self.masterVarsLen        = None

        self.master_constr_indices  = [] #index of constraints that belong to master problems only
        self.sub_constr_indices     = [] #index of constraints that belong to the subproblems

        # self.filterVars()
        self.master        = None #to store the cpx model of the master formulation
        self.scenario_dict = scenario_dict # to store the information regarding each scenario

    def filterVars_smkp(self): #hardcoding done here to identify the variables
    
        """
        identifies variables that belong only to the master problem
        also variables which are continuous in the second stage
        and variables with 1 as upper bound in the second stage
        """
        
        allvars = self.vars
        masterVars = list()
        
        secondStageBin = list()

        #there are no continuous variables in smkp problem
        for var_index, var in enumerate(allvars):
            if var.count("x") == 1 or var.count("p"):           #hardcoding done over here
                if var.count("_") == 1:
                    masterVars.append(var)
            else:
                secondStageBin.append(var)


        #variables and their types in different stages     
        self.masterVars            = masterVars.copy()
        self.original_masterVars   = masterVars.copy()
        self.secondStageBinVars    = secondStageBin.copy()
        self.masterVarsLen         = len(self.masterVars)
    
    def filterConstrs_smkp(self): 
        
        """
        filter constraints for Smkp problems
        """

        constrs = self.cpx_model.linear_constraints.get_rows()
        for i in range(len(constrs)):
            constr = constrs[i]
            ind, val = constr.unpack()
            if self.areIndSecStage(ind) == True:
                self.sub_constr_indices.append(i)
            else:
                self.master_constr_indices.append(i)

    def areIndSecStage(self, ind):

        """
        determines if the indices are second stage
        """    

        flag = False
        for j in ind:
            if self.vars[j] not in self.original_masterVars:
                flag = True
                break
        
        return flag

def loadScenarios(filename):
    """
    returns: 
    tailormade for sslp instances
    scenario_dict (dict): keys are scenario ids
    values: objects of Subproblem class

    Subproblem class essentially represents the scenarios with following attributes:
        probability
        constraintMap (dict): keys are constraint ids and values are rhs value (float)

    """

    scenario_dict = {}
    with open(filename, "r") as f:
        data = f.readlines()
    
    idx = 0
    for line in data:
        words = line.split()
        if len(words) > 2:
            if words[0] == "SC":
                scen = Subproblem(idx)

                scenario_dict[idx] = scen
                scen.probability = float(words[3])
                scen.constraintMap = {}
                idx += 1
            elif words[0] == "RHS":
                scen.constraintMap[words[1]] = float(words[2])

    
    return scenario_dict

def count_cuts(master):
    
    total = 0
    search = list(range(11)) + [20]
    
    for i in search:
        total += master.solution.MIP.get_num_cuts(i)

    return total

def turnLogOff(master):

    #master problem (or cpx problem) without log
    master.set_log_stream(None)
    master.set_error_stream(None)
    master.set_warning_stream(None)
    master.set_results_stream(None)

def recordCSV(out_path, csv_name, fieldnames, mode = "w+"):

    with open(out_path + csv_name, mode = mode) as f:

        writer = csv.writer(f)
        writer.writerow(fieldnames)

def getsolution(cpx_model):

    """

    returns: solution from the master problem (cpx_model)
    x (numpy array of x values)
    t (scalar) cost to go approximation
             
    """


    sols = cpx_model.solution.get_values()
    x  = np.array(sols[:-1])
    t = sols[-1]

    return x, t

def relax_integer_problem(master):

    """
    relax the master (cplex model) problem (or any cplex integer problem)
    """

    varN = master.variables.get_num()
    varTypes = master.variables.get_types()
    changeInd = []
    boundInd = []

    binaryInd = []
    integerInd = []

    for i in range(varN):
        if varTypes[i] == 'B':
            binaryInd.append(i)
            changeInd.append((i, 'C'))
            boundInd.append((i, 1))
        elif varTypes[i] == 'I':
            changeInd.append((i, 'C'))
            integerInd.append(i)

    
    master.variables.set_types(changeInd)
    master.variables.set_upper_bounds(boundInd)

    return binaryInd, integerInd

def revert_back_to_mip(master, binaryInd, integerInd):

    """
    makes the master problem mip on specified indices
    """  

    typeList = [(i, 'B') for i in binaryInd] + [(i, 'I') for i in integerInd]
    if len(binaryInd) > 0 or len(integerInd) > 0:
        master.variables.set_types(typeList)

def isImprovementSignificant(lb_list, ub_list, count, impr_gap = 1e-2, impr_lb = 0.05, past_iter = 50, tolerance = 1e-6):

    """
    lb_list: list of lower bounds in successive iterations
    ub_list: list of upper bounds in successive iterations

    if no significant improvement in lower bound or gap then solve mips and terminate
    """

    if lb_list[-1] == float('-inf') or ub_list[-1] == float('inf'):
        return False
    
    gap_now = abs(ub_list[-1] - lb_list[-1])/(tolerance + abs(ub_list[-1]))
    gap_after = abs(ub_list[-past_iter] - lb_list[-past_iter])/(tolerance + abs(ub_list[-past_iter]))
    
    if abs(gap_now - gap_after) < impr_gap and abs(lb_list[-1] - lb_list[-past_iter])/abs(lb_list[-past_iter]) < impr_lb:
        return False

    return True

def isFractional(val, tolerance = 1e-4):
    
    val_ = math.ceil(val)
    _val = math.floor(val)

    if val < val_ - tolerance and val > _val + tolerance:
        return True


    return False

def isFractionalSol(val_list):

    """
    determines if the solution ve
    """

    flag = False
    for val in val_list:
        if isFractional(val, tolerance=1e-6):
            flag = True
            break
    
    return flag

def source_row(in_list, tolerance = 1e-4):
    """
    in_list: list with indices representing structural variables and values representing whether they are in the basis

    in_list: corresponding to the variable that takes fractional value and is also in basis we pass the column of that variable
    in basis inverse, clearly such a coloumn will have only one 1 and rest zero enteries.

    computes the source row, or the first row in which entry is one
    """


    for i in range(len(in_list)):
        if abs(in_list[i]- 1) < tolerance:
            return i

    raise ValueError("Not a basis coloumn")

def gades_transformation_step(fsv, x, rhs_const_gfc, rhs_x, tolerance = 1e-6):

    """
    performs the gades transformation step so the current solution becomes non-basic
    """

    rhs_x_new = np.zeros(fsv)


    for i in range(fsv):
        if abs(x[i] - 1) < tolerance:
            rhs_x_new[i] =  -rhs_x[i]

    return rhs_const_gfc + np.dot(x, rhs_x), rhs_x_new

def gfc_step(str_aMatrix_source_row, slack_row, rhs_const_gfc, rhs_x, x, fsv):

    """
    str_aMatrix_source_row: source row corresponding to structural variables
    slack_row: source row corresponding to initial slack variables, slacks corresponding to equality constraints are not included
    rhs_const_gfc: right hand side constant determined for computing gfc cut
    rhs_x: right hand side x coefficient values
    x: incumbent solution

    returns:
        Determines Chavatal inequality to get Gomory's fractional cut
        
    """

    out_aMatrix_row = np.floor(str_aMatrix_source_row)
    out_slack_row = np.floor(slack_row)
    out_rhs_x = -np.floor(-rhs_x)
    out_const = np.floor(rhs_const_gfc)
    out_const, out_rhs_x = gades_transformation_step(fsv, x, out_const, out_rhs_x)

    return out_aMatrix_row, out_slack_row, out_const, out_rhs_x

def gfc_step_ver2(str_aMatrix_source_row, slack_row_fsv, slack_row_fsv_ineq, rhs_const_gfc, rhs_x, x, fsv):

    """
    str_aMatrix_source_row: source row corresponding to structural variables
    slack_row: source row corresponding to initial slack variables, slacks corresponding to equality constraints are not included
    rhs_const_gfc: right hand side constant determined for computing gfc cut
    rhs_x: right hand side x coefficient values
    x: incumbent solution

    returns:
        Determines Chavatal inequality to get Gomory's fractional cut
        
    """

    out_aMatrix_row        = np.floor(str_aMatrix_source_row)
    out_slack_row_fsv      = np.floor(slack_row_fsv)
    out_slack_row_fsv_ineq = np.floor(slack_row_fsv_ineq)
    out_rhs_x              = -np.floor(-rhs_x)
    out_const              = np.floor(rhs_const_gfc)
    out_const, out_rhs_x   = gades_transformation_step(fsv, x, out_const, out_rhs_x)


    return out_aMatrix_row, out_slack_row_fsv, out_slack_row_fsv_ineq, out_const, out_rhs_x

