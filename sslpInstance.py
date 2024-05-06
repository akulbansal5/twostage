"""

SSLP: Stochastic server location problems
new version of sslpInstance.py

"""


import os
import copy
import pickle as pk                 
import cplex as cpx                                 
import numpy as np                 
import gurobipy as gp
from primitive import Primitive    


dir_path    = os.path.dirname(os.path.realpath(__file__))
data_path   = dir_path + "/stofiles/"
lp_path     = dir_path + "/lpfiles/"
out_path    = dir_path + "/output/"
pickle_path = dir_path + "/pk_files/"

# master = masterObj.cpx_model
# t = masterObj.get_solution()
# global_model.updateSubproblems_withIncmbt() 

class GlobalModelSSLP():

    def __init__(self, datapath, name):
        
        """
        #loads the data from the name.mps file 
        #contains info regarding both master and their subproblems
        """

        # super().__init__(name)
        self.name = name
        self.datapath = datapath
        self.mps_name   = name + ".mps"
        self.cpx_model  = cpx.Cplex(datapath + self.mps_name)
        # self.cpx_model.write(lp_path + f"{name}_original.lp")
        self.vars    = self.cpx_model.variables.get_names()
        self.constrs = self.cpx_model.linear_constraints.get_names()
        self.original_masterVars    = None
        self.masterVars             = None
        self.secondStageContVars    = None
        self.secondStageBinVars     = None
        self.masterVarsLen          = None
        self.master_constr_indices  = []    #index of constraints that belong to master problems only
        self.sub_constr_indices     = []    #index of constraints that belong to the subproblems
        self.filterVars()
        self.masterObj              = None   #to store object of Master class
        self.scenario_dict          = None      # to store the information regarding each scenario
        self.sub_idxs               = None      #subproblem ids
        self.Nscen                  = 0
        self.scenProb               = None #probability vector indication probability of each scenario
        self.Ncopy                  = 0 #number of copy variables
        self.ben_cuts = 0
        self.gfc_cuts = 0
        self.iters = 0
        self.LB  = None
        self.UB = None
        self.gap = None

    def filterVars(self):
    
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
        self.secondStageBinVars  = secondStageBin.copy()
        self.secondStageContVars = secondStageCont.copy()
        self.masterVarsLen       = len(self.masterVars)
    
    def filterConstrs(self): 
        
        """
        filter constraints for SSLP problems into two groups - master constraints and second stage subproblem constraints
        """

        constrs = self.cpx_model.linear_constraints.get_rows()
        for i in range(len(constrs)):
            constr = constrs[i]
            ind, val = constr.unpack()
            if self.areIndSecStage(ind) == True:
                self.sub_constr_indices.append(i)
            else:
                self.master_constr_indices.append(i)

    def areIndSecStage(self, indices):

        """
        indices (list of ints): variable indices
        determines if the indices are second stage
        """    

        flag = False
        for j in indices:
            if self.vars[j] not in self.original_masterVars:
                flag = True
                break
        
        return flag

    def loadScenarios(self, filename):
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
        
        prob = []
        idx  = 0
        for line in data:
            words = line.split()
            if len(words) > 2:
                if words[0] == "SC":
                    scen = Subproblem(idx)

                    #we associate an idx to it
                    scenario_dict[idx] = scen
                    scen.probability = float(words[3])
                    prob.append(scen.probability)
                    scen.constraintMap = {}
                    idx += 1
                elif words[0] == "RHS":
                    scen.constraintMap[words[1]] = float(words[2])

        self.scenario_dict = scenario_dict
        self.sub_idxs = list(scenario_dict.keys())
        self.Nscen = idx

        
        return scenario_dict, np.array(prob)

    def loadScenarios_rui(self, map, prob):
        """
        returns: 
        
        scenario_dict (dict): keys are scenario ids
        values: objects of Subproblem class

        Subproblem class essentially represents the scenarios with following attributes:
            probability
            constraintMap (dict): keys are constraint ids and values are rhs value (float)

        """
        scenario_dict = {}
        Nscen = len(map)
        
        for idx in range(Nscen):
            scen = Subproblem(idx)
            scenario_dict[idx] = scen
            scen.probability = prob[idx]
            scen.constraintMap = copy.deepcopy(map[idx])

        self.scenario_dict = scenario_dict
        self.sub_idxs = list(scenario_dict.keys())
        self.Nscen = Nscen
        self.scenProb = np.array(prob)

        return scenario_dict, np.array(prob)    

    def turnIntoLessThanConstraints(self):

        """
        global_model (object of GlobalModel class): converts the >= constraints into less than equal to constraints
        this is because our implementation assumes positive slacks
        """
        
        constr_num = self.cpx_model.linear_constraints.get_num()
        constr_senses = self.cpx_model.linear_constraints.get_senses()
        constr_rows = self.cpx_model.linear_constraints.get_rows()
        constr_rhs  = self.cpx_model.linear_constraints.get_rhs()

        for cId in range(constr_num):

            if constr_senses[cId] == 'G':
                ind, val = constr_rows[cId].unpack()
                val      = list(-np.array(val))
                new = cpx.SparsePair(ind = ind, val = val)
                self.cpx_model.linear_constraints.set_linear_components(cId, new)

                old_rhs = constr_rhs[cId]
                self.cpx_model.linear_constraints.set_rhs(cId, -old_rhs)
                self.cpx_model.linear_constraints.set_senses(cId, 'L')

    def turnLogOff(self, cpx_model):

        #master problem (or cpx problem) without log
        cpx_model.set_log_stream(None)
        cpx_model.set_error_stream(None)
        cpx_model.set_warning_stream(None)
        cpx_model.set_results_stream(None)

    def updateSubproblems_withIncmbt(self, isMIP = False):

        """
        update the subproblem with incumbent solution
        sols_in_stage2: numpy array of master solution that affects the subproblem 
        """

        sols_in_stage2 = self.masterObj.sols_in_stage2

        for id in self.sub_idxs:
            sub = self.scenario_dict[id]
            if isMIP == True:
                rhs_info = list(zip(range(sub.constr_count_mip), sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2)))
                sub.cpx_model_mip.linear_constraints.set_rhs(rhs_info)
            else:
                rhs_info = list(zip(range(sub.constr_count), sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2)))
                sub.cpx_model.linear_constraints.set_rhs(rhs_info)

    def updateSubproblems_withIncmbt_grb(self, isMIP = False):

        """
        update the subproblem with incumbent solution
        sols_in_stage2: numpy array of master solution that affects the subproblem 

        tailored specific for gurobi solver
        """

        sols_in_stage2 = self.masterObj.sols_in_stage2

        for id in self.sub_idxs:
            sub = self.scenario_dict[id]
            if isMIP == True:
                # rhs_info = list(zip(range(sub.constr_count_mip), sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2)))
                rhs_info = list(sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2))
                try:
                    sub.grb_model_mip.setAttr("RHS", sub.grb_model_mip_constrs, rhs_info)
                except:
                    sub.grb_model_mip.write(lp_path + f"mip_error_{id}.lp")
                    raise ValueError("Unable to update RHS in mip-subproblem")
                # sub.cpx_model_mip.linear_constraints.set_rhs(rhs_info)
                
            else:
                # rhs_info = list(zip(range(sub.constr_count), sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2)))
                rhs_info = list(sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2))
                # sub.cpx_model.linear_constraints.set_rhs(rhs_info)
                sub.grb_model.setAttr("RHS", sub.grb_model_constrs, rhs_info)

    def updateSubproblems_withIncmbt_grb_callback(self, xVec, isMIP = False):

        """
        update the subproblem with incumbent solution
        sols_in_stage2: numpy array of master solution that affects the subproblem 

        tailored specific for gurobi solver
        _callback: works inside callbacks in gurobi
        """

        for id in self.sub_idxs:
            sub = self.scenario_dict[id]
            if isMIP == True:
                # rhs_info = list(zip(range(sub.constr_count_mip), sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2)))
                rhs_info = list(sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, xVec))
                try:
                    sub.grb_model_mip.setAttr("RHS", sub.grb_model_mip_constrs, rhs_info)
                except:
                    sub.grb_model_mip.write(lp_path + f"mip_error_{id}.lp")
                    raise ValueError(f"Unable to update RHS in mip-subproblem {len(sub.grb_model_mip_constrs)}={len(rhs_info)}")
                # sub.cpx_model_mip.linear_constraints.set_rhs(rhs_info)
                
            else:
                # rhs_info = list(zip(range(sub.constr_count), sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2)))
                rhs_info = list(sub.rhs_const + np.dot(sub.rhs_tech_row, xVec))
                # sub.cpx_model.linear_constraints.set_rhs(rhs_info)
                sub.grb_model.setAttr("RHS", sub.grb_model_constrs, rhs_info)

    def updateSub_withIncmbt_grb(self, id, isMIP = False):

        """
        updates only a single subproblem with the incumbent solution


        update the subproblem with incumbent solution
        sols_in_stage2: numpy array of master solution that affects the subproblem 

        tailored specific for gurobi solver
        """

        sols_in_stage2 = self.masterObj.sols_in_stage2


        sub = self.scenario_dict[id]

        if isMIP == True:
            # rhs_info = list(zip(range(sub.constr_count_mip), sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2)))
            rhs_info = list(sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2))
            try:
                sub.grb_model_mip.setAttr("RHS", sub.grb_model_mip_constrs, rhs_info)
            except:
                sub.grb_model_mip.write(lp_path + f"mip_error_{id}.lp")
                raise ValueError(f"Unable to update RHS in mip-subproblem {len(sub.grb_model_mip_constrs)}={len(rhs_info)}")

            # sub.cpx_model_mip.linear_constraints.set_rhs(rhs_info)
            
        else:
            # rhs_info = list(zip(range(sub.constr_count), sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2)))
            rhs_info = list(sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2))
            # sub.cpx_model.linear_constraints.set_rhs(rhs_info)
            sub.grb_model.setAttr("RHS", sub.grb_model_constrs, rhs_info)

    def updateSub_withIncmbt_grb_callback(self, xVec, id, isMIP = False):

        """
        updates only a single subproblem with the incumbent solution


        update the subproblem with incumbent solution
        sols_in_stage2: numpy array of master solution that affects the subproblem 

        tailored specific for gurobi solver
        """

        sols_in_stage2 = xVec


        sub = self.scenario_dict[id]

        if isMIP == True:
            # rhs_info = list(zip(range(sub.constr_count_mip), sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2)))
            rhs_info = list(sub.rhs_const_mip + np.dot(sub.rhs_tech_row_mip, sols_in_stage2))
            try:
                sub.grb_model_mip.setAttr("RHS", sub.grb_model_mip_constrs, rhs_info)
            except:
                sub.grb_model_mip.write(lp_path + f"mip_error_{id}.lp")
                raise ValueError("Unable to update RHS in mip-subproblem")

            # sub.cpx_model_mip.linear_constraints.set_rhs(rhs_info)
            
        else:
            # rhs_info = list(zip(range(sub.constr_count), sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2)))
            rhs_info = list(sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2))
            # sub.cpx_model.linear_constraints.set_rhs(rhs_info)
            sub.grb_model.setAttr("RHS", sub.grb_model_constrs, rhs_info)

    def solve_all_subproblems_as_mip(self, ones):

        """
        solve all subproblems as mixed integer program and get their objective value
        """

        local_obj = 0
        laporte_coeff = 0
        laporte_rhs   = 0

        for id in self.sub_idxs:
            sub = self.scenario_dict[id]
            # new_type_pairs1 = [(i,'B') for i in sub.secondStageBinVars]
            # new_type_pairs2 = [(i,'I') for i in sub.secondStageIntVars]
            # sub.cpx_model_mip.variables.set_types(new_type_pairs1 + new_type_pairs2)
            # self.turnLogOff(sub.cpx_model_mip)
            sub.cpx_model_mip.solve()
            sub_obj = sub.cpx_model_mip.solution.get_objective_value()
            local_obj += sub.probability*sub_obj
            coeff = sub_obj - sub.lower_bound
            rhs = sub_obj- coeff*ones
            laporte_coeff += sub.probability*coeff
            laporte_rhs += sub.probability*rhs

        
        return local_obj, laporte_coeff, laporte_rhs

    def compute_future_cost(self, xVec):

        updateSubproblems_withIncmbt_grb_callback

    def master_generation(self):

        """
        creates a Master object which contains the cplex model for the master problem as well
        adds the variable t (or theta) that captures the cost in future stages
        returns : the master problem (cplex model)
        """

        master = cpx.Cplex()
        master.variables.add(obj = self.cpx_model.objective.get_linear(self.masterVars), types = ["B"]*self.masterVarsLen, names = self.masterVars)
        
        #only one constraint appears in the master problem
        master.linear_constraints.add(lin_expr = [self.cpx_model.linear_constraints.get_rows("c1")], senses = [self.cpx_model.linear_constraints.get_senses("c1")], rhs = [self.cpx_model.linear_constraints.get_rhs("c1")], names = ["c1"]) 
        
        #bound for SSLP problems and how to take it further
        t_lb = np.sum(self.cpx_model.objective.get_linear(self.secondStageBinVars))
        master.variables.add(obj = [1.0], lb = [t_lb], types = ["C"], names = ['t'])
        
        self.masterVars.append('t')
        self.masterVarsLen += 1
        self.turnLogOff(master)

        #master problem is declared over here
        self.masterObj = Master(self.name, master)

    def master_generation_multicut(self):

        """
        returns : the master problem (cplex model)
        the multicut version of the master problem
        """

        master = cpx.Cplex()
        master.variables.add(obj = self.cpx_model.objective.get_linear(self.masterVars), types = ["B"]*self.masterVarsLen, names = self.masterVars)
        
        #only one constraint appears in the master problem
        master.linear_constraints.add(lin_expr = [self.cpx_model.linear_constraints.get_rows("c1")], senses = [self.cpx_model.linear_constraints.get_senses("c1")], rhs = [self.cpx_model.linear_constraints.get_rhs("c1")], names = ["c1"]) 
        
        #bound for SSLP problems and how to take it further
        t_lb = np.sum(self.cpx_model.objective.get_linear(self.secondStageBinVars))

        # futureCostVarNames = [f't{i}' for i in range(self.Nscen)]
        
        for id in self.sub_idxs:
            master.variables.add(obj = [self.scenProb[id]], lb = [t_lb], types = ["C"], names = [f't{id}'])
            self.masterVars.append(f't{id}')


        # self.masterVars.append('t')
        self.masterVarsLen += self.Nscen
        self.turnLogOff(master)

        #master problem is declared over here
        self.masterObj = Master(self.name, master, self.Nscen)
         
    def updateConstraintRowsIndex_ver2(self, rows, var_name, coeff):

        """
        update constraints of the problem
        """

        constrs = []
        for sparseObj in rows:


            ind, val = sparseObj.unpack()
            # print(ind)
            
            new_ind_arr = np.array(ind) + np.ones(len(ind))

            new_ind = [int(i) for i in new_ind_arr]
            
            val = list(val)
            
            # print(new_ind)
            # print([type(i) for i in new_ind])

            new_ind.append(var_name)
            val.append(coeff)
            constrs.append(cpx.SparsePair(ind = new_ind, val  = val))
        
        # print(constrs)
        return constrs

    def updateConstraintRowsIndex(self, rows):

        constrs = []
        for sparseObj in rows:


            ind, val = sparseObj.unpack()
            # print(ind)
            new_ind = [int(i) for i in np.array(ind) + np.ones(len(ind))]
            # print(new_ind)
            # print([type(i) for i in new_ind])
            constrs.append(cpx.SparsePair(ind = new_ind, val  = val))
        
        # print(constrs)
        return constrs

    def subproblem_equals(self, addCopy = False):


        """
        addCopy (bool): If True then we add copy constraints to the problem
        


        returns: subproblem for sslp problem, has fixed recourse


        WARNING: FOR GOMOROY FRACTIONAL CUTS USING GADE'S APPROACH THE SUBPROBLEMS SHOULD BE SOLVED USING DUAL SIMPLEX
                 IN CURRENT IMPLEMENTATION WE SOLVE THE SUBPROBLEMS USING THE DEFAULT APPROACH

        
        
        """

        #sslp subproblem 
        sub = cpx.Cplex()
    
        # add y0 variable for the objective function
        # sub.variables.add(obj  = [1.0], names = ["y0"], lb = [lower_bd])
        sub.variables.add(obj  = [1.0], names = ["y0"], lb = [-cpx.infinity])
        #default values for lb are 0

        #x variables in the first stage (x variables)
        #here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
        # if addCopy:
        #     sub.variables.add(ub = [1]*len(self.original_masterVars), names = self.original_masterVars)
        # else:
        sub.variables.add(names = self.original_masterVars)

        #y variables
        #upper bound of 1 is added as constraint
        sub.variables.add(names= self.secondStageBinVars , lb = [0]*len(self.secondStageBinVars))

        #x variables in second stage (y_{0, \omega} variables)
        sub.variables.add(names= self.secondStageContVars)

        #also add the linear constraint corresponding to the objective function
        ind1 = self.secondStageBinVars.copy()
        val1 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageBinVars)))

        ind2 = self.secondStageContVars.copy()
        val2 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageContVars)))

        ind0 = ["y0"]
        val0 = [1]

        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind0 + ind1 + ind2, val = val0 + val1 + val2)], senses = ["E"], rhs = [0], names = ['c0'])
        
        y0_index = sub.variables.get_num()-1
        sub.y0_index = y0_index


        #add linear constraints (other than the first constraint which belongs to the master problem)
        #first constraint is removed because it belongs to the master problem
        # newRows = updateConstraintRowsIndex(self.cpx_model.linear_constraints.get_rows(self.constrs[1:]))

        
        cons_count = len(self.constrs)
        for i in range(1, cons_count): #hard coding done here in the position where the index begins from
            cons_sense = self.cpx_model.linear_constraints.get_senses(i)
            cons_name = self.cpx_model.linear_constraints.get_names(i)

            if  cons_sense != 'E':

                var_name = "s_" + cons_name
                sub.variables.add(names = ["s_" + cons_name], lb = [0])
                cons_rows = [self.cpx_model.linear_constraints.get_rows(i)]
                
                if cons_sense == 'G':
                    new_rows = self.updateConstraintRowsIndex_ver2(cons_rows, var_name, -1)       
                elif cons_sense == 'L':
                    new_rows = self.updateConstraintRowsIndex_ver2(cons_rows, var_name, 1)
                
                sub.linear_constraints.add(lin_expr = new_rows, senses = ['E'], rhs = [self.cpx_model.linear_constraints.get_rhs(i)], names = [cons_name])
            
            else:
                cons_rows = [self.cpx_model.linear_constraints.get_rows(i)]
                new_rows = self.updateConstraintRowsIndex(cons_rows)
                sub.linear_constraints.add(lin_expr = new_rows, senses = ['E'], rhs = [self.cpx_model.linear_constraints.get_rhs(i)], names = [cons_name])

        
        # #ADD UPPER BOUND ON STATE VARIABLES FROM THE PREVIOUS STAGE
        # if addCopy:
        #     for i in range(len(self.original_masterVars)):
        #         copy_ind = [self.original_masterVars[i]]
        #         copy_val = [1]
        #         copy_rhs = [1]

        #         #upper bound on copy variables
        #         sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['L'], rhs = copy_rhs, names = [f"cbd{i}"])

        
        #ADD UPPER BOUNDS ON STATE VARIABLES FROM THE PREVIOUS STAGE AS EQUALITY CONSTRAINTS
        if addCopy:
            for i in range(len(self.original_masterVars)):
                sub.variables.add(names = [f"sc_{i}"], lb = [0])
                copy_ind                = [self.original_masterVars[i], f"sc_{i}"]
                copy_val                = [1, 1]
                copy_rhs                = [1]

                #upper bound on copy variables
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['E'], rhs = copy_rhs, names = [f"cbd{i}"])



        #ADD THE COPY CONSTRAINTS
        if addCopy:
            for i in range(len(self.original_masterVars)):
                copy_ind = [self.original_masterVars[i]]
                copy_val = [1]
                copy_rhs = [0]
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['E'], rhs = copy_rhs, names = [f"copy{i}"])
        
        #the number of copy variables
        self.Ncopy = len(self.original_masterVars)
        
        

        # sub.parameters.lpmethod.set(2)
        self.turnLogOff(sub)

        return sub

    def subproblem_nonequals(self, addCopy = False):

        """
        returns: subproblem for sslp problem, has fixed recourse
        """

        #sslp subproblem 
        sub = cpx.Cplex()
        
        #add all variables

        #we add y0 variable
        lower_bd = np.sum(self.cpx_model.objective.get_linear(self.secondStageBinVars))

        #upper bound attained by solving 
        upper_bd = np.sum(self.cpx_model.objective.get_linear(self.secondStageContVars))


        # add y0 variable for the objective function
        # sub.variables.add(obj  = [1.0], names = ["y0"], lb = [lower_bd])
        sub.variables.add(obj  = [1.0], names = ["y0"], lb = [-cpx.infinity])

        #x variables in the first stage (x variables)
        #here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
        sub.variables.add(names = self.original_masterVars)

        #y variables
        # sub.variables.add(ub = [1]*len(self.secondStageBinVars), names= self.secondStageBinVars)
        
        sub.variables.add(names= self.secondStageBinVars , lb = [0]*len(self.secondStageBinVars))

        #x variables in second stage (y_{0, \omega} variables)
        sub.variables.add(names= self.secondStageContVars)

        #also add the linear constraint corresponding to the objective function
        

        ind1 = self.secondStageBinVars.copy()
        val1 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageBinVars)))

        ind2 = self.secondStageContVars.copy()
        val2 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageContVars)))

        ind0 = ["y0"]
        val0 = [1]

        #here the constraints are added by variable names
        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind0 + ind1 + ind2, val = val0 + val1 + val2)], senses = ["E"], rhs = [0], names = ['c0'])
        
        y0_index = sub.variables.get_num()-1
        sub.y0_index = y0_index

        #add linear constraints (other than the first constraint which belongs to the master problem)
        #first constraint is removed because it belongs to the master problem
        newRows = self.updateConstraintRowsIndex(self.cpx_model.linear_constraints.get_rows(self.constrs[1:]))

        sub.linear_constraints.add(newRows, senses = self.cpx_model.linear_constraints.get_senses(self.constrs[1:]), rhs = self.cpx_model.linear_constraints.get_rhs(self.constrs[1:]), names = self.constrs[1:])
        

        
        #ADD UPPER BOUND ON STATE VARIABLES FROM THE PREVIOUS STAGE
        if addCopy:
            for i in range(len(self.original_masterVars)):
                copy_ind = [self.original_masterVars[i]]
                copy_val = [1]
                copy_rhs = [1]

                #upper bound on copy variables
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['L'], rhs = copy_rhs, names = [f"cbd{i}"])

        #ADD THE COPY CONSTRAINTS
        if addCopy:
            for i in range(len(self.original_masterVars)):
                copy_ind = [self.original_masterVars[i]]
                copy_val = [1]
                copy_rhs = [0]
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['E'], rhs = copy_rhs, names = [f"copy{i}"])
        


        # yBoundConstrs = [cpx.SparsePair(ind = [yVar], val = [1]) for yVar in self.secondStageBinVars]
        
        # yBoundConstr_names = [f"c{i}" for i in range(len(self.constrs)+1, len(self.constrs)+1 + len(yBoundConstrs))]
        

        #even if these constraints are not added the variables will be less than 1 due to other constraints
        # sub.linear_constraints.add(yBoundConstrs, senses = ['L']*len(self.secondStageBinVars), rhs = [1]*len(self.secondStageBinVars))

        # sub.parameters.lpmethod.set(2)

        self.turnLogOff(sub)

        return sub

    def subproblem_mip(self, addCopy = False):

        """
        returns: subproblem for sslp problem, has fixed recourse
        """

        #sslp subproblem
        sub = cpx.Cplex()
        
        #add all variables

        #we add y0 variable
        lower_bd = np.sum(self.cpx_model.objective.get_linear(self.secondStageBinVars))


        #upper bound attained by solving 
        upper_bd = np.sum(self.cpx_model.objective.get_linear(self.secondStageContVars))


        # add y0 variable for the objective function
        # sub.variables.add(obj  = [1.0], names = ["y0"], lb = [lower_bd])
        sub.variables.add(obj  = [1.0], names = ["y0"], lb = [-cpx.infinity], types = ['C'])

        #x variables in the first stage (x variables)
        #here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
        sub.variables.add(names = self.original_masterVars)

        #y variables
        # sub.variables.add(ub = [1]*len(self.secondStageBinVars), names= self.secondStageBinVars)
        
        sub.variables.add(names= self.secondStageBinVars , lb = [0]*len(self.secondStageBinVars), types = ['B']*len(self.secondStageBinVars))

        #x variables in second stage (y_{0, \omega} variables)
        
        # types = ['C']*len(self.secondStageContVars)
        sub.variables.add(names= self.secondStageContVars)

        #also add the linear constraint corresponding to the objective function
        

        ind1 = self.secondStageBinVars.copy()
        val1 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageBinVars)))

        ind2 = self.secondStageContVars.copy()
        val2 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageContVars)))

        ind0 = ["y0"]
        val0 = [1]

        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind0 + ind1 + ind2, val = val0 + val1 + val2)], senses = ["E"], rhs = [0], names = ['c0'])
        
        y0_index = sub.variables.get_num()-1
        sub.y0_index = y0_index


        #add linear constraints (other than the first constraint which belongs to the master problem)
        #first constraint is removed because it belongs to the master problem
        newRows = self.updateConstraintRowsIndex(self.cpx_model.linear_constraints.get_rows(self.constrs[1:]))

        sub.linear_constraints.add(newRows, senses = self.cpx_model.linear_constraints.get_senses(self.constrs[1:]), rhs = self.cpx_model.linear_constraints.get_rhs(self.constrs[1:]), names = self.constrs[1:])

        #ADD UPPER BOUND ON STATE VARIABLES FROM THE PREVIOUS STAGE
        if addCopy:
            for i in range(len(self.original_masterVars)):
                copy_ind = [self.original_masterVars[i]]
                copy_val = [1]
                copy_rhs = [1]

                #upper bound on copy variables
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['L'], rhs = copy_rhs, names = [f"cbd{i}"])

        #ADD THE COPY CONSTRAINTS
        if addCopy:
            for i in range(len(self.original_masterVars)):
                copy_ind = [self.original_masterVars[i]]
                copy_val = [1]
                copy_rhs = [0]
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['E'], rhs = copy_rhs, names = [f"copy{i}"])
        
        # yBoundConstrs = [cpx.SparsePair(ind = [yVar], val = [1]) for yVar in self.secondStageBinVars]
        
        # yBoundConstr_names = [f"c{i}" for i in range(len(self.constrs)+1, len(self.constrs)+1 + len(yBoundConstrs))]
        
        # sub.linear_constraints.add(yBoundConstrs, senses = ['L']*len(self.secondStageBinVars), rhs = [1]*len(self.secondStageBinVars))

        # sub.parameters.lpmethod.set(2)

        self.turnLogOff(sub)

        return sub

    def subproblem_lagrn(self, addCopy = True):
        """
        creates Lagrangian subproblem in which the copy constraints are relaxed
        """

        """
        returns: subproblem for sslp problem, has fixed recourse
        """

        #sslp subproblem
        sub = cpx.Cplex()

        # add y0 variable for the objective function
        # sub.variables.add(obj  = [1.0], names = ["y0"], lb = [lower_bd])
        sub.variables.add(obj  = [1.0], names = ["y0"], lb = [-cpx.infinity], types = ['C'])

        #x variables in the first stage (x variables)
        #here the objective is 0 because in subproblem the objective is 0 corresponding to these problems
        sub.variables.add(names = self.original_masterVars, types = ['B']*len(self.original_masterVars))

        #y variables
        # sub.variables.add(ub = [1]*len(self.secondStageBinVars), names= self.secondStageBinVars)
        
        sub.variables.add(names= self.secondStageBinVars , lb = [0]*len(self.secondStageBinVars), types = ['B']*len(self.secondStageBinVars))

        #x variables in second stage (y_{0, \omega} variables)
        
        # types = ['I']*len(self.secondStageContVars)
        sub.variables.add(names= self.secondStageContVars)

        #also add the linear constraint corresponding to the objective function
        

        ind1 = self.secondStageBinVars.copy()
        val1 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageBinVars)))

        ind2 = self.secondStageContVars.copy()
        val2 = list(-np.array(self.cpx_model.objective.get_linear(self.secondStageContVars)))

        ind0 = ["y0"]
        val0 = [1]

        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind0 + ind1 + ind2, val = val0 + val1 + val2)], senses = ["E"], rhs = [0], names = ['c0'])
        
        y0_index = sub.variables.get_num()-1
        sub.y0_index = y0_index


        #add linear constraints (other than the first constraint which belongs to the master problem)
        #first constraint is removed because it belongs to the master problem
        
        newRows = self.updateConstraintRowsIndex(self.cpx_model.linear_constraints.get_rows(self.constrs[1:]))

        sub.linear_constraints.add(newRows, senses = self.cpx_model.linear_constraints.get_senses(self.constrs[1:]), rhs = self.cpx_model.linear_constraints.get_rhs(self.constrs[1:]), names = self.constrs[1:])

        #ADD UPPER BOUND ON STATE VARIABLES FROM THE PREVIOUS STAGE
        if addCopy:
            for i in range(len(self.original_masterVars)):
                copy_ind = [self.original_masterVars[i]]
                copy_val = [1]
                copy_rhs = [1]

                #upper bound on copy variables
                sub.linear_constraints.add(lin_expr=[cpx.SparsePair(ind = copy_ind, val = copy_val)], senses = ['L'], rhs = copy_rhs, names = [f"cbd{i}"])

        self.turnLogOff(sub)

        return sub
        
    def create_subproblems(self, isEqual = True, addCopy = False, createLagrn = False):

        """
        isEqual (if true all constraints are written as = by adding slacks)
        """

        scen_count = len(self.scenario_dict)
        for idx in range(scen_count):
            if isEqual:    
                self.scenario_dict[idx].cpx_model        = self.subproblem_equals(addCopy)
                self.scenario_dict[idx].solution         = self.scenario_dict[idx].cpx_model.solution
                self.scenario_dict[idx].solution_adv     = self.scenario_dict[idx].cpx_model.solution.advanced
            else:
                self.scenario_dict[idx].cpx_model        = self.subproblem_nonequals()
                self.scenario_dict[idx].solution         = self.scenario_dict[idx].cpx_model.solution
                self.scenario_dict[idx].solution_adv     = self.scenario_dict[idx].cpx_model.solution.advanced

            self.scenario_dict[idx].cpx_model_mip        = self.subproblem_mip(addCopy)
            self.scenario_dict[idx].solution_mip         = self.scenario_dict[idx].cpx_model_mip.solution
            self.scenario_dict[idx].constr_count         = self.scenario_dict[idx].cpx_model.linear_constraints.get_num()
            self.scenario_dict[idx].constr_count_mip     = self.scenario_dict[idx].constr_count
            self.scenario_dict[idx].secondStageBinVars   = self.secondStageBinVars
            self.scenario_dict[idx].secondStageIntVars   = self.secondStageContVars

            if createLagrn:
                self.scenario_dict[idx].cpx_model_lagrn  = self.subproblem_lagrn()

            # if idx == 0:
            #     self.scenario_dict[idx].cpx_model.write(lp_path + f"subLP_{idx}_before_clean.lp")
            #     self.scenario_dict[idx].cpx_model_mip.write(lp_path + f"subMIP_{idx}_before_clean.lp")
            #     print("SubLP published")
    
    def create_subproblems_withCopy(self, isEqual = True):

        """
        isEqual (if true all constraints are written as = by adding slacks)
        """

        scen_count = len(self.scenario_dict)
        for idx in range(scen_count):
            if isEqual:    
                self.scenario_dict[idx].cpx_model        = self.subproblem_equals()
                # self.scenario_dict[idx].solution         = self.scenario_dict[idx].cpx_model.solution
                # self.scenario_dict[idx].solution_adv     = self.scenario_dict[idx].cpx_model.solution.advanced
            else:
                self.scenario_dict[idx].cpx_model        = self.subproblem_nonequals()
                # self.scenario_dict[idx].solution         = self.scenario_dict[idx].cpx_model.solution
                # self.scenario_dict[idx].solution_adv         = self.scenario_dict[idx].cpx_model.solution.advanced

            self.scenario_dict[idx].cpx_model_mip        = self.subproblem_mip()
            # self.scenario_dict[idx].solution_mip         = self.scenario_dict[idx].cpx_model_mip.solution
            self.scenario_dict[idx].constr_count         = self.scenario_dict[idx].cpx_model.linear_constraints.get_num()
            self.scenario_dict[idx].constr_count_mip     = self.scenario_dict[idx].constr_count
            self.scenario_dict[idx].secondStageBinVars   = self.secondStageBinVars
            self.scenario_dict[idx].secondStageIntVars   = self.secondStageContVars

    def lhs_technology_matrix(self):

        """
        Input: sub (cplex model) that we will parameterize
        """

        
        sub = self.scenario_dict[0].cpx_model
        
        sparseTech = sub.variables.get_cols(self.original_masterVars)
        sparseTechLen = len(sparseTech)
        techMatrix = np.zeros((sub.linear_constraints.get_num(), sparseTechLen))

        for j in range(sparseTechLen):
            sparseCol = sparseTech[j]
            ind, val = sparseCol.unpack()
            for i in range(len(ind)):
                rowi = ind[i]
                techMatrix[rowi, j] = val[i]

        return techMatrix

    def rhs_technology_matrix_withCopy(self):

        """
        Input: sub (cplex model) that we will parameterize

        The rhs technology matrix with copy constraints
        """

        
        sub = self.scenario_dict[0].cpx_model
        
        # sparseTech = sub.variables.get_cols(self.original_masterVars)
        # sparseTechLen = len(sparseTech)
        Nrows      = sub.linear_constraints.get_num()
        techMatrix = np.zeros((Nrows, self.Ncopy))
        start = Nrows-self.Ncopy

        for j in range(self.Ncopy):
            techMatrix[start + j, j] = 1

        

        return techMatrix

    def delete_master_vars(self, sub):
        """
        delete the master variables from the original problem
        
        """   

        sub.variables.delete(self.original_masterVars)

    def delete_master_vars_all_sub(self):

        """
        delete master variables from all subproblems
        """

        scen_count = len(self.sub_idxs)
        for idx in range(scen_count):     
            self.delete_master_vars(self.scenario_dict[idx].cpx_model)
            self.delete_master_vars(self.scenario_dict[idx].cpx_model_mip)

    def updateLowerBoundForSubproblems(self):

        """
        updates the lower bound for the subproblems
        assumes that the problem has c0 constraint already setup
        """



        self.global_lower_bd = 0

        
        for idx in self.sub_idxs: 
            
            sub = self.scenario_dict[idx]
            lb = 0
            
            for conName in sub.constraintMap:
                rhs = sub.constraintMap[conName]
                sparseObj = sub.cpx_model.linear_constraints.get_rows(conName)
                indices, values = sparseObj.unpack()
                coeff_pairs = [('c0', i) for i in indices]
                lb = lb  - max(sub.cpx_model.linear_constraints.get_coefficients(coeff_pairs))*rhs

            sub.lower_bound = lb
            self.global_lower_bd += sub.probability*sub.lower_bound

    def updateUpperBoundForSubproblems(self):

        """
        updates the lower bound for the subproblems
        assumes that the problem has c0 constraint already setup
        """


        
        for idx in self.sub_idxs: 
            
            sub = self.scenario_dict[idx]
            ub = 0
            
            for conName in sub.constraintMap:
                rhs = sub.constraintMap[conName]
                sparseObj = sub.cpx_model.linear_constraints.get_rows(conName)
                indices, values = sparseObj.unpack()
                coeff_pairs = [('c0', i) for i in indices]
                ub = ub  + (1000-1)*min(sub.cpx_model.linear_constraints.get_coefficients(coeff_pairs))*rhs

            sub.upper_bound = ub

    def constr_name_to_index_map(self):

        """
        scroll through constraints to determine what name corresponds to what index
        """

        sub = self.scenario_dict[0].cpx_model
        name_list = sub.linear_constraints.get_names()
        out_dict = {}

        for i in range(len(name_list)):
            out_dict[name_list[i]] = i

        return out_dict

    def rhs_random_constant(self, nameToIndexMap):

        """
        creates a dictionary of arrays
        key: scenario id, val: array indicating the rhs in that problem
        nameToIndexMap(dict): maps the name of constraint in the sub to the index of the constraint
        """

        scen_ids = self.scenario_dict.keys()
        

        for id in scen_ids:
            idsub = self.scenario_dict[id]
            rhs_array = np.zeros(idsub.cpx_model.linear_constraints.get_num())
            cnames = idsub.cpx_model.linear_constraints.get_names()
            for cname in cnames:
                idx = nameToIndexMap[cname]
                if cname in idsub.constraintMap: 
                    rhs_array[idx] = idsub.constraintMap[cname]
                else:
                    rhs_array[idx] = idsub.cpx_model.linear_constraints.get_rhs(cname)
                    
            idsub.rhs_const = rhs_array
            idsub.rhs_const_mip = copy.deepcopy(rhs_array)

    def rhs_update_lagrn(self, nameToIndexMap):

        """
        creates a dictionary of arrays
        key: scenario id, val: array indicating the rhs in that problem
        nameToIndexMap(dict): maps the name of constraint in the sub to the index of the constraint
        """

        scen_ids = self.scenario_dict.keys()
        

        for id in scen_ids:
            idsub = self.scenario_dict[id]
            rhs_array = np.zeros(idsub.cpx_model_lagrn.linear_constraints.get_num())
            cnames = idsub.cpx_model_lagrn.linear_constraints.get_names()
            for cname in cnames:
                idx = nameToIndexMap[cname]
                if cname in idsub.constraintMap: 
                    rhs_array[idx] = idsub.constraintMap[cname]
                    idsub.cpx_model_lagrn.linear_constraints.set_rhs(cname, idsub.constraintMap[cname])
                else:
                    rhs_array[idx] = idsub.cpx_model_lagrn.linear_constraints.get_rhs(cname)
                    
            # idsub.rhs_const = rhs_array
            # idsub.rhs_const_mip = copy.deepcopy(rhs_array)

            
    def main(self, isEqual = True, addCopy = False, createLagrn = False, isRuiInst = False):


        if not isRuiInst:
            self.loadScenarios(self.datapath + f"{self.name}.sto")
        else:
            map = pk.load(open(pickle_path + self.name + "_map.p", 'rb'))
            prob = pk.load(open(pickle_path + self.name + "_prob.p", 'rb'))
            self.loadScenarios_rui(map, prob)


        # self.filterVars()
        self.filterConstrs()
        self.turnIntoLessThanConstraints()
        self.master_generation()
        self.create_subproblems(isEqual = isEqual, addCopy = addCopy, createLagrn=createLagrn)
        
        # lhs_tech_matrix = self.lhs_technology_matrix()    
        # self.delete_master_vars_all_sub()

        if not addCopy:
            lhs_tech_matrix = self.lhs_technology_matrix()
            self.delete_master_vars_all_sub()   
        else:
            rhs_tech_matrix = self.rhs_technology_matrix_withCopy()  
        
        self.updateLowerBoundForSubproblems()
        self.updateUpperBoundForSubproblems()
        
        nameToIndexMap  = self.constr_name_to_index_map()
        self.rhs_random_constant(nameToIndexMap)
        if createLagrn:
            self.rhs_update_lagrn(nameToIndexMap)

        if not addCopy:
            for idx in self.sub_idxs:
                self.scenario_dict[idx].rhs_tech_row = copy.deepcopy(-lhs_tech_matrix)
                self.scenario_dict[idx].rhs_tech_row_mip = copy.deepcopy(-lhs_tech_matrix)
        else:
            for idx in self.sub_idxs:
                self.scenario_dict[idx].rhs_tech_row     = copy.deepcopy(rhs_tech_matrix)
                self.scenario_dict[idx].rhs_tech_row_mip = copy.deepcopy(rhs_tech_matrix)

    def main_multicut(self, isEqual = True, addCopy = False, createLagrn = False, isRuiInst = False):

        if not isRuiInst:
            _, self.scenProb = self.loadScenarios(self.datapath + f"{self.name}.sto")
        else:
            map = pk.load(open(pickle_path + self.name + "_map.p", 'rb'))
            prob = pk.load(open(pickle_path + self.name + "_prob.p", 'rb'))
            _, self.scenProb = self.loadScenarios_rui(map, prob)


        # self.filterVars()
        self.filterConstrs()
        self.turnIntoLessThanConstraints()
        self.master_generation_multicut()
        self.create_subproblems(isEqual = isEqual, addCopy = addCopy, createLagrn=createLagrn)
        
        if not addCopy:
            lhs_tech_matrix = self.lhs_technology_matrix()
            self.delete_master_vars_all_sub()   
        else:
            rhs_tech_matrix = self.rhs_technology_matrix_withCopy()  
        
        self.updateLowerBoundForSubproblems()
        self.updateUpperBoundForSubproblems()
        
        nameToIndexMap  = self.constr_name_to_index_map()

        self.rhs_random_constant(nameToIndexMap)
        if createLagrn:
            self.rhs_update_lagrn(nameToIndexMap)
            
        if not addCopy:
            for idx in self.sub_idxs:
                self.scenario_dict[idx].rhs_tech_row     = copy.deepcopy(-lhs_tech_matrix)
                self.scenario_dict[idx].rhs_tech_row_mip = copy.deepcopy(-lhs_tech_matrix)
        else:
            for idx in self.sub_idxs:
                self.scenario_dict[idx].rhs_tech_row     = copy.deepcopy(rhs_tech_matrix)
                self.scenario_dict[idx].rhs_tech_row_mip = copy.deepcopy(rhs_tech_matrix)

    def convert_problem_to_grb(self, isMultiCut = False, createLagrn = False):

        """
        at the very end when all updates are done convert the master problem into gurobi model
        """
        self.masterObj.cpx_model.write(lp_path + "MASTER.lp")
        self.masterObj.grb_model = gp.read(lp_path + "MASTER.lp")
        self.masterObj.grb_model.Params.OutputFlag = 0

        if not isMultiCut:
            self.masterObj.tVar_grb = self.masterObj.grb_model.getVarByName("t")

        
        self.masterObj.state_vars_grb  = [self.masterObj.grb_model.getVarByName(i) for i in self.masterObj.vars_in_stage2]
        self.masterObj.future_vars_grb = [self.masterObj.grb_model.getVarByName(i) for i in self.masterObj.future_cost_var_names]


        for idx in self.sub_idxs:
            self.scenario_dict[idx].cpx_model.write(lp_path + f"SUB_{idx}.lp")
            self.scenario_dict[idx].cpx_model_mip.write(lp_path + f"SUB_MIP_{idx}.lp")
            if createLagrn:
                self.scenario_dict[idx].cpx_model_lagrn.write(lp_path + f"SUB_MIP_LAGRN_{idx}.lp")

            self.scenario_dict[idx].grb_model             = gp.read(lp_path + f"SUB_{idx}.lp")
            self.scenario_dict[idx].grb_model_mip         = gp.read(lp_path + f"SUB_MIP_{idx}.lp")
            if createLagrn:
                self.scenario_dict[idx].grb_model_lagrn   = gp.read(lp_path + f"SUB_MIP_LAGRN_{idx}.lp")

            self.scenario_dict[idx].grb_model.Params.OutputFlag       = 0
            self.scenario_dict[idx].grb_model_mip.Params.OutputFlag   = 0
            if createLagrn:
                self.scenario_dict[idx].grb_model_lagrn.Params.OutputFlag = 0

            self.scenario_dict[idx].grb_model_constrs     = self.scenario_dict[idx].grb_model.getConstrs()
            self.scenario_dict[idx].grb_model_mip_constrs = self.scenario_dict[idx].grb_model_mip.getConstrs()
            
            self.scenario_dict[idx].constrs_mip_grb = len(self.scenario_dict[idx].grb_model_mip_constrs)
            self.scenario_dict[idx].constrs_grb = len(self.scenario_dict[idx].grb_model_constrs)

            self.masterObj.state_vars_grb  = [self.masterObj.grb_model.getVarByName(i) for i in self.masterObj.vars_in_stage2]
            if createLagrn:
                self.scenario_dict[idx].grb_lp_copy_constrs  = [self.scenario_dict[idx].grb_model.getConstrByName(f"copy{i}") for i in range(len(self.original_masterVars))]
                self.scenario_dict[idx].grb_lagrn_copy_vars     = [self.scenario_dict[idx].grb_model_lagrn.getVarByName(i) for i in self.original_masterVars]
        


class Subproblem(Primitive):

    #subproblem and scenario means the same thing

    def __init__(self, idx):
        """
        idx: scenario id
        constraintMap (dict): constraint name and corresponding rhs (#hardcoding for sslp dataset)    
        prob (float): probability of scenario
        """

        super().__init__()
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
        self.grb_model = None           #Gurobi model same as cplex model
        self.grb_model_mip = None
        self.grb_model_constrs = None
        self.grb_model_mip_constrs = None   
        self.solution = None
        self.dual = None                #contains the dual vector info used to create the cut
        self.cut = None                 #benders cut used for adding to the masters
        self.gfcs = 0                   #number of gfc cuts added to this subproblem
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
        self.prev_sols = {}             #solutions from previous solves that have been stored

    def get_cons_names(self):

        return self.cpx_model.linear_constraints.get_names()

    


class Master(Primitive):
    
    # master class for declaring the master problem and relevant information

    def __init__(self, name, master, futureCostVarCount = 1):
        
        super().__init__()       
        self.name = name
        # self.master = master
        self.cpx_model      = master      #contains the cplex model
        self.grb_model      = None
        self.optCuts        = 0            #records the number of constraints in the Master problem
        self.xVarsInd       = None
        self.pVarsInd       = None
        self.xVars_ind      = None       #x variables indices in the problem
        self.pVars_ind      = None       #p variables indices in the problem
        self.xVars_names    = None     #x variables names 
        self.pVars_names    = None     #name of p variables in the problem

        master_var_names = master.variables.get_names()

        self.cols_in_stage2   = master.variables.get_num() - futureCostVarCount      #variable indices that affect stage 2
        self.vars_in_stage2   = master_var_names[:-futureCostVarCount]               #variable names (str) that affect stage 2
        self.future_cost_var_names = master_var_names[-futureCostVarCount:]           
        self.future_vars_grb = None                                                  #variables objects of t1, .., t_Nscen
        self.state_vars_grb = None
        
        self.sols_in_stage2 = None
        self.curr_sol       = None
        self.t              = None       # value variable that approximates future cost
        self.tVar_grb       = None
        self.tVar_val       = None
        
    def get_solution(self):


        """
        returns: solution from the master problem
        x (numpy array of x values)
        t (scalar) cost to go approximation  
        returns:
        format:
        entire solution vector other than t, solution vector that affects the second stage subproblem, t
        """

        sols = self.cpx_model.solution.get_values()
        self.curr_sol  = np.array(sols[:-1])
        self.sols_in_stage2 = self.curr_sol
        self.t = sols[-1]

        return self.t

    def get_futurecost_gurobi(self):

        """
        
        """

        self.curr_sol =  np.array(self.grb_model.getAttr("X", self.state_vars_grb))
        self.sols_in_stage2 = self.curr_sol
        self.tVar_val =  self.tVar_grb.x

        return self.tVar_val

    def get_futurecost_gurobi_multicut(self):

        """
        gets the future cost for variables t1, t2, ... ,t_Nscen
        Also updates the solution vector for state variables        
        """

        self.curr_sol       =  np.array(self.grb_model.getAttr("X", self.state_vars_grb))
        self.sols_in_stage2 =  self.curr_sol
        self.tVar_val       =  np.array(self.grb_model.getAttr("X", self.future_vars_grb))

        return self.tVar_val



        













    



    
    
    

        

    