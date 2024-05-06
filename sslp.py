import cplex as cpx
import numpy as np
from primitives import gfc_step_ver2, turnLogOff, isFractional, source_row, gades_transformation_step, gfc_step
import copy


class SSLPProblemInstance:

    def __init__(self, inst_name):
        
        """
        Input:
            inst_name (str): name of the form sslp_servercount_clientcount_scenarios
        """

        self.name = inst_name                    #name of instance
        self.fsv = None                          #number of servers (first stage variables (excluding t))
        self.scenN = None                        #number of scenarios
        self.masterVarsLen = None                #number of fsv including t or theta
        self.masterVars = []                     #first stage variable including variable t (or theta)
        self.original_masterVars = []            #first stage variable names
        self.secondStageBinVars = []             #second stage binary variable names y_ij
        self.secondStageContVars = []            #second stage continuous variables y0 variables    
        self.vars = []                           #name all variables
        self.master_constr_indices = []          #indices of master problem in global_model
        self.sub_constr_indices = []             #indices of subproblem in global_model
        self.sub_constr_with_fsv = []            #list of constraint indices (in original subproblems) which depend on first stage variables
        self.subs_constr_with_fsv = {}           #scenario ids -> list of contraints indices which have first stage variables
        self.subs_constr_with_fsv_mip = {}       #subproblem constraints (mip) with first stage variables
        self.sub_ineq_constr_no_fsv = []         #subproblem inequality constraints with no first stage variables
        self.subs_ineq_constr_no_fsv = {}        #subproblem inequality constraints with no first stage variables
        self.sub_vars_with_fsv = []              #index of fsv in second stage subproblems (only used for figuring out rhs tech matrix)
        self.sub_ineq_constr = []                #index on inequality (non equality) constraints in the subproblem
        self.subs_ineq_constr  = {}              #for each scenario we store the index of inequality constraints in that problem
        self.sub_constr_nameToIndex_map = {}     #map for initial subproblems mapping names of constraints to corresponeding indices
        self.sub_constr_count = {}               #number of constraints in each subproblem
        self.sub_constr_count_mip = {}           #number of constraints in each subproblem mip
        self.rhs_const = {}                      #constants on rhs for each scenario
        self.rhs_const_mip = {}                  #constants on rhs in mip (where no gfc is added)
        self.rhs_const_ineq = {}                 #rhs constant corresponding to inequality constraint for each scenario
        self.rhs_tech = {}                       #right hand side technology matrix for each scenario
        self.rhs_tech_mip = {}                   #right hand side technology matrix for each scenario - mip version
        self.clients = None                      #number of clients
        self.scendict_lp = {}                    #scenario ids -> cplex subproblems lp
        self.scendict_mip = {}                   #scenario ids -> cplex subproblems mip
        self.scenprob = {}                       #scenario ids -> probability
        self.scenmaps = {}                       #scenario ids -> maps (dicts: constraints name -> rhs)
        self.sub_obj = {}                        #sub objective for each subproblem
        self.coeffs = {}                         #x coefficients of the benders cut for each scenario
        self.const = {}                          #constant or rhs term in bernders cut for each scenario
        self.global_model = None                 #used for loading the data from mps file
        self.master = cpx.Cplex()                #master problem
        
    def datapopulate(self, datapath):

        """
        scenario map (dict os dicts) for each scenario
        inner dict maps constraint name to random rhs

        """

        
        with open(datapath + f"{self.name}.sto", "r") as f:
            data = f.readlines()
        
        idx = 0
        for line in data:
            words = line.split()
            if len(words) > 2:
                if words[0] == "SC":
                    self.scenprob[idx] = float(words[3])
                    self.scenmaps[idx] = {}
                    idx += 1
                elif words[0] == "RHS":
                    self.scenmaps[idx-1][words[1]] = float(words[2])
                
        self.scenN = idx

        
        self.global_model = cpx.Cplex(datapath + f"{self.name}.mps")

    def filterVars(self): 
    
        """
        identifies variables that belong only to the master problem
        also variables which are continuous in the second stage
        and variables with 1 as upper bound in the second stage

        assumes that y0 variables is of the form x_#_0
        """
        
        self.vars = self.global_model.variables.get_names()

        for var in self.vars:
            if var.count("x") == 1:
                if var.count("_") == 1:
                    self.original_masterVars.append(var)
                else:
                    self.secondStageContVars.append(var)
            else:
                self.secondStageBinVars.append(var)

        self.masterVars = self.original_masterVars.copy()
        self.fsv = len(self.original_masterVars)
        self.masterVarsLen = self.fsv

    def filterConstrs_sslp(self): 
        
        """
        filter constraints for SSLP problems
        """


        self.constrs = self.global_model.linear_constraints.get_names()
        constrs = self.global_model.linear_constraints.get_rows()
        

        for i, constr in enumerate(constrs):
            ind, val = constr.unpack()
            if self.areIndSecStage(ind) == True:
                self.sub_constr_indices.append(i)
            else:
                self.master_constr_indices.append(i)

    def filterSubConstrsWithFsvAndIneqs(self, id):
        """
        determines constraints in subproblems that also have (a) first stage variables
        (b) Inequality constraints and no first stage variables
        (c) Inequality constraints
        """

        sub     = self.scendict_lp[id]
        constrs = sub.linear_constraints.get_rows()
        vars    = sub.variables.get_names()
        senses = sub.linear_constraints.get_senses()

        for index, constr in enumerate(constrs):
            ind, val = constr.unpack()

            hasFsv = False
            for j in ind:
                if vars[j].count("x") == 1 and vars[j].count("_") == 1:
                    hasFsv = True
                    break

            if hasFsv:
                self.sub_constr_with_fsv.append(index)
                self.sub_ineq_constr.append(index)
            elif senses[index] != 'E':
                self.sub_ineq_constr_no_fsv.append(index)
                self.sub_ineq_constr.append(index)

    def updateSubConstrsWithFsvAndIneqs(self):

        """
        update the subproblem constraints with 
        (a) first stage variables
        (b) inequality constraints and no first stage variables
        """
        
        for id in range(self.scenN):
            self.subs_constr_with_fsv[id]     = self.sub_constr_with_fsv.copy()
            self.subs_constr_with_fsv_mip[id] = self.sub_constr_with_fsv.copy()
            self.subs_ineq_constr_no_fsv[id]  = self.sub_ineq_constr_no_fsv.copy()
            self.subs_ineq_constr[id]         = self.sub_ineq_constr.copy()                     

    def filterSubVarsWithFsv(self, id):

        vars    = self.scendict_lp[id].variables.get_names()

        for index, var in enumerate(vars):
            if var.count("x") == 1 and var.count("_") == 1:
                self.sub_vars_with_fsv.append(index)

    def areIndSecStage(self, ind):

        """
        ind: list of indices
        determines if the indices are second stage
        """    

        flag = False
        for j in ind:
            if self.vars[j] not in self.original_masterVars:
                flag = True
                break
        
        return flag

    def areIndFirstStage(self, ind):

        """
        ind: list of indices
        determines if the indices are second stage
        """    

        flag = False
        for j in ind:
            if self.vars[j] in self.original_masterVars:
                flag = True
                break
        
        return flag

    def turnIntoLessThanConstraints(self):

        """
        converts the >= constraints into less than equal to constraints
        so that sign of duals are all positive
        """

        if len(self.sub_constr_indices) == 0:
            self.filterConstrs_sslp()

        for cId in self.sub_constr_indices:

            if self.global_model.linear_constraints.get_senses(cId) == 'G':
                ind, val = self.global_model.linear_constraints.get_rows(cId).unpack()
                val = list(-np.array(val))
                new = cpx.SparsePair(ind = ind, val = val)
                self.global_model.linear_constraints.set_linear_components(cId, new)

                old_rhs = self.global_model.linear_constraints.get_rhs(cId)
                self.global_model.linear_constraints.set_rhs(cId, -old_rhs)
                self.global_model.linear_constraints.set_senses(cId, 'L')

    def gen_master(self):

        """
        generates the master problem
        ensure that the the first constraint in the global model is the capacity constraint
        """

        #add variables to the master problem -> same as those in the global model
        self.master.variables.add(obj = self.global_model.objective.get_linear(self.original_masterVars), types = ["B"]*self.fsv, names = self.original_masterVars)

        #constraint #c1 is assumed to capture the cardinality constraint in the original problem
        
        self.master.linear_constraints.add(lin_expr = [self.global_model.linear_constraints.get_rows("c1")], senses = [self.global_model.linear_constraints.get_senses(0)], rhs = [self.global_model.linear_constraints.get_rhs(0)], names = ["c1"]) 
        self.master.variables.add(obj = [1.0], lb = [-cpx.infinity], types = ["C"], names = ['t'])
        self.masterVars.append('t')
        self.masterVarsLen += 1
        turnLogOff(self.master)

    def create_all_subproblems(self):
        
        """
        
        """
        
        for idx in range(self.scenN):
            self.scendict_lp[idx] = self.subproblem_sslp(idx)
            self.scendict_mip[idx] = self.subproblem_sslp_mip(idx)

    def updateConstraintRowsIndex(self, rows):

        #the variable index in increased by one because we add the y0 variable as the first variable

        constrs = []
        for sparseObj in rows:


            ind, val = sparseObj.unpack()
            new_ind = [int(i) for i in np.array(ind) + np.ones(len(ind))]
            constrs.append(cpx.SparsePair(ind = new_ind, val  = val))
    
        return constrs
    
    def subproblem_sslp(self, idx):

        """
        here original master variables are also added and they are deleted later
        """

        sub = cpx.Cplex()
        sub.variables.add(obj  = [1.0], names = ["y0"], lb = [-cpx.infinity])

        #note all variables are continuous by default
        sub.variables.add(names = self.original_masterVars)   #these first stage variables are deleted later in another function from subproblem defn
        sub.variables.add(names= self.secondStageBinVars , lb = [0]*len(self.secondStageBinVars))
        sub.variables.add(names= self.secondStageContVars)     #x variables in second stage (y_{0, \omega} variables)

        #the objective is added as a constraint in the problem
        ind1 = self.secondStageBinVars.copy()
        val1 = list(-np.array(self.global_model.objective.get_linear(self.secondStageBinVars)))
        ind2 = self.secondStageContVars.copy()
        val2 = list(-np.array(self.global_model.objective.get_linear(self.secondStageContVars)))
        ind0 = ["y0"]
        val0 = [1]
        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind0 + ind1 + ind2, val = val0 + val1 + val2)], senses = ["E"], rhs = [0], names = ['c0'])
        
        #rest of the constraints
        newRows = self.updateConstraintRowsIndex(self.global_model.linear_constraints.get_rows(self.constrs[1:]))
        sub.linear_constraints.add(newRows, senses = self.global_model.linear_constraints.get_senses(self.constrs[1:]), rhs = self.global_model.linear_constraints.get_rhs(self.constrs[1:]), names = self.constrs[1:])
        


        #y <= 1 constraint
        yBoundConstrs = [cpx.SparsePair(ind = [yVar], val = [1]) for yVar in self.secondStageBinVars]
        yBoundConstr_names = [f"c{i}" for i in range(len(self.constrs)+1, len(self.constrs)+1 + len(yBoundConstrs))]
        sub.linear_constraints.add(yBoundConstrs, senses = ['L']*len(self.secondStageBinVars), rhs = [1]*len(self.secondStageBinVars))

        #updating the rhs specific to idx
        for cname, crhs in self.scenmaps[idx].items():
            sub.linear_constraints.set_rhs(cname, crhs)

        turnLogOff(sub)
        sub.parameters.lpmethod.set(2) #dual simple method
        self.sub_constr_count[idx] = sub.linear_constraints.get_num()

        return sub

    def subproblem_sslp_mip(self, idx):

        sub = cpx.Cplex()
        sub.variables.add(obj  = [1.0], names = ["y0"], lb = [-cpx.infinity], types = ["I"])

        #note all variables are continuous by default
        sub.variables.add(names = self.original_masterVars)   #these first stage variables are deleted later in another function from subproblem defn
        sub.variables.add(names= self.secondStageBinVars , lb = [0]*len(self.secondStageBinVars), types = ['B']*len(self.secondStageBinVars))
        sub.variables.add(names= self.secondStageContVars, types = ['I']*len(self.secondStageContVars))     #x variables in second stage (y_{0, \omega} variables)

        #the objective is added as a constraint in the problem
        ind1 = self.secondStageBinVars.copy()
        val1 = list(-np.array(self.global_model.objective.get_linear(self.secondStageBinVars)))
        ind2 = self.secondStageContVars.copy()
        val2 = list(-np.array(self.global_model.objective.get_linear(self.secondStageContVars)))
        ind0 = ["y0"]
        val0 = [1]
        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind0 + ind1 + ind2, val = val0 + val1 + val2)], senses = ["E"], rhs = [0], names = ['c0'])
        
        #rest of the constraints
        newRows = self.updateConstraintRowsIndex(self.global_model.linear_constraints.get_rows(self.constrs[1:]))
        sub.linear_constraints.add(newRows, senses = self.global_model.linear_constraints.get_senses(self.constrs[1:]), rhs = self.global_model.linear_constraints.get_rhs(self.constrs[1:]), names = self.constrs[1:])
        

        #updating the rhs specific to idx
        for cname, crhs in self.scenmaps[idx].items():
            sub.linear_constraints.set_rhs(cname, crhs)

        turnLogOff(sub)
        self.sub_constr_count_mip[idx] = sub.linear_constraints.get_num()


        return sub

    def get_rhs_tech_matrix(self):

        """
        for each scenario we need to maintain a matrix seperately for x
        """

    def lhs_technology_matrix(self, id):

        """
        Input: sub (cplex model) that we will parameterize
        Ensure that sub_constr_with_fsv and self.sub_vars_with_fsv is populated already
        """

        #create the technology matrix
        sub = self.scendict_lp[id]
        
        rowLen = len(self.sub_constr_with_fsv)
        colLen = self.fsv
        
        techMatrix = np.zeros((rowLen, colLen))
    
    
        for rowInd in range(rowLen):
            rowElem = self.sub_constr_with_fsv[rowInd]
            for colInd in range(colLen):
                colElem = self.sub_vars_with_fsv[colInd]   
                techMatrix[rowInd, colInd] = sub.linear_constraints.get_coefficients(rowElem, colElem)

        return techMatrix

    def lhs_technology_matrix_ver2(self, id):

        """
        Input: sub (cplex model) that we will parameterize
        Ensure that sub_constr_with_fsv and self.sub_vars_with_fsv is populated already
        ver2: here we use constraint indices from self.sub_ineq_constr instead of self.sub_constr_with_fsv
        """

        #create the technology matrix
        sub = self.scendict_lp[id]
        
        rowLen = len(self.sub_ineq_constr)
        colLen = self.fsv
        
        techMatrix = np.zeros((rowLen, colLen))
    
    
        for rowInd in range(rowLen):
            rowElem = self.sub_ineq_constr[rowInd]
            for colInd in range(colLen):
                colElem = self.sub_vars_with_fsv[colInd]   
                techMatrix[rowInd, colInd] = sub.linear_constraints.get_coefficients(rowElem, colElem)

        return techMatrix 

    def lhs_technology_matrix_verOrig(self, id):

        """
        Input: id sub problem id
        """

        #create the tech matrix

        sub = self.scendict_lp[id]

        rowLen = sub.linear_constraints.get_num()
        sparseTech = sub.variables.get_cols(self.original_masterVars)

        

        techMatrix = np.zeros((rowLen, self.fsv))

        for j in range(self.fsv):
            sparseCol = sparseTech[j]
            ind, val = sparseCol.unpack()
            for i in range(len(ind)):
                rowi = ind[i]
                techMatrix[rowi, j] = val[i]

        return techMatrix

    def delete_master_vars_all_sub(self):

        """
        delete master variables from all subproblems
        """

        
        for idx in range(self.scenN):     
            self.delete_master_vars(self.scendict_lp[idx])
            self.delete_master_vars(self.scendict_mip[idx])

    def delete_master_vars(self, sub):
        
        """
        delete the master variables from the given sub-problem
        """
        
        sub.variables.delete(self.original_masterVars)

    def constr_name_to_index_map(self, id):

        """
        scroll through constraints to determine what name corresponds to what index
        used for determining the rhs_const vector
        """


        sub = self.scendict_lp[id]
        name_list = sub.linear_constraints.get_names()
        

        for i, name in enumerate(name_list):
            self.sub_constr_nameToIndex_map[name] = i
        
    def rhs_random_constant_ver2(self, id):

        """
        creates rhs constant only for constraints that also have x variables
        """

        sub = self.scendict_lp[id]
        all_rhs = np.array(sub.linear_constraints.get_rhs())
        rhs_array = all_rhs[self.sub_constr_with_fsv]
        rhs_ineqs = all_rhs[self.sub_ineq_constr_no_fsv]

        rhs_array = np.array(sub.linear_constraints.get_rhs(self.sub_constr_with_fsv))

        for id in range(self.scenN):
            self.rhs_const[id] = copy.deepcopy(rhs_array)
            self.rhs_const_mip[id] = copy.deepcopy(rhs_array)
            self.rhs_const_ineq[id] = copy.deepcopy(rhs_ineqs)

    def rhs_random_constant_ver3(self, id):

        """
        creates rhs constant only for constraints that also have x variables
        ver3 : here the rhs is store for all inequality constraints - this is used for translating the problem to standard form
        """

        
        sub = self.scendict_lp[id]
        all_rhs = np.array(sub.linear_constraints.get_rhs())
        rhs_noneqs = all_rhs[self.sub_ineq_constr]

        rhs_array = np.array(sub.linear_constraints.get_rhs(self.sub_constr_with_fsv))

        for id in range(self.scenN):
            self.rhs_const[id] = copy.deepcopy(rhs_noneqs)
            self.rhs_const_mip[id] = copy.deepcopy(rhs_noneqs)
            
  
    def rhs_random_constant_verOrig(self):

        """
        creates a dictionary of arrays
        key: scenario id, val: array indicating the rhs in that problem
        """

        scen_ids = list(range(self.scenN))

        for id in scen_ids:
            idsub = self.scendict_lp[id]
            # rhs_array = np.zeros(idsub.linear_constraints.get_num())
            # cnames = idsub.linear_constraints.get_names()
            

            # for cname in cnames:
            #     idx = self.sub_constr_nameToIndex_map[cname]
            #     rhs_array[idx] = idsub.linear_constraints.get_rhs(cname)
                    
            rhs_array              = np.array(idsub.linear_constraints.get_rhs())
            self.rhs_const[id]     = copy.deepcopy(rhs_array)
            self.rhs_const_mip[id] = copy.deepcopy(rhs_array)
        
    def update_rhs_tech_matrix(self, lhs_tech_matrix):
        
        for idx in range(self.scenN):     
            self.rhs_tech[idx] = copy.deepcopy(-lhs_tech_matrix)
            self.rhs_tech_mip[idx] = copy.deepcopy(-lhs_tech_matrix)
            
    def readAndIntializeSSLP(self, datapath):

        """
        data is loaded from .sto and .cor ( or .mps) file
        files have variables of form x_#, y_#_#, x_#_#
        constraints are labelled c#
        first constraint should be the capacity costraint on number of servers
        note the versions we are using in this function definition
        """

        # loads scenario maps and global model 
        self.datapopulate(datapath)
        self.filterVars()
        self.filterConstrs_sslp()
        self.turnIntoLessThanConstraints()

        self.gen_master()
        self.create_all_subproblems()
        self.filterSubConstrsWithFsvAndIneqs(0)

        # records which constraints in each subproblem also have first stage variables
        self.updateSubConstrsWithFsvAndIneqs()
        self.filterSubVarsWithFsv(0)

        # self.update_rhs_tech_matrix(self.lhs_technology_matrix_verOrig(0))
        self.update_rhs_tech_matrix(self.lhs_technology_matrix_verOrig(0))
        self.delete_master_vars_all_sub()
        self.rhs_random_constant_verOrig()
        
    def updateSubproblems_withIncmbt(self, x):

        """
        update the subproblem with incumbent solution x
        """

        for id in range(self.scenN):
            sub = self.scendict_lp[id]
            new_col = self.rhs_const[id] +  np.dot(self.rhs_tech[id], x)
            new_rhs = list(zip(self.subs_constr_with_fsv[id], new_col))
            sub.linear_constraints.set_rhs(new_rhs)

    def updateSubproblems_withIncmbt_ver2(self, x):

        """
        update the subproblem with incumbent solution x
        ver2 : using a different index list
        """

        for id in range(self.scenN):
            sub = self.scendict_lp[id]
            new_col = self.rhs_const[id] +  np.dot(self.rhs_tech[id], x)
            new_rhs = list(zip(self.subs_ineq_constr[id], new_col))
            sub.linear_constraints.set_rhs(new_rhs)

    def updateSubproblems_withIncmbt_verOrig(self, x):

        """
        update the subproblem with incumbent solution x
        ver2 : using a different index list
        """

        for id in range(self.scenN):
            sub = self.scendict_lp[id]
            new_col = self.rhs_const[id] +  np.dot(self.rhs_tech[id], x)
            new_rhs = list(zip(range(self.sub_constr_count[id]), new_col))                      
            sub.linear_constraints.set_rhs(new_rhs)

    def solve_all_subproblems_as_mip(self):

        """
        solve all subproblems as mixed integer program and get their objective value
        """


        local_obj_mip = 0                  
        sub_objs = {}
        for id in range(self.scenN):
            sub = self.scendict_mip[id]
            sub.solve()
            sub_obj = sub.solution.get_objective_value()
            sub_objs[id] = sub_obj
            local_obj_mip += self.scenprob[id]*sub_obj
       
        return local_obj_mip, sub_objs

    def updateSubproblems_withIncmbt_mip_verOrig(self, x):

        """
        update the subproblems with incumbent solution
        """

        for id in range(self.scenN):
            sub = self.scendict_mip[id]
            new_col = self.rhs_const_mip[id] + np.dot(self.rhs_tech_mip[id], x)
            new_rhs = list(zip(range(self.sub_constr_count_mip[id]), new_col))
            sub.linear_constraints.set_rhs(new_rhs)

    def updateSubproblems_withIncmbt_mip_ver2(self, x):

        """
        update the subproblem with incumbent solution
        update the mip subproblem with incumbent solution
        ver2 : using a different index list
        """

        for id in range(self.scenN):
            sub = self.scendict_mip[id]                                             
            new_col = self.rhs_const_mip[id] + np.dot(self.rhs_tech_mip[id], x)
            new_rhs = list(zip(self.subs_ineq_constr[id], new_col))
            sub.linear_constraints.set_rhs(new_rhs)

    def updateSub_withIncmbt(self, id, sub, x):  
        
        """
        updates the given subproblem with incumbent solution
        """

        new_col = self.rhs_const[id] +  np.dot(self.rhs_tech[id], x)
        new_rhs = list(zip(self.subs_constr_with_fsv[id], new_col))
        sub.linear_constraints.set_rhs(new_rhs)

    def updateSub_withIncmbt_ver2(self, id, sub, x):  
        
        """
        updates the given subproblem with incumbent solution
        ver2 : using a different index list
        """

        new_col = self.rhs_const[id] +  np.dot(self.rhs_tech[id], x)
        new_rhs = list(zip(self.subs_ineq_constr[id], new_col))
        sub.linear_constraints.set_rhs(new_rhs)

    def updateSub_withIncmbt_verOrig(self, id, sub, x):  
        
        """
        updates the given subproblem with incumbent solution
        TODO: only update the final constraint
        """

        new_col = self.rhs_const[id] + np.dot(self.rhs_tech[id], x)          
        new_rhs = list(zip(range(self.sub_constr_count[id]), new_col))
        sub.linear_constraints.set_rhs(new_rhs)

    def firstFractionalBasisVar(self, sub, x, str_var_basis_info, tolerance = 1e-4):
        """
        determines the first variable which is fractional and in the basis
        
        """

        
        ind_vals = sub.solution.get_values()

        for ind, ind_basis in enumerate(str_var_basis_info):
            if abs(ind_basis - 1) < tolerance:
                ind_val = ind_vals[ind]
                if isFractional(ind_val):
                    return ind, ind_val, True
        return -1, -1, False

    def add_parameterized_gfc(self, id, sub, x, tolerance = 1e-4):

        """
        We add a parameterized gomory's fractional cut for scen id and first stage vector x
        tolerance parameter determines the tolerance within which we check if a variable is integer
        """
        
        str_var_basis_info = sub.solution.basis.get_basis()[0]

        # determines the first basis variable which is fractional
        ind, ind_val, exists = self.firstFractionalBasisVar(sub, x, str_var_basis_info, tolerance)

        if exists:
            s                      = source_row(sub.solution.advanced.binvacol(ind))            
            str_aMatrix_source_row = np.array(sub.solution.advanced.binvarow(s))                                           
            slack_row              = np.array(sub.solution.advanced.binvrow(s))                                                             #slack row has length same as the number of constraints, slack row is also the basis inverse
            slack_row_fsv          = slack_row[self.subs_constr_with_fsv[id]]                                                               #subproblem constraints with first stage variables
            slack_row_fsv_ineq     = slack_row[self.subs_constr_with_fsv[id] + self.subs_ineq_constr_no_fsv[id]]                            #slack row for constraints with first stage variables and no equality senses
            rhs_const_gfc          = np.dot(slack_row_fsv_ineq, np.concatenate(self.rhs_const[id], self.rhs_const_ineq[id]))                #rhs constant corresponding to gfc
            rhs_x                  = np.dot(slack_row_fsv, self.rhs_tech[id])                                                               #rhs_x row corresponding to        
            rhs_const_gfc, rhs_x   = gades_transformation_step(self.fsv, x, rhs_const_gfc, rhs_x)                                           #gades transformation step
            
            out_aMatrix_row, out_slack_row_fsv, out_slack_row_fsv_ineq, out_const, out_rhs_x  = gfc_step_ver2(str_aMatrix_source_row, slack_row_fsv, slack_row_fsv_ineq, rhs_const_gfc, rhs_x, x, self.fsv)
            out_aMatrix_row, out_const, out_rhs_x                 = self.convert_to_str_form_ver1(self, sub, out_aMatrix_row, out_slack_row_fsv, out_slack_row_fsv_ineq, out_const, out_rhs_x, id)
            self.updateSubproblem(out_aMatrix_row, out_const, out_rhs_x, sub, id)
            
        
        return exists

    def add_parameterized_gfc_verOrig(self, id, sub, x, tolerance = 1e-4):

        """
        We add a parameterized gomory's fractional cut for scen id and first stage vector x
        tolerance parameter determines the tolerance within which we check if a variable is integer
        """
        
        str_var_basis_info = sub.solution.basis.get_basis()[0]

        # determines the first basis variable which is fractional
        ind, ind_val, exists = self.firstFractionalBasisVar(sub, x, str_var_basis_info, tolerance)

        if exists:
            s                      = source_row(sub.solution.advanced.binvacol(ind))            
            str_aMatrix_source_row = np.array(sub.solution.advanced.binvarow(s))                                           
            slack_row              = np.array(sub.solution.advanced.binvrow(s)) 
            rhs_const_gfc          = np.dot(slack_row, self.rhs_const[id])                                                                  #slack row has length same as the number of constraints, slack row is also the basis inverse
            rhs_x                  = np.dot(slack_row, self.rhs_tech[id])                                                               #rhs_x row corresponding to        
            rhs_const_gfc, rhs_x   = gades_transformation_step(self.fsv, x, rhs_const_gfc, rhs_x)                                           #gades transformation step
            
            out_aMatrix_row, out_slack_row, out_const, out_rhs_x  = gfc_step(str_aMatrix_source_row, slack_row, rhs_const_gfc, rhs_x, x, self.fsv)
            out_aMatrix_row, out_const, out_rhs_x                 = self.convert_to_str_form_verOrig(id, sub, out_aMatrix_row, out_slack_row, out_const, out_rhs_x)
            self.updateSubproblem_verOrig(id, out_aMatrix_row, out_const, out_rhs_x, sub)
            
        
        return exists

    def add_parameterized_gfc_ver2(self, id, sub, x, tolerance = 1e-4):

        """
        We add a parameterized gomory's fractional cut for scen id and first stage vector x
        tolerance parameter determines the tolerance within which we check if a variable is integer
        """
        
        str_var_basis_info = sub.solution.basis.get_basis()[0]

        # determines the first basis variable which is fractional
        ind, ind_val, exists = self.firstFractionalBasisVar(sub, x, str_var_basis_info, tolerance)

        if exists:
            s                      = source_row(sub.solution.advanced.binvacol(ind))            
            str_aMatrix_source_row = np.array(sub.solution.advanced.binvarow(s))                                           
            slack_row              = np.array(sub.solution.advanced.binvrow(s))[self.subs_ineq_constr[id]]                                                           #slack row has length same as the number of constraints, slack row is also the basis inverse
            rhs_const_gfc          = np.dot(slack_row, self.rhs_const[id])                                                              #rhs constant corresponding to gfc
            rhs_x                  = np.dot(slack_row, self.rhs_tech[id])                                                               #rhs_x row corresponding to        
            rhs_const_gfc, rhs_x   = gades_transformation_step(self.fsv, x, rhs_const_gfc, rhs_x)                                       #gades transformation step
            
            out_aMatrix_row, out_slack_row, out_const, out_rhs_x  = gfc_step(str_aMatrix_source_row, slack_row, rhs_const_gfc, rhs_x, x, self.fsv)
            out_aMatrix_row, out_const, out_rhs_x                 = self.convert_to_str_form_ver2(id, sub, out_aMatrix_row, out_slack_row, out_const, out_rhs_x)
            self.updateSubproblem(out_aMatrix_row, out_const, out_rhs_x, sub, id)                                                       #update the subproblem 
            # print(f"srow: {srow_time}, slrow: {slackrow_time}, ot: {ops_time}, tr: {trans_time}, gfc: {gfc_step_time}, str: {str_form_time}, uptime: {update_time}")
        
        return exists

    def convert_to_str_form_ver1(self, sub, out_aMatrix_row, out_slack_row_fsv, out_slack_row_fsv_ineq, out_const, out_rhs_x, id):

        """
        convert it into a form so that there are no slack variables and no y0 variables in the equation
        out_slack_row -> only has slacks for constraints which have first stage variables
        """
        
        #project out y0 variable
        row0  = out_aMatrix_row[0]


        if row0 != 0:
            ind, val = sub.linear_constraints.get_rows(0).unpack()
            out_aMatrix_row[ind] = out_aMatrix_row[ind] -row0*np.array(val)

        
        indices = self.subs_constr_with_fsv[id]+self.subs_ineq_constr_no_fsv[id]
        non_zero_indices = [i for i in range(len(out_slack_row_fsv_ineq)) if out_slack_row_fsv_ineq[i] != 0]
        rows = [sub.cpx_model.linear_constraints.get_rows(indices[i]).unpack() for i in non_zero_indices]

        #this would require the entire rhs_const or the one corresponding to non-equality constraints
        out_const = out_const - np.dot(out_slack_row_fsv_ineq, self.rhs_const_ineq[id])

        
        out_rhs_x = out_rhs_x - np.dot(out_slack_row_fsv, self.rhs_tech[id])

        for i, (ind, val) in enumerate(rows):
            val_array = np.array(val)
            rowi = out_slack_row_fsv_ineq[non_zero_indices[i]]
            out_aMatrix_row[ind] = out_aMatrix_row[ind] -rowi*val_array

        return out_aMatrix_row, out_const, out_rhs_x

    def convert_to_str_form_ver2(self, id, sub, out_aMatrix_row, out_slack_row, out_const, out_rhs_x):

        """
        convert it into a form so that there are no slack variables and no y0 variables in the equation
        not clear how this will go
        ver2: uses inequality constraint indices

        """


        row0  = out_aMatrix_row[0]
        if row0 != 0:
            ind, val = sub.linear_constraints.get_rows(0).unpack()
            out_aMatrix_row[ind] = out_aMatrix_row[ind] -row0*np.array(val)

        ineq_indices = self.subs_ineq_constr[id]
        non_zero_indices = [i for i, si in enumerate(out_slack_row) if si != 0]
        rows = [sub.linear_constraints.get_rows(ineq_indices[i]).unpack() for i in non_zero_indices]

        out_slack_row_nonzero = out_slack_row[non_zero_indices]
        out_const = out_const - np.dot(out_slack_row_nonzero, self.rhs_const[id][non_zero_indices])
        out_rhs_x = out_rhs_x - np.dot(out_slack_row_nonzero, self.rhs_tech[id][non_zero_indices, :])

        for i, (ind, val) in enumerate(rows):
            val_array = np.array(val)
            rowi = out_slack_row[non_zero_indices[i]]
            out_aMatrix_row[ind] = out_aMatrix_row[ind] -rowi*val_array
            
        return out_aMatrix_row, out_const, out_rhs_x

    def convert_to_str_form_verOrig(self, id, sub, out_aMatrix_row, out_slack_row, out_const, out_rhs_x):

        """
        convert it into a form so that there are no slack variables and no y0 variables in the equation
        not clear how this will go
        """

        #if the last component (y0) of the aMatrix row is not 0 that is y0 has some coefficient then we make it 0
        row0  = out_aMatrix_row[0]
        sub_rows = sub.linear_constraints.get_rows()
        if row0 != 0:

            #index indicates the variables and val indicates the corresponding coefficients 
            ind, val = sub_rows[0].unpack()
            out_aMatrix_row[ind] = out_aMatrix_row[ind] -row0*np.array(val)


        non_zero_indices = [i for i in range(1, len(out_slack_row)) if out_slack_row[i] != 0]
        rows = [sub_rows[i].unpack() for i in non_zero_indices]

        out_slack_non_zero = out_slack_row[non_zero_indices]
        out_const = out_const - np.dot(out_slack_non_zero, self.rhs_const[id][non_zero_indices])
        out_rhs_x = out_rhs_x - np.dot(out_slack_non_zero, self.rhs_tech[id][non_zero_indices, :])

        for i, (ind, val) in enumerate(rows):
            val_array = np.array(val)
            rowi = out_slack_row[non_zero_indices[i]]
            out_aMatrix_row[ind] = out_aMatrix_row[ind] -rowi*val_array
        
        return out_aMatrix_row, out_const, out_rhs_x

    def updateSubproblem(self, out_aMatrix_row, out_const, out_rhs_x, sub, id):

        """
        out_aMatrix_row = numpy array
        """

        ind = list(range(len(out_aMatrix_row)))
        vals = out_aMatrix_row
        sub_cons = sub.linear_constraints.get_num()

        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind, val = vals)], senses = ['L'], rhs = [out_const], names = [f's{sub_cons}'])
        self.rhs_const[id] = np.append(self.rhs_const[id], out_const)

        new_row = np.zeros((1, len(out_rhs_x))) #this a 1\times n matrix
        new_row[0] = out_rhs_x.copy()
        self.rhs_tech[id] = np.concatenate((self.rhs_tech[id], new_row), axis = 0)
        self.subs_ineq_constr[id].append(sub_cons)

    def updateSubproblem_verOrig(self, id, out_aMatrix_row, out_const, out_rhs_x, sub):

        """
        
        """
    
        #first element is y0 which we no longer need in defining the new constraint
        ind = list(range(1, len(out_aMatrix_row)))
        vals = out_aMatrix_row[1:]
        sub_cons = self.sub_constr_count[id]
        sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ind, val = vals)], senses = ['L'], rhs = [out_const], names = [f's{sub_cons}'])
        self.sub_constr_count[id] = sub_cons + 1
        self.rhs_const[id] = np.append(self.rhs_const[id], out_const)
        new_row = np.zeros((1, self.fsv))
        new_row[0] = out_rhs_x
        self.rhs_tech[id] = np.concatenate((self.rhs_tech[id], new_row), axis = 0)
        




        


