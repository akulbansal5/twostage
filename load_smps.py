# reads the core file for loading two-stage stochastic
# problems from SMPS format
# create a mps file by changing .cor to .mps


import os
import numpy as np
dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/stofiles"
lp_path = dir_path + "/lpfiles/"

import copy
import cplex as cpx
from primitives import Subproblem


class Master:

    def __init__(self, name):
        
        self.name = name
        self.cpx_model = None # contains the cplex model
        self.optCuts = 0 # records the number of constraints in the Master problem
    

class loadSMPSdata:

    def __init__(self, name, scenario_dict):
        """
        """

        self.mps_name = name + ".mps"    
        self.cpx_model = cpx.Cplex(self.mps_name)
        self.cpx_model.write(lp_path + "orig.lp")
        self.vars = self.cpx_model.variables.get_names()
        self.constrs = self.cpx_model.linear_constraints.get_names()

        # print('number of constraints: ', self.constrs)
        self.master = None
        
        self.scenVars = None
        self.original_masterVars = None
        self.masterVars = None
        self.secondStageContVars = None
        self.secondStageBinVars = None

        self.masterVarsLen = None


        #has scenario subproblems (as cplex objects) corresponding to scenario id
        self.scenario_dict = scenario_dict
        self.scen_count = None



    def printinfo(self):
        print(f'Constraints: {self.cpx_model.linear_constraints.get_num()}')
        print(f'Variables: {self.cpx_model.variables.get_num()}')
        print(f'Integer variables: {self.cpx_model.variables.get_num_integer()}')
        print(f'Binary variables: {self.cpx_model.variables.get_num_binary()}')
        print(f'Variable names: {self.cpx_model.variables.get_names()}')
        print(f"Variable types: {self.cpx_model.variables.get_types()}")
        print(f"constraint names: {self.cpx_model.linear_constraints.get_names()}")
    
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

    def create_master_problem(self):
        
        """
        creates a master problem based on the given data

        #add variables
        #add constraints
        #add objective

        """

        master = cpx.Cplex()
        master.variables.add(obj = self.cpx_model.objective.get_linear(self.masterVars), types = "B"*self.masterVarsLen, names = self.masterVars)

        #hard coding is done to load the master constraints
        master.linear_constraints.add(lin_expr = [self.cpx_model.linear_constraints.get_rows("c1")], senses = [self.cpx_model.linear_constraints.get_senses("c1")], rhs = [self.cpx_model.linear_constraints.get_rhs("c1")], names = ["c1"]) 
        

        #declare a variable t and obtain a suitable lowerbound
        master.variables.add(obj = [1.0], lb = -2000, types = ["C"], names = ['t'])
        self.masterVars.append('t')
        self.masterVarsLen += 1

        #master problem is declared over here
        self.master = master

        
        return self.master




    def create_scenario_problem(self, idx):
        """
        create scenario problems for given idx
        all variables are declared continuous because in Gade's problem
        the second stage variables are continuous
        """

        scensubprob = cpx.Cplex()

        #add all variables


        #x variables in the first stage (x variables)
        #here the objective is 0 because these are subproblems
        scensubprob.variables.add(obj = [0]*len(self.original_masterVars), types = "C"*len(self.original_masterVars), names = self.original_masterVars)


        #y variables
        scensubprob.variables.add(obj = self.cpx_model.objective.get_linear(self.secondStageBinVars), ub = [1]*len(self.secondStageBinVars), types = "C"*len(self.secondStageBinVars), names= self.secondStageBinVars)
        

        #x variables in second stage (y_0 variables)
        scensubprob.variables.add(obj = self.cpx_model.objective.get_linear(self.secondStageContVars), types = "C"*len(self.secondStageContVars), names= self.secondStageContVars)  
        
        scensubprob.linear_constraints.add(self.cpx_model.linear_constraints.get_rows(self.constrs[1:]), senses = self.cpx_model.linear_constraints.get_senses(self.constrs[1:]), rhs = self.cpx_model.linear_constraints.get_rhs(self.constrs[1:]), names = self.constrs[1:]) 

        return scensubprob


    def create_all_subproblems(self):

        scen_count = len(self.scenario_dict)
        self.scen_count = scen_count
        for idx in range(scen_count):

            self.scenario_dict[idx].cpx_model = self.create_scenario_problem(idx)
            



    def update_subproblems_with_randomness(self):

        for idx in range(self.scen_count):
            #items command lists the (key, value pairs)
            cons_rhs_pairs = list(self.scenario_dict[idx].constraintMap.items())

            self.scenario_dict[idx].cpx_model.linear_constraints.set_rhs(cons_rhs_pairs)


def loadScenarios(filename):
    """
    returns: 
    
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
    idx = 0
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


    return scenario_dict, np.array(prob)


def loadScenarios_smkp(filename):
    """
    returns: 
    
    scenario_dict (dict): keys are scenario ids
    values: objects of Subproblem class

    Subproblem class essentially represents the scenarios with following attributes:
        probability
        constraintMap (dict): keys are variable names and values are objective coefficients (float)

    smkp: loading scenarios specifically for smkp instances
    """


    scenario_dict = {}
    with open(filename, "r") as f:
        data = f.readlines()
    prob = []
    idx = 0
    for line in data:
        words = line.split()
        if len(words) > 2:
            if words[0] == "SC":
                scen = Subproblem(idx)

                #we associate an idx to it
                scenario_dict[idx] = scen
                scen.probability = float(words[3])
                prob.append(scen.probability)
                scen.varObjMap = {}
                idx += 1
            else:
                scen.varObjMap[words[0]] = float(words[2])
    
    return scenario_dict, np.array(prob)



def loadScenarios_rui(map, prob):
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

    return scenario_dict, np.array(prob)


def main(filename):

    scenario_dict = loadScenarios(filename + ".sto")
    dataObj = loadSMPSdata(filename, scenario_dict)
    

    #partitions the variables in the SSLP problem
    dataObj.filterVars()

    #creates master problem
    master_problem = dataObj.create_master_problem()
    m_obj = Master(filename)
    m_obj.cpx_model = master_problem


    dataObj.create_all_subproblems()

    dataObj.update_subproblems_with_randomness()

    return m_obj, scenario_dict


#master_object, scenario_dict = main("sslp_5_25_50")

