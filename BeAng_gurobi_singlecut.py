# _______________________________________________________
#BeAng: Benders decompostion with Angulo's alternate cutting criterion
#       and Laporte and Louveaux's integer L-shaped cuts
#       uses Cplex (without callbacks)
# _____________________
# 
#BeAng_gurobi: Same as BeAng final but changes the solver from cplex to Gurobi
# ____________________________________

import os
import csv
import time
import numpy as np
import cplex as cpx
import gurobipy as gp
from sslpInstance import GlobalModelSSLP


# from sslpInstance import GlobalModel
from benders_root import benders_standard

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/stofiles/"
lp_path = dir_path + "/lpfiles/"
out_path = dir_path + "/output/"


def specialSum(x, tol = 1e-4):

    """
    x: master solution
    returns:
        ones: number of ones in x
        x_0: indices which have 0
        x_1: indices which have 1
    """

    x_non_0 = []
    x_0     = []
    ones    = 0


    for i, i_val in enumerate(x):
        if i_val > tol:
            x_non_0.append(i)
            ones += 1
        else:
            x_0.append(i)

    return ones, x_0, x_non_0

def specialSum_gurobi(x, state_var_objects, tol = 1e-4):

    """
    x: master solution
    state_var_objects: gurobi object variables
    returns:
        ones: number of ones in x
        x_0: indices which have 0
        x_1: indices which have 1
    """

    x_non_0 = []
    x_0 = []
    ones = 0


    for i, i_val in enumerate(x):
        if i_val > tol:
            x_non_0.append(state_var_objects[i])
            ones += 1
        else:
            x_0.append(state_var_objects[i])

    return ones, x_0, x_non_0




def benders_angulo(global_model, max_iteration = 1000, max_time = 3600, tolerance = 1e-6, \
    gap_tolerance = 1e-3, earlyGap = 1e-2, csv_name = 'benders_standard_seg.csv',\
        pFlag = False, earlypFlag = False, ben_count = 0, ben_lb = None, ben_total_time = 0, min_time = 60):
    
    """
    Input:
        global_model (GlobalModel): Contains information regarding master and subproblems
                                    defined in files like sslpInstance.py smkpInstance.
        max_iteration (int): maximum number of iterations of the algorithm
        tolerance (float):  used to avoid precision errors in enforcing Benders cuts
        gap_tolerance (float): gap = abs(ub - lb)/abs(ub) and if gap < gap_tolerance code terminates
        earlyGap: used for determining when to recrod the output
        pFlag: print flag -> prints a few extra details if true
        ben_count: number of benders cuts
        min_time: minimum time allowed in this phases


    Algorithm:
        Applies Angulo et al's Benders decomposition which uses ineger L-shaped cuts
    Returns:
        Updated master problem with benders cuts and integer L-sahaped cuts from upperbounding step
    """

    print("-----------------------------------Angulo's alternating cut----------------------------------------")
    start_time = time.time()
    sub_idxs = global_model.sub_idxs #subproblem ids
    count = 0
    ub = float('inf')
    lb = float('-inf')
    gap = float('inf')


    nodes = 0
    cuts = 0
    angulo_cuts = 0
    gfc_cuts = 0
    ben_cuts = 0
    
    
    master_solve_time = 0
    sub_solve_time = 0
    angulo_obtain_time = 0
    updating_subs_time = 0
    sub_solve_time_mip = 0
    mip_ub_time = 0
    mip_ub_iters = 0
    dual_info_time = 0
    benders_cut_time = 0
    isFrac_time = 0
    cutnodeinfoTime = 0
    gfc_obtain_time= 0


    bInfoTime_t     = 0
    sind_time_t     = 0
    arow_time_t     = 0
    slrow_time_t    = 0
    rhs_info_time_t = 0
    trans_time_t     = 0
    gfc_step_time_t  = 0
    update_time_t    = 0
    fbsistime_t      = 0
    getSolTime_t     = 0
    getPosTime_t     = 0
    loopTime_t         = 0
    fbsis_time_t       = 0
    dual_access_time   = 0
    
    ### where is the problem relaxed when getting benders cuts?
    
    # binaryInd, integerInd = relax_integer_problem(master)

    masterObj             = global_model.masterObj
    #master               = masterObj.cpx_model
    master                = masterObj.grb_model   #### GUROBI ###
    
    # master_vars_in_stage2 = masterObj.vars_in_stage2    #names of variables in master problem - 1 (the last one which is t or theta)                               # variables in stage 2 problem                                      
    master_vars_in_stage2 = masterObj.state_vars_grb      #list of gurobi variable objects, does not include the t variable (or theta variable)


    master_cols_in_stage2 = masterObj.cols_in_stage2    #number of columns in master problem - 1
    sub_dict              = global_model.scenario_dict                #Access the scenario dictionary 
    # masterSolsIP = np.zeros((1, master_cols-1)) - 1                 #master solutions to integer program
    # masterSolsLP = np.zeros((1, master_cols-1)) - 1                 #master solutions to linear program

    total_time = time.time() - start_time

    while count < max_iteration and gap > gap_tolerance and total_time < max_time:  
    
        iter_start_time    = time.time()
        time_counter       = time.time()    
        # master.parameters.timelimit.set(max(max_time - total_time, 60))

        master.setParam('TimeLimit', max(max_time - total_time, min_time))        # Time limit in seconds
        # master.solve()        
        master.optimize()                                              
        master_solve_time += time.time() - time_counter

        # time_counter = time.time()
        # bbnodes = master.solution.MIP.get_incumbent_node()
        # mipcuts = masterObj.count_cuts()
        # cutnodeinfoTime += time.time() - time_counter

        # nodes += bbnodes
        # cuts  += mipcuts

        if master.status == gp.GRB.OPTIMAL:
            
            # Retrieve the objective value
            # objective_value = master.objVal
            # print("Optimal objective value:", objective_value)

            lb = master.objVal
        else:
            print("No optimal solution found for master problem in given time.")

        # lb = master.solution.get_objective_value()
        # t = masterObj.get_solution()


        t         = masterObj.get_futurecost_gurobi() #also stores the state vectors
        
        cutFromLP = False
        
        #updating subsolve time
        time_counter = time.time()

        #UPDATE this because the model is now a gurobi model
        global_model.updateSubproblems_withIncmbt_grb()  
        
        
        updating_subs_time += time.time() - time_counter
        local_obj = 0
        
        for id in sub_idxs:

            sub = sub_dict[id]
            time_counter = time.time()                              
            
            # sub.cpx_model.solve()    
            sub.grb_model.optimize()
            
            #first we solve linear program
            sub_solve_time += time.time() - time_counter            

            # sub_obj = sub.solution.get_objective_value()
            sub_obj = sub.grb_model.objVal

            sub.sub_obj = sub_obj
            local_obj += sub_obj*sub.probability

        # only add a benders cut if the local objective from LP relaxation > t
        global_coeff = np.zeros(master_cols_in_stage2) #in this case all variab
        global_const = 0

        
        if local_obj > t + tolerance:          #only adds standard Benders cut
            
            for id in sub_idxs:

                sub = sub_dict[id]
                time_counter = time.time()
                # sub_duals = np.array(constr.Pi)
                dual_vector = sub.grb_model.getAttr("Pi", sub.grb_model_constrs)
                sub_duals   = np.array(dual_vector)

                # sub_duals = np.array(sub.solution.get_dual_values())
                dual_access_time += time.time() - time_counter


                const            = np.dot(sub_duals, sub.rhs_const)
                coeff            = np.dot(sub_duals, sub.rhs_tech_row)
                sub.coeffs       = coeff
                sub.rhs_constant = const
                global_coeff     = global_coeff + sub.probability*sub.coeffs
                global_const     = global_const + sub.probability*sub.rhs_constant
                


            
            # master.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ['t'] + master_vars_in_stage2, val = [1] + list(-global_coeff))], senses = ['G'], rhs = [global_const], names = [f'ben_{count}'])        
            expr_vars   = [masterObj.tVar_grb] + master_vars_in_stage2
            expr_coeffs = [1] + list(-global_coeff)
            lin_expr = gp.LinExpr(expr_coeffs, expr_vars)
            master.addConstr(lin_expr >= global_const, name = f'ben_{count}')


            ben_cuts += 1
            cutFromLP = True

        
        #if x not in V and Benders cut is not added from LP-subs
        if cutFromLP == False:                                  
            
            # masterSolsIP = np.vstack((masterSolsIP , x))                                                 
            # global_coeff = np.zeros(master_cols_in_stage2)              
            # global_const = 0
            local_obj = 0

            ### UPDATE THIS TO GUROBI ###

            global_model.updateSubproblems_withIncmbt_grb(isMIP = True)


            #solve all the mip-subproblems and enforce laporte as required
            for id in sub_idxs:
                sub = sub_dict[id]
                time_counter = time.time()
                # sub.cpx_model_mip.solve()
                sub.grb_model_mip.optimize()
                                              
                sub_solve_time_mip += time.time() - time_counter
                # sub.curr_obj = sub.solution_mip.get_objective_value()
                sub.curr_obj = sub.grb_model_mip.objVal

                # sub.prev_solsIP[self.mSolIndIP] = sub.curr_obj                          # solution to integer programming problem
                local_obj += sub.probability*sub.curr_obj


            # self.localObjs.append(local_obj)
            ub = min(ub, local_obj + lb - t)

            if local_obj > t + tolerance:

                time_counter = time.time()
                gfc_cuts += 1
                ones, x_0, x_non_0 = specialSum_gurobi(masterObj.curr_sol, masterObj.state_vars_grb)
                coeff = local_obj - global_model.global_lower_bd
                rhs = -coeff*ones + local_obj

                intL_vars = [masterObj.tVar_grb] + x_non_0 + x_0
                intL_coeff = [1] + [-coeff]*len(x_non_0) + [coeff]*len(x_0)
                intL_expr = gp.LinExpr(intL_coeff, intL_vars)

                # master.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ['t'] + x_non_0 + x_0, val = [1] + [-coeff]*len(x_non_0) + [coeff]*len(x_0))], senses = ['G'], rhs = [rhs], names = [f"lap_{count}"])            
                # master.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = intL_vars, val = [1] + [-coeff]*len(x_non_0) + [coeff]*len(x_0))], senses = ['G'], rhs = [rhs], names = [f"lap_{count}"])            
                master.addConstr(intL_expr >= rhs, name = f"lap_{count}")
                
                gfc_obtain_time += time.time() - time_counter

                
                # print("gdc added")


        count += 1
        if ub != float('inf'):
            gap = abs(ub - lb)/(abs(ub) + 1e-7)                                                     

        
        if pFlag:
            print(f"Angulo Iter: {count}, t: {t}, lb: {lb}, ub: {ub}, gap: {gap}")


        if earlypFlag:
            if gap < earlyGap:
                gade_total_time = time.time() - start_time
                row = ['early', count, ben_count, cuts, nodes, gfc_cuts, ben_cuts, ben_lb, lb, ub, gap, ben_total_time, ben_total_time+ gade_total_time, master_solve_time, sub_solve_time, sub_solve_time_mip, gfc_obtain_time, updating_subs_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, arow_time_t, sind_time_t, slrow_time_t, update_time_t, getSolTime_t, getPosTime_t, loopTime_t]
                with open(csv_name, mode = 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                earlyGap = earlyGap/10

        total_time += time.time() - iter_start_time
        
    gade_total_time = time.time() - start_time


    return count, cuts, nodes, gfc_cuts, lb, ub, gap, gade_total_time, master_solve_time, ben_cuts, sub_solve_time, gfc_obtain_time, updating_subs_time, sub_solve_time_mip, dual_info_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, sind_time_t, arow_time_t, slrow_time_t, rhs_info_time_t, trans_time_t, gfc_step_time_t, update_time_t, getSolTime_t, getPosTime_t, loopTime_t

def global_runs_BeAng(fileofobjects, gade_iter = 10000, total_time = 120, gade_tol = 1e-3, ben_tol = 1e-3, ben_iter = 10000, ben_time = 3600, csvname = "benders_angulo_output.csv", isEqual = True, pFlag = False, earlyGap = 1e-1, earlypFlag = False):

    """
    
    """

    with open(csvname, mode = 'w+') as f:

        fieldnames = ["name", "iters","ben-iters", "m-cuts", "m-nodes", "lap-cuts", "ben-cuts", "ben_lb", "lb", "ub", "gap",  "ben_time", "total_time", "m-time", "subLP-time", "subMIp-time", "g-time", "usub-time", "dualac_time", "mcut_time", "isFrac_time", "mip-ub_iters", "binfo_time", "fbsisTime", "arowTime", "soRowTime", "slRowTime", "gfcUpdTime", "getSolTime", "getPosTime", "getloopTime"]
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        
    for gm in fileofobjects:

        print("FILENAME is: ", gm.name)
        name = gm.name
        # gm = GlobalModel(data_path, name)
        gm.main(isEqual = isEqual)

        #this is the main function that will help convert things to gurobi
        gm.convert_problem_to_grb()
        
        # ben_count, ben_lb, ub, gap, ben_total_time, master_solve_time, sub_solve_time, updating_subs_time, dual_info_time, benders_cut_time = benders_standard(gm, gap_tolerance=ben_tol, max_iteration=ben_iter, max_time=ben_time, pFlag = pFlag)        
        

        ben_total_time = 0
        ben_count = 0
        ben_lb = 0


        count, cuts, nodes, gfc_cuts, lb, ub, gap, gade_total_time, master_solve_time, ben_cuts, sub_solve_time, gfc_obtain_time, updating_subs_time, sub_solve_time_mip, dual_info_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, sind_time_t, arow_time_t, slrow_time_t, rhs_info_time_t, trans_time_t, gfc_step_time_t, \
            update_time_t, getSolTime_t, getPosTime_t, loopTime_t \
                = benders_angulo(gm, max_iteration= gade_iter, max_time = total_time - ben_total_time, \
                    gap_tolerance=gade_tol, pFlag=pFlag, earlyGap=earlyGap, earlypFlag=earlypFlag, ben_count=ben_count, ben_lb=ben_lb, ben_total_time=ben_total_time, csv_name=csvname)

        row = [name, count, ben_count, cuts, nodes, gfc_cuts, ben_cuts, ben_lb, lb, ub, gap, ben_total_time, ben_total_time+ gade_total_time, master_solve_time, sub_solve_time, sub_solve_time_mip, gfc_obtain_time, updating_subs_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, arow_time_t, sind_time_t, slrow_time_t, update_time_t, getSolTime_t, getPosTime_t, loopTime_t]
        

        
        with open(csvname, mode = 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("-----")
    



filenames      = ['sslp_5_25_50', 'sslp_5_25_100'] + ['sslp_10_50_50', 'sslp_10_50_100', 'sslp_10_50_500', 'sslp_10_50_1000', 'sslp_10_50_2000', 'sslp_15_45_5', 'sslp_15_45_10', 'sslp_15_45_15']
filenames1     = ['sslp_5_25_50']
fileofobjects  = [GlobalModelSSLP(data_path, name) for name in filenames1]  
global_runs_BeAng(fileofobjects, gade_iter = 10000, total_time = 600, gade_tol = 1e-4, ben_tol = 1e-3, ben_iter = 10000, ben_time = 3600, csvname = "benders_angulo_output.csv", isEqual = True, pFlag = True, earlyGap = 1e-1, earlypFlag = False)
