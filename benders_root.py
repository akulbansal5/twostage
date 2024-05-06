
import time
import numpy as np
import cplex as cpx




def benders_standard(global_model, max_iteration = 1000, max_time = 3600, tolerance = 1e-6, gap_tolerance = 1e-2, pFlag = True):
    
    """
    Input:
        masterObj (Master class): Object of master class
        subObj (GSubproblem class): Object of GSubproblem (read Global Subproblem) class
        max_iteration (int): maximum number of iterations of the algorithm
        tolerance (float):  used to avoid precision errors in enforcing Benders cuts
        gap_tolerance (float): gap = abs(ub - lb)/abs(ub) and if gap < gap_tolerance code terminates

    Algorithm:
        Applies Benders decomposition to relaxed master and relaxed subproblems.

    Returns:
        Updated master problem with benders cuts added to it.
    """

    
    masterObj = global_model.masterObj

    count = 0                                                                           
    ub    = float('inf')
    lb    = float('-inf')
    gap   = float('inf')
    total_time = 0
    
    master_solve_time  = 0
    sub_solve_time     = 0
    updating_subs_time = 0
    dual_info_time     = 0
    benders_cut_time   = 0

    master                = masterObj.cpx_model                                                                           
    binaryInd, integerInd = masterObj.relax_integer_problem()                                 # relax the integer program
    master_vars_in_stage2 = masterObj.vars_in_stage2                                          # name of variables in stage 2                        
    master_cols_in_stage2 = masterObj.cols_in_stage2                                          # number of columns in stage 2                           
    sub_idxs              = global_model.sub_idxs                                             # subproblem ids in the problem
    sub_dict              = global_model.scenario_dict                                           # subproblem dictionary in the problem
    
    

    while count < max_iteration and gap > gap_tolerance and total_time < max_time:    

        iter_start_time = time.time()
        time_counter = time.time()
        master.solve()
        master_solve_time += time.time() - time_counter
        lb = master.solution.get_objective_value()
        t = masterObj.get_solution()            #entire solution vector, vector that affects stage 2, future cost to be approximated
        time_counter = time.time()
        global_model.updateSubproblems_withIncmbt(isMIP = False)    #update all subproblems with incumbent solution      
        updating_subs_time += time.time() - time_counter
        local_obj = 0 
        
        #solve the subproblems
        for id in sub_idxs:                                                             
            sub = sub_dict[id]
            time_counter = time.time()                                                  
            sub.cpx_model.solve()
            sub_solve_time += time.time() - time_counter 
            
            sub_obj = sub.solution.get_objective_value()
            local_obj += sub_obj*sub.probability

            time_counter = time.time()
            sub_duals    = np.array(sub.solution.get_dual_values())
            const        = np.dot(sub_duals, sub.rhs_const)
            coeff        = np.dot(sub_duals, sub.rhs_tech_row)
            sub.coeffs   = coeff
            sub.rhs_constant = const 
            dual_info_time  += time.time() - time_counter
    
        
        ub = min(ub, local_obj + lb - t)
        time_counter = time.time()
        
        if local_obj > t + tolerance:

            global_coeff = np.zeros(master_cols_in_stage2)
            global_const = 0

            for id in sub_idxs:
                sub = sub_dict[id]
                global_coeff = global_coeff + sub.probability*sub.coeffs
                global_const = global_const + sub.probability*sub.rhs_constant
 
            #benders cut
            master.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ['t'] + master_vars_in_stage2, val = [1] + list(-global_coeff))], senses = ['G'], rhs = [global_const], names = [f'bend_{count}'])
        
        benders_cut_time += time.time() - time_counter

        count += 1
        iter_time = time.time() - iter_start_time
        total_time += iter_time

        if pFlag == True:
            print(f"Benders Root Iter: {count}, t: {t}, lb: {lb}, ub: {ub}, gap: {gap}")

        gap = abs(ub - lb)/(abs(ub)+1e-7)
    
    #convert the problem to integer form
    masterObj.revert_back_to_mip(binaryInd, integerInd)
    print(f"-- Final Benders Gap: {gap}")
    return count, lb, ub, gap, total_time, master_solve_time, sub_solve_time, updating_subs_time, dual_info_time, benders_cut_time
