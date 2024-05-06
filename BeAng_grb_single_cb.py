# _______________________________________________________
#BeAng: Benders decompostion with Angulo's alternate cutting criterion
#       and Laporte and Louveaux's integer L-shaped cuts
#       uses Cplex (without callbacks)
# _____________________
# 
#BeAng_gurobi: Same as BeAng final but changes the solver from cplex to Gurobi
#BeAng_grb_single_cb: Benders and Integer LShaped cut added in single-cut and callback fashion
#single cut means that we maintain a single variable in the master problem for approximating future cost
#
#  ____________________________________



import os
import csv
import time
import numpy as np
import cplex as cpx
import gurobipy as gp
from gurobipy import GRB
from sslpInstance import GlobalModelSSLP
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

def specialSum_gurobi(x, state_var_objects, tol = 0.2):

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

def benders_angulo(global_model, max_iteration = 1000, max_time = 3600, tolerance = 1e-8, \
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
        Applies Angulo et al's Benders decomposition which uses ineger L-shaped cuts (multicut with callbacks)
    Returns:
        Updated master problem with benders cuts and integer L-sahaped cuts from upperbounding step
    """

    print("-----------------------------------Angulo's alternating cut----------------------------------------")
    start_time = time.time()
    sub_idxs = global_model.sub_idxs #subproblem ids
    

    # global LB
    # global UB
    # global gfc_cuts
    # global ben_cuts
    # global gap
    # global count

    count = 0
    UB = float('inf')
    LB = float('-inf')
    gap = float('inf')
    master_gap = None
    scenProb = global_model.scenProb

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
    cut_store = {id:[] for id in sub_idxs}
    constr_sol_store = {}
    constr_subobj_store = {}
    


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
    


    masterObj             = global_model.masterObj
    master                = masterObj.grb_model   
    master.setParam('OutputFlag', True)
    master_vars_in_stage2 = masterObj.state_vars_grb                         #list of gurobi variable objects, does not include the t variable (or theta variable)
    master_cols_in_stage2 = masterObj.cols_in_stage2  
    sub_dict              = global_model.scenario_dict                       #Access the scenario dictionary 

    total_time = time.time() - start_time


    def cb(model, where):
        if where == GRB.Callback.MIPSOL:
            
            # model._xVec_Curr = xVec

            LB        = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
            model._LB = LB
            UB        = model._UB

            if UB != float('inf'):
                model._gap = abs(UB - LB)/(abs(UB) + 1e-7)
            if model._iters > max_iteration:
                model._cond = 0
                model.terminate()
            if model._gap < gap_tolerance:
                model._cond = 1
                model.terminate()

            tCost    = model.cbGetSolution(masterObj.tVar_grb)
            xVec     = model.cbGetSolution(masterObj.state_vars_grb)
            xVec_np  = np.array(xVec)
            global_model.updateSubproblems_withIncmbt_grb_callback(xVec_np)
            # subCosts             = np.zeros(global_model.Nscen)
            
            #solve all subproblems as LP
            LPFutureCosts = 0
            for id in sub_idxs:
                sub = sub_dict[id]
                sub.grb_model.optimize()
                subObj = sub.grb_model.objVal
                LPFutureCosts = LPFutureCosts+ subObj*sub.probability #TODO: make this a numpy operation
            
            gcoeff = np.zeros(master_cols_in_stage2) #in this case all variab
            gconst = 0
            if LPFutureCosts > tCost + tolerance:
                for id in sub_idxs:
                    sub = sub_dict[id]
                    sub_duals    = np.array(sub.grb_model.getAttr("Pi", sub.grb_model_constrs))
                    rhs_cut      = np.dot(sub_duals, sub.rhs_const)
                    coeffs       = np.dot(sub_duals, sub.rhs_tech_row)
                    gcoeff       = gcoeff + sub.probability*coeffs
                    gconst       = gconst + sub.probability*rhs_cut

                expr_vars   = [masterObj.tVar_grb] + master_vars_in_stage2
                expr_coeffs = [1] + list(-gcoeff)
                lin_expr    = gp.LinExpr(expr_coeffs, expr_vars)
                model.cbLazy(lin_expr >= gconst)
                model._ben_cuts   = model._ben_cuts + 1
            else:
                MIPFutureCosts = 0
                global_model.updateSubproblems_withIncmbt_grb_callback(xVec_np, isMIP = True)
                for id in sub_idxs:
                    sub = sub_dict[id]
                    sub.grb_model_mip.optimize()
                    subObjMIP      = sub.grb_model_mip.objVal
                    MIPFutureCosts = MIPFutureCosts + sub.probability*subObjMIP
                masterObjective = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                model._UB = min(UB, masterObjective - tCost + MIPFutureCosts)
                if MIPFutureCosts > tCost + tolerance:
                    ones, x_0, x_non_0 = specialSum_gurobi(xVec, masterObj.state_vars_grb)
                    coeff              = MIPFutureCosts - global_model.global_lower_bd
                    rhs                = -coeff*ones + MIPFutureCosts
                    intL_vars          = [masterObj.tVar_grb] + x_non_0 + x_0
                    intL_coeff         = [1] + [-coeff]*len(x_non_0) + [coeff]*len(x_0)
                    intL_expr          = gp.LinExpr(intL_coeff, intL_vars)

                    # model._MIPFutureCosts = MIPFutureCosts
                    # model._coeff = coeff
                    # model._rhs   = rhs
                    # model._gLB   = global_model.global_lower_bd
                    # model._xVec = xVec
                    # model._mCost = masterObjective - tCost

                    model.cbLazy(intL_expr >= rhs)
                    model._gfc_cuts = model._gfc_cuts + 1

            model._iters = model._iters + 1

            # if UB != float('inf'):
            #     gap = abs(UB - LB)/(abs(UB) + 1e-7)
            # if global_model.iters > max_iteration or gap < gap_tolerance:
            #     master._UB = UB
            #     master._LB = LB
            #     master._gap = gap
            #     model.terminate()

    master.Params.lazyConstraints = 1
    master.Params.timeLimit = max(max_time - total_time, min_time)
    try:
        master._UB       = UB
        master._LB       = LB
        master._gap      = gap
        master._ben_cuts = ben_cuts
        master._gfc_cuts = gfc_cuts
        master._iters    = count
        master._cond     = -1
        # master._xVec_Curr = []
        # master._xVec     = []
        # master._MIPFutureCosts = -1
        # master._gLB           = None
        # master._rhs           = None
        # master._mCost         = None
        # master._coeff         = None

        master.optimize(cb)

    except gp.GurobiError as e:
        print(f"Error code {e.rrno}: {e}")

    # print(f"global lower bound: {master._gLB}")
    # print(f"rhs of the cut: {master._rhs}")
    # print(f"Future cost: {master._MIPFutureCosts}")
    # print(f"Coefficient in the cut: {master._coeff}")
    # print(f"xVector of the cut: {master._xVec}")
    # print(f"Cost from the master problem: {master._mCost}")
    # print(f"last xVector encountered: {master._xVec_Curr}")

    gade_total_time = time.time() - start_time
    gfc_cuts = master._gfc_cuts
    ben_cuts = master._ben_cuts
    if master._UB < float('inf'):
        UB       = master._UB
    elif master.MIPGap < gap_tolerance:
        UB       = master.objVal

    LB       = master.objVal
    gap      = abs(UB - LB)/(abs(UB) + 1e-7)
    count    = master._iters
    cond     = master._cond

    print(f"iters = {count}, LB = {LB}, UB = {UB}, gap = {gap}, cond: {cond}, gCuts = {gfc_cuts}, bCuts = {ben_cuts}")
    # print(master._UB, master._LB, master._LB2, LB, master._gap)

    return count, cuts, nodes, gfc_cuts, LB, UB, gap, gade_total_time, master_solve_time, ben_cuts, sub_solve_time, gfc_obtain_time, updating_subs_time, sub_solve_time_mip, dual_info_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, sind_time_t, arow_time_t, slrow_time_t, rhs_info_time_t, trans_time_t, gfc_step_time_t, update_time_t, getSolTime_t, getPosTime_t, loopTime_t

def global_runs_BeAng(fileofobjects, gade_iter = 10000, total_time = 120, gade_tol = 1e-3, ben_tol = 1e-3, ben_iter = 10000, ben_time = 3600, csvname = "benders_angulo_output.csv", isEqual = True, addCopy = False, pFlag = False, earlyGap = 1e-1, earlypFlag = False, isRuiInst = False):

    """
    
    """

    with open(csvname, mode = 'a+') as f:

        fieldnames = ["name", "iters","ben-iters", "m-cuts", "m-nodes", "lap-cuts", "ben-cuts", "ben_lb", "lb", "ub", "gap",  "ben_time", "total_time", "m-time", "subLP-time", "subMIp-time", "g-time", "usub-time", "dualac_time", "mcut_time", "isFrac_time", "mip-ub_iters", "binfo_time", "fbsisTime", "arowTime", "soRowTime", "slRowTime", "gfcUpdTime", "getSolTime", "getPosTime", "getloopTime"]
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        
    for gm in fileofobjects:

        print("FILENAME is: ", gm.name)
        name = gm.name
        # gm = GlobalModel(data_path, name)
        gm.main(isEqual = isEqual, addCopy = addCopy, isRuiInst = isRuiInst)

        #this is the main function that will help convert things to gurobi
        gm.convert_problem_to_grb(isMultiCut = False)

        gm.masterObj.grb_model.write(lp_path + f"Master_{gm.name}.lp")
        sub            = gm.scenario_dict[0]
        Nfv = gm.masterObj.cols_in_stage2
        sols_in_stage2 = np.ones(Nfv)

        rhs_info = list(sub.rhs_const + np.dot(sub.rhs_tech_row, sols_in_stage2))
        sub.grb_model.setAttr("RHS", sub.grb_model_constrs, rhs_info)
        sub.grb_model.write(lp_path + f"Sub_{gm.name}.lp")

        # master_cols_in_stage2 = gm.masterObj.cols_in_stage2

        # xVec0 = np.zeros(master_cols_in_stage2)
        # gm.updateSubproblems_withIncmbt_grb_callback(xVec0, isMIP = True)
        # idx   = 0
        # gm.scenario_dict[idx].grb_model_mip.write(lp_path + f"subMIP_{idx}_0.lp")


        # xVec1 = np.ones(master_cols_in_stage2) + np.array([-1,-1, 0, 0, 0])
        # gm.updateSubproblems_withIncmbt_grb_callback(xVec0, isMIP = True)
        # idx = 1
        # gm.scenario_dict[idx].grb_model_mip.write(lp_path + f"subMIP_{idx}_1.lp")
        # print("SubMIPs published")


        # ben_count, ben_lb, ub, gap, ben_total_time, master_solve_time, sub_solve_time, updating_subs_time, dual_info_time, benders_cut_time = benders_standard(gm, gap_tolerance=ben_tol, max_iteration=ben_iter, max_time=ben_time, pFlag = pFlag)        
        

        ben_total_time = 0
        ben_count = 0
        ben_lb = 0


        count, cuts, nodes, gfc_cuts, lb, ub, gap, gade_total_time, master_solve_time, ben_cuts, sub_solve_time, gfc_obtain_time, updating_subs_time, sub_solve_time_mip, dual_info_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, sind_time_t, arow_time_t, slrow_time_t, rhs_info_time_t, trans_time_t, gfc_step_time_t, \
            update_time_t, getSolTime_t, getPosTime_t, loopTime_t \
                = benders_angulo(gm, max_iteration= gade_iter, max_time = total_time - ben_total_time, \
                    gap_tolerance=gade_tol, pFlag=pFlag, earlyGap=earlyGap, earlypFlag=earlypFlag, ben_count=ben_count, ben_lb=ben_lb, ben_total_time=ben_total_time, csv_name=csvname)

        row = [name, count, ben_count, cuts, nodes, gfc_cuts, ben_cuts, ben_lb, lb, ub, gap, ben_total_time, ben_total_time+ gade_total_time, master_solve_time, sub_solve_time, sub_solve_time_mip, gfc_obtain_time, updating_subs_time, dual_access_time, benders_cut_time, isFrac_time, mip_ub_iters, bInfoTime_t, fbsis_time_t, arow_time_t, sind_time_t, slrow_time_t, update_time_t, getSolTime_t, getPosTime_t, loopTime_t]
        
        # print("fraction of time in gfc equals: ", gfc_obtain_time/total_time, "total time: ", total_time, "gfc_cuts: ", gfc_cuts, "mS time: ", master_solve_time, 'sS time: ', sub_solve_time, "dual time: ", dual_info_time)

        # print(f"bInfoTime_t, {bInfoTime_t}, f_basis_timet: {fbsis_time_t}, sind_time_t, {sind_time_t}, arow_time_t, {arow_time_t}, slrow_time_t, {slrow_time_t}, rhs_info_time_t, {rhs_info_time_t}, trans_time_t, {trans_time_t}, gfc_step_time_t, {gfc_step_time_t}, update_time_t, {update_time_t}")
        
        
        with open(csvname, mode = 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print("-----")
    

filenames      = ['sslp_5_25_50', 'sslp_5_25_100'] + ['sslp_10_50_50', 'sslp_10_50_100', 'sslp_10_50_500', 'sslp_10_50_1000', 'sslp_10_50_2000', 'sslp_15_45_5', 'sslp_15_45_10', 'sslp_15_45_15']
filetail       = ['sslp_15_45_5', 'sslp_15_45_10', 'sslp_15_45_15', 'sslp_10_50_2000']
filenames1     = ['sslp_5_25_50']
ruifiles       = ['rui_sslp1_20_100_50']
rui_inst       = ['rui_sslp1_20_100_50', 'rui_sslp2_20_100_50', 'rui_sslp3_20_100_50', 'rui_sslp1_20_100_200', 'rui_sslp2_20_100_200', 'rui_sslp3_20_100_200', 'rui_sslp1_30_70_50', 'rui_sslp2_30_70_50', 'rui_sslp3_30_70_50', 'rui_sslp1_30_70_200', 'rui_sslp2_30_70_200', 'rui_sslp3_30_70_200', 'rui_sslp1_40_50_50', 'rui_sslp2_40_50_50', 'rui_sslp3_40_50_50', 'rui_sslp1_40_50_200', 'rui_sslp2_40_50_200', 'rui_sslp3_40_50_200', 'rui_sslp1_50_40_50', 'rui_sslp2_50_40_50', 'rui_sslp3_50_40_50', 'rui_sslp1_50_40_200', 'rui_sslp2_50_40_200', 'rui_sslp3_50_40_200']
rui_inst_test       = ['rui_sslp3_20_100_50']
fileofobjects  = [GlobalModelSSLP(data_path, name) for name in rui_inst_test]
global_runs_BeAng(fileofobjects, gade_iter = 3000, total_time = 3600, gade_tol = 1e-4, ben_tol = 1e-3, ben_iter = 10000, ben_time = 3600, csvname = out_path+"benders_angulo_output_rUI_Apr28_24.csv", isEqual = True, addCopy = False, pFlag = True, earlyGap = 1e-1, earlypFlag = False, isRuiInst = True)


