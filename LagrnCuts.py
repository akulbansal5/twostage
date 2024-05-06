"""
In this file we compute the Lagrangian cuts
"""

import os
import csv
import numpy as np
import gurobipy as gp

dir_path = os.path.dirname(os.path.realpath(__file__))
data_path = dir_path + "/stofiles/"
lp_path = dir_path + "/lpfiles/"
out_path = dir_path + "/output/"


def stepSize(gnorm, iter, epsilon):

    """
    
    """
    

    stSize = 20 / np.sqrt(iter+1)

    return stSize / (gnorm + epsilon) 


def stepSize_ver2(ub, newLB, gnorm, iter):

    eps = 2/np.sqrt(iter+1)

    return eps*(ub-newLB)/(gnorm**2+1e-4)


def stepSize_ver3(ub, newLB, gnorm, iter):

    eps = 5/(iter+1)

    return eps*(ub-newLB)

def stepSize_ver4(ub, newLB, gnorm, iter):

    # eps = 5/(iter+1)

    return 1/(iter+1)




def update_obj_coeff_grb(newCoefficients, model, varObjects):

    model.setAttr("Obj", varObjects, newCoefficients)

def LGDualSolver_ver1(pi, maxIter, gapTol, normTol, model, copyVarObjects, forwardSol, subLB = None, subUB = None, subMIP_model = None):

    """
    pi (list or numpy array): starting value of pi
    maxIter (int): maximum number of iterations for solving the Lagrangian dual problem
    gapTol (float):  tolerance in the gap for terminating the subgradient algorithm
    normTol (float): tolernace in the gradient norm for terminating the subgradient algorithm
    model(gurobi model): gurobi model for the lagrangian subproblem
    copyVarObjects (variable objects from gurobi): The variable objects corr to copy variables in the Lagrangian subproblem
    forwardSol (numpy array): solution passed from the previous stage
    subLB (float): lower bound on the subproblem for given forward solution obtained from LP relaxation
    subUB (float): upper bound on the subproblem for given forward solution obtained from exact MIP
    subMIP_model (gurobi model): 
    returns:
    """


    gnorm = 1
    iter  = 0
    infs_unbd_codes = [3,4,5]
    gap = abs((subUB - subLB)/(subUB + 1e-7))
    orig_gap = gap
    # gapTol = max(min(orig_gap/2, gapTol),1*1e-3)
    gapTol = gapTol

    exitWithoutLagObj = True

    while iter < maxIter and gap > gapTol and gnorm > normTol:

        exitWithoutLagObj = False
        #update the coefficients
        model.setAttr("Obj", copyVarObjects, -pi)

        #solve the Lagrangin subproblem with given choice of lagrangian coefficients pi
        model.optimize()

        

        

        #throw error if unable to solve 
        if model.status in infs_unbd_codes:
            model.write(lp_path + "lagrn.lp")
            raise ValueError(f"Lagrangian problem is unbounded or infeasible, status: {model.status}")
        
        
        copySol = np.array(model.getAttr("X", copyVarObjects))
        lagObj  = model.objVal
        newLB   = lagObj + np.dot(forwardSol, pi)
        grad    = forwardSol - copySol
        gnorm   = np.linalg.norm(grad)
        gap     = abs((subUB - max(newLB, subLB))/subUB)
        iterStepSize = stepSize_ver2(subUB, newLB, gnorm, iter)
        
        # iterStepSize = stepSize(gnorm, iter, normTol)

        if gnorm > normTol and gap > gapTol:
            pi = pi + iterStepSize*grad
            iter += 1
            exitWithoutLagObj = True

        # if gnorm < normTol:

            
        #     model.write(lp_path + "test_lagrn_with_0norm.lp")
        #     subMIP_model.write(lp_path + "test_submip.lp")
        #     print(f"     Gap in MIP: {subMIP_model.MIPGap}")
        #     print(f"     Gap in Lagrn: {model.MIPGap}")
        #     mip_sol = subMIP_model.getAttr("X", subMIP_model.getVars())
        #     lagrn_sol = model.getAttr("X", model.getVars())
        #     mip_varNames   = subMIP_model.getAttr("VarName", subMIP_model.getVars())
        #     lagrn_varNames = model.getAttr("VarName", model.getVars())
        #     sol_eq = np.array_equal(mip_sol, lagrn_sol)
        #     print(f"     Are the two solutions equal: {sol_eq}")


            # # Stack arrays horizontally
            # stacked_arrays = np.column_stack((mip_sol, lagrn_sol, mip_varNames, lagrn_varNames))

            # # Save to CSV
            # np.savetxt(out_path +'arrays.csv', stacked_arrays, delimiter=',', fmt='%d', header='MIP Sol, Lagrn Sol', comments='')

            # data = zip(mip_varNames, mip_sol, lagrn_varNames, lagrn_sol)

            # Write the tuples to a CSV file
            # with open(out_path + 'test_output.csv', 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(['mipNames', 'mipVals', 'lagrnNames', 'lagrnVals'])  # Write header
            #     writer.writerows(data)

            # mip_y0 = subMIP_model.getVarByName("y0")
            # lagrn_y0 =  model.getVarByName("y0")
            # print(f"     MIP obj: {mip_y0.x}, Lagrn Obj: {lagrn_y0.x}")
            # print("      Diagnosis complete")


        print(f"    Lag Iteration: {iter}, lagObj:{lagObj:.1f}, lb: {newLB:.1f}, ub: {subUB:.1f}, gap: {gap:.4f}, ogap:{orig_gap:.3f}, gtol:{gapTol:.3f}, gnorm: {gnorm:.4f}, step: {iterStepSize:.2f}")

    if exitWithoutLagObj:
        model.setAttr("Obj", copyVarObjects, -pi)
        model.optimize()
        lagObj  = model.objVal


    print(f"pi: {pi}, forwardSol: {forwardSol}")



    piTemp = forwardSol.copy()
    for i in range(len(piTemp)):
        if piTemp[i] == 1:
            piTemp[i] = abs(subUB)
        else:
            piTemp[i] = -2*abs(subUB)
    model.setAttr("Obj", copyVarObjects, -piTemp)
    model.optimize()
    copySoltemp = np.array(model.getAttr("X", copyVarObjects))
    lagObjtemp  = model.objVal
    newLBtemp   = lagObjtemp + np.dot(forwardSol, piTemp)
    print("newLB temp", newLBtemp)
    print("copy Solution temp:", copySoltemp)


    return pi, lagObj

def LGDualSolver(pi, maxIter, gapTol, normTol, model, copyVarObjects, forwardSol, subLB = None, subUB = None, subMIP_model = None, subLBx = None, subUBx = None, tolerance = 1e2):

    """
    pi (list or numpy array): starting value of pi
    maxIter (int): maximum number of iterations for solving the Lagrangian dual problem
    gapTol (float):  tolerance in the gap for terminating the subgradient algorithm
    normTol (float): tolernace in the gradient norm for terminating the subgradient algorithm
    model(gurobi model): gurobi model for the lagrangian subproblem
    copyVarObjects (variable objects from gurobi): The variable objects corr to copy variables in the Lagrangian subproblem
    forwardSol (numpy array): solution passed from the previous stage
    subLB (float): lower bound on the subproblem for given forward solution obtained from LP relaxation
    subUB (float): upper bound on the subproblem for given forward solution obtained from exact MIP
    subMIP_model (gurobi model): 
    returns:
    """

    noearlyExit = False
    

    piTemp = forwardSol.copy()
    for i in range(len(piTemp)):
        if abs(1 - piTemp[i]) < 1e-6:
            piTemp[i] = abs(subUB - subLBx) + tolerance
        else:
            piTemp[i] = -abs(subUB-subLBx)-tolerance

    
    model.setAttr("Obj", copyVarObjects, -piTemp)
    model.optimize()
    copySoltemp = np.array(model.getAttr("X", copyVarObjects))
    lagObjtemp  = model.objVal
    newLBtemp   = lagObjtemp + np.dot(forwardSol, piTemp)
    newLBwithGap = model.objBound + np.dot(forwardSol, piTemp)
    gap = abs((subUB - newLBtemp)/(subUB + 1e-7))
    if gap < gapTol:
        print(f"    >>Lagrn Early Exit with gap: {gap:.4f}")
        return piTemp, lagObjtemp
    else:
        model.write(lp_path + "test_lagrn_with_0norm_new.lp")
        subMIP_model.write(lp_path + "test_submip_new.lp")

        print(f"     Gap in MIP: {subMIP_model.MIPGap}")
        print(f"     Gap in Lagrn: {model.MIPGap}")
        mip_sol = subMIP_model.getAttr("X", subMIP_model.getVars())
        lagrn_sol = model.getAttr("X", model.getVars())
        mip_varNames   = subMIP_model.getAttr("VarName", subMIP_model.getVars())
        lagrn_varNames = model.getAttr("VarName", model.getVars())
        sol_eq = np.array_equal(mip_sol, lagrn_sol)
        print(f"     Are the two solutions equal: {sol_eq}")


        # Stack arrays horizontally
        # stacked_arrays = np.column_stack((mip_sol, lagrn_sol, mip_varNames, lagrn_varNames))

        # Save to CSV
        # np.savetxt(out_path +'arrays.csv', stacked_arrays, delimiter=',', fmt='%d', header='MIP Sol, Lagrn Sol', comments='')

        data = zip(mip_varNames, mip_sol, lagrn_varNames, lagrn_sol)

        # Write the tuples to a CSV file
        with open(out_path + 'test_output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['mipNames', 'mipVals', 'lagrnNames', 'lagrnVals'])  # Write header
            writer.writerows(data)

        mip_y0 = subMIP_model.getVarByName("y0")
        lagrn_y0 =  model.getVarByName("y0")
        print(f"     MIP obj: {mip_y0.x}, Lagrn Obj: {lagrn_y0.x}")
        print("      Diagnosis complete")
        noearlyExit = True
        print(f"     No early exit: {gap}")

        print(f"tempLB: {newLBtemp}, tempLBgap:{newLBwithGap} subUB: {subUB}, piTemp:{piTemp}, forwardSol: {forwardSol}, copySolTemp: {copySoltemp}")


    gnorm = 1
    iter  = 0
    infs_unbd_codes = [3,4,5]
    gap = abs((subUB - subLB)/(subUB + 1e-7))
    orig_gap = gap
    # gapTol = max(min(orig_gap/2, gapTol),1*1e-3)
    gapTol = gapTol

    exitWithoutLagObj = True

    while iter < maxIter and gap > gapTol and gnorm > normTol:

        exitWithoutLagObj = False
        #update the coefficients
        model.setAttr("Obj", copyVarObjects, -pi)

        #solve the Lagrangin subproblem with given choice of lagrangian coefficients pi
        model.optimize()

        

        

        #throw error if unable to solve 
        if model.status in infs_unbd_codes:
            model.write(lp_path + "lagrn.lp")
            raise ValueError(f"Lagrangian problem is unbounded or infeasible, status: {model.status}")
        
        
        copySol = np.array(model.getAttr("X", copyVarObjects))
        lagObj  = model.objVal
        newLB   = lagObj + np.dot(forwardSol, pi)
        grad    = forwardSol - copySol
        gnorm   = np.linalg.norm(grad)
        gap     = abs((subUB - max(newLB, subLB))/subUB)
        iterStepSize = stepSize_ver2(subUB, newLB, gnorm, iter)
        
        # iterStepSize = stepSize(gnorm, iter, normTol)

        if gnorm > normTol and gap > gapTol:
            pi = pi + iterStepSize*grad
            iter += 1
            exitWithoutLagObj = True

        # if gnorm < normTol:

            
        #     model.write(lp_path + "test_lagrn_with_0norm.lp")
        #     subMIP_model.write(lp_path + "test_submip.lp")
        #     print(f"     Gap in MIP: {subMIP_model.MIPGap}")
        #     print(f"     Gap in Lagrn: {model.MIPGap}")
        #     mip_sol = subMIP_model.getAttr("X", subMIP_model.getVars())
        #     lagrn_sol = model.getAttr("X", model.getVars())
        #     mip_varNames   = subMIP_model.getAttr("VarName", subMIP_model.getVars())
        #     lagrn_varNames = model.getAttr("VarName", model.getVars())
        #     sol_eq = np.array_equal(mip_sol, lagrn_sol)
        #     print(f"     Are the two solutions equal: {sol_eq}")


            # # Stack arrays horizontally
            # stacked_arrays = np.column_stack((mip_sol, lagrn_sol, mip_varNames, lagrn_varNames))

            # # Save to CSV
            # np.savetxt(out_path +'arrays.csv', stacked_arrays, delimiter=',', fmt='%d', header='MIP Sol, Lagrn Sol', comments='')

            # data = zip(mip_varNames, mip_sol, lagrn_varNames, lagrn_sol)

            # Write the tuples to a CSV file
            # with open(out_path + 'test_output.csv', 'w', newline='') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow(['mipNames', 'mipVals', 'lagrnNames', 'lagrnVals'])  # Write header
            #     writer.writerows(data)

            # mip_y0 = subMIP_model.getVarByName("y0")
            # lagrn_y0 =  model.getVarByName("y0")
            # print(f"     MIP obj: {mip_y0.x}, Lagrn Obj: {lagrn_y0.x}")
            # print("      Diagnosis complete")


        print(f"    Lag Iteration: {iter}, lagObj:{lagObj:.1f}, lb: {newLB:.1f}, ub: {subUB:.1f}, gap: {gap:.4f}, ogap:{orig_gap:.3f}, gtol:{gapTol:.3f}, gnorm: {gnorm:.4f}, step: {iterStepSize:.2f}")

    if exitWithoutLagObj:
        model.setAttr("Obj", copyVarObjects, -pi)
        model.optimize()
        lagObj  = model.objVal

    if noearlyExit:
        print(f"tempLB: {newLBtemp}, subUB: {subUB}, piTemp:{piTemp}, pi: {pi}, forwardSol: {forwardSol}, copySolTemp: {copySoltemp}")
        print("pause")


    # piTemp = forwardSol.copy()
    # for i in range(len(piTemp)):
    #     if piTemp[i] == 1:
    #         piTemp[i] = abs(subUB)
    #     else:
    #         piTemp[i] = -2*abs(subUB)
    # model.setAttr("Obj", copyVarObjects, -piTemp)
    # model.optimize()
    # copySoltemp = np.array(model.getAttr("X", copyVarObjects))
    # lagObjtemp  = model.objVal
    # newLBtemp   = lagObjtemp + np.dot(forwardSol, piTemp)
    # print("newLB temp", newLBtemp)
    # print("copy Solution temp:", copySoltemp)


    return pi, lagObj