"""

formulate the lagrangian sub-problem in which we relax the copy constraint
Check 4.4 in Ahmed's paper
"""

import cplex as cpx

def lagrangian_subproblem_formulation(id, rhs_const, rhs_tech_row):
    
    """
    creates a subproblem for a given id
    """


    sub  = cpx.Cplex()
    sub.variables.add(obj = [1, 0], types = [sub.variables.type.integer] * 2, names = ['y0', 'y'])
    sub.variables.add(obj = [0, 0], ub = [1, 1], names = ['z0', 'z1'], types = ['C']*2)


    lhsz0 = -rhs_tech_row[0]
    lhsz1 = -rhs_tech_row[1]
    lhsz2 = -rhs_tech_row[2]
    

    lhs0 = lhsz0.tolist()
    lhs1 = lhsz1.tolist()
    lhs2 = lhsz2.tolist()

    rhs = [float(rhs_const[id][j]) for j in range(len(rhs_const[id]))]
    
    #main constraints
    sub.linear_constraints.add(lin_expr = [cpx.SparsePair(ind = ['y0', 'y'] + ['z0', 'z1'], val = [1, -4] + lhs0), cpx.SparsePair(ind = ['y']+ ['z0', 'z1'], val = [-20]+ lhs1), cpx.SparsePair(ind = ['y']+ ['z0', 'z1'], val = [1]+ lhs2)], senses = ['E', 'L', 'L'], rhs = rhs, names = ['s0', 's1', 's2'])

    
    #set the objective later (but this much problem will remain fixed across all iterations)
    sub.write("original_sub.lp")

    return sub



def instanceSolve(ld_sub, start):

    ld_sub.objective.set_linear('z0', -start[0])
    ld_sub.objective.set_linear('z1', -start[1])

    ld_sub.set_log_stream(None)
    ld_sub.set_error_stream(None)
    ld_sub.set_warning_stream(None)
    ld_sub.set_results_stream(None)

    #solve the problem and obtain the solution
    ld_sub.solve()

    ld_obj = ld_sub.solution.get_objective_value()

    
    return ld_obj
