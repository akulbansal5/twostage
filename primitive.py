


class Primitive:
    
    """
    Define attributes common to all instances
    """

    def __init__(self):   

        self.cpx_model = None
        self.sols      = None

    def relax_integer_problem(self):

        """
        assumes self has an associated attribute cpx_model (Cplex class object)
        """

        varN     = self.cpx_model.variables.get_num()
        varTypes = self.cpx_model.variables.get_types()

        changeInd  = []
        boundInd   = []
        binaryInd  = []
        integerInd = []

        for i in range(varN):
            if varTypes[i] == 'B':
                binaryInd.append(i)
                changeInd.append((i, 'C'))
                boundInd.append((i, 1))
            elif varTypes[i] == 'I':
                changeInd.append((i, 'C'))
                integerInd.append(i)
       
        self.cpx_model.variables.set_types(changeInd)
        self.cpx_model.variables.set_upper_bounds(boundInd)

        return binaryInd, integerInd

    def revert_back_to_mip(self, binaryInd, integerInd):

        """
        makes the master problem mip on specified indices
        assumes self has an associated attribute cpx_model (Cplex class object)
        """  

        typeList = [(i, 'B') for i in binaryInd] + [(i, 'I') for i in integerInd]
        if len(binaryInd) > 0 or len(integerInd) > 0:
            self.cpx_model.variables.set_types(typeList)

    def isImprovementSignificant(self, lb_list, ub_list, count, impr_gap = 1e-2, impr_lb = 0.05, past_iter = 50, tolerance = 1e-6):

        """
        lb_list: list of lower bounds in successive iterations
        ub_list: list of upper bounds in successive iterations
        if no significant improvement in lower bound or gap then solve mips and terminate:

                if any of lower or upper bound is - \infty and + \infty then return False.
                if gap now and gap in past iteration has not improved by (impr_gap 1%) then return False.
                if relative lower bound from past to current iteration has not improved significantly then return False.
        """

        if lb_list[-1] == float('-inf') or ub_list[-1] == float('inf'):
            return False

        gap_now = abs(ub_list[-1] - lb_list[-1])/(tolerance + abs(ub_list[-1]))
        gap_after = abs(ub_list[-past_iter] - lb_list[-past_iter])/(tolerance + abs(ub_list[-past_iter]))

        if abs(gap_now - gap_after) < impr_gap and abs(lb_list[-1] - lb_list[-past_iter])/abs(lb_list[-past_iter]) < impr_lb:
            return False

        return True

    def isFractionalSol(self, tolerance = 1e-4):
        
        """
        determines if the solution we expect is positive
        """

        flag = False
        for val in self.sols:
            if self.isFractional(val, tolerance):
                flag = True
                break
        return flag

    def isFractional(self, val, tolerance = 1e-4):
        
        """
        determines if the solution is fractional
        """

        if val%1 == 0:
            return False
        else:
            return abs(val - round(val)) > tolerance

    def count_cuts(self):
    
        """
        master: 
        """
        
        total = 0
        search = list(range(11)) + [20]
        
        for i in search:
            total += self.cpx_model.solution.MIP.get_num_cuts(i)

        return total



    


    