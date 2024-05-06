# read the .sto (stochastic file) in the SMPS format (for two-stage problems)


class readstoc:
    """
    reads the stochastic data in the problem
    """

    
    def __init__(self, name):
        self.name = name + ".sto"
        self.rv = list()
        self.dist = list()
        self.cumul_dist = list()
        self.rvnum = 0

    def readfile(self):
        with open(self.name, "r") as f:
            data = f.readlines()
        count = 0
        cumul = 0

        for line in data:
            words = line.split()
            
    