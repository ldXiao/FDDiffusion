import numpy as np



class BoundaryCondition():

    def __init__(self, loci:float, type="D", constraint=None):
        """
        :param loci: place of the boundary condition
        :param type: BC type choosen between Dirichlete and Neumann
        :param constraint: callable function of t, time dependent,
        if not specified, just put lambda t: 0.0
        """
        BCdict = {"D": "Dirichlet", "N": "Neumann"}
        self.loci = loci
        if(constraint):
            self.constraint = constraint
        else:
            self.constraint = lambda t: 0.0
        self.type = BCdict[type]

