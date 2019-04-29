from .FDiff import SturmLiouville
from .discretizatoin import SpaceGrid
from .BCs import BoundaryCondition
from scipy.sparse import csr_matrix, dia_matrix, spmatrix
from scipy.sparse.linalg import spsolve
import numpy as np

class BVP():
    def __init__(self, ST:SturmLiouville, BCs:list, rhs:callable(float)):
        self.grd = ST.grd
        self.left = ST.a
        self.right = ST.b
        self.h = ST.h
        self.p = ST.p
        self.q = ST.q
        self.fdmat = ST.fdmat
        self.BCleft, self.BCright = BCs[0], BCs[1]
        self.rhs = rhs(self.grd)
        # print(self.fdmat.shape)
        self.success = False

    def compute(self):
        try:
            p = self.p

            N = p.shape[0]-1 # get the number of points, where p is the staggered grid has one more point
            boundary_type_dict={"Dirichlet": -1, "Neumann": 0}

            geometry_type = \
                (boundary_type_dict[self.BCleft.type], boundary_type_dict[self.BCright.type])
            # print(geometry_type)

            shape = (N+sum(geometry_type), N+sum(geometry_type))
            diagonal = -p[:-1] + -p[1:]
            off_diagonal = p[1:-1]
            data0 = diagonal[-geometry_type[0]: N+geometry_type[1]].copy() \
                    + self.q[-geometry_type[0]: N+geometry_type[1]].copy() * (self.h ** 2)
            data1 = off_diagonal[-geometry_type[0]:N+geometry_type[1]-1].copy()
            datam1 = off_diagonal[-geometry_type[0]:N+geometry_type[1]-1].copy()
            if(N>3):
                # print("called")
                # print("raw data1",data1)
                data1[0] += (1 + geometry_type[0]) * p[0]
                datam1[-1] += (1 + geometry_type[1]) * p[-1]
                # print(p[0], p[-1])
            data1 = np.hstack([[0],data1])
            datam1 = np.hstack([datam1,[0]])
            offsets = np.array([-1,0,1])
            data = np.vstack([datam1, data0, data1])
            # print(data[0])
            A=dia_matrix((data, offsets), shape=shape).tocsr()/ (self.h ** 2)
            self.fdmat = A

            self.rhs[0] -= (geometry_type[0] + 1) * 2 * p[0] * self.BCleft.constraint(0) / (self.h)
            self.rhs[-1] -= (geometry_type[1] + 1) * 2 * p[-1] * self.BCright.constraint(0) / (self.h)
            self.rhs[1] -= (-geometry_type[0]) * p[1] * self.BCleft.constraint(0) / (self.h ** 2)
            self.rhs[-2] -= (-geometry_type[1]) * p[-2] * self.BCright.constraint(0) / (self.h ** 2)
            self.rhs = self.rhs[-geometry_type[0]: N + geometry_type[1]]
            self.success = True
        except:
            self.success = False


    def solve(self):
        if(self.success):
            boundary_type_dict = {"Dirichlet": -1, "Neumann": 0}
            geometry_type = \
                (boundary_type_dict[self.BCleft.type], boundary_type_dict[self.BCright.type])

            left = np.array([])
            right = np.array([])
            if(self.BCleft.type== "Dirichlet" ):
                left = np.hstack([[self.BCleft.constraint(0)],left])
            if(self.BCright.type=="Dirichlet"):
                right = np.hstack([right,[self.BCright.constraint(0)]])

            return np.hstack([left, spsolve(self.fdmat, self.rhs), right])
        else:
            print("compute failed")

