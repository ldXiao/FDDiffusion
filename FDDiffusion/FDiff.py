from .discretizatoin import SpaceGrid, StaggeredSpaceGrid
import numpy as np
from scipy.sparse import dia_matrix, csr_matrix


class Operator():
    # the base class of differential operators
    def forward(self, ysample:np.array):
        """
        :param grd: a grid of type SpaceGrid or StaggeredSpaceGrid
        :return: np.array
        """
        print("called parent")
        return ysample
    def __call__(self, ysample:np.array):

        return self.forward(ysample)

class SturmLiouville(Operator):
    def __init__(self, grd:SpaceGrid, p:callable(float)=None, q:callable(float)=None):
        """
        initialize the generalized SturmLiouville operator (p(x) y_x)_x + q(x) y + g(y)
        on SpaceGrid grd
        :param p: callable
        :param q: callable
        :param f: callable
        """
        super(SturmLiouville, self).__init__()
        self.grd = grd.grid
        # get geometric information from grd
        a, b, N, h = grd.left, grd.right, grd.len, grd.step
        self.a = a
        self.b = b
        self.h = h
        # construct the
        self.stgrd = StaggeredSpaceGrid(a, b, N).grid

        if(p):
            self.p = np.array([p(x) for x in self.stgrd])
            # print(self.p)
        else:
            self.p = np.oness_like(self.stgrd)
        if(q):
            self.q = np.array([q(x) for x in self.grd])
        else:
            self.q = np.zeros_like(self.grd)


        # an (N-2) x (N) matix to evaluate the non-bounrady part, remember that p is of lenth N+1
        diag0 = np.array([self.p[1:-2]])
        diag2 = np.array([self.p[2:-1]])
        diag1 = -diag0 - diag2

        data = np.vstack([diag0+ self.q[1:-1], diag1, diag2])
        offsets = np.array([0,-1,-2])
        self.fdmat = dia_matrix((data, offsets), shape=(N, N-2)).transpose().tocsr()/ pow(self.h, 2)
        # self.fdmat = dia_matrix((data, offsets), shape=(N, N-2)).transpose().tocsr()/ pow(self.h, 2)
        # print(self.fdmat.data)


    def forward(self, ysample:np.array):
        # get geometric information from grd
        N = self.grd.shape[0]
        # construct the
        if(ysample.shape[0]!=N):
            print("Error the lonth of list does not match")
        else:
            # print("called")
            return self.fdmat.dot(ysample)

    def __call__(self, ysample:np.array):
        return self.forward(ysample)


