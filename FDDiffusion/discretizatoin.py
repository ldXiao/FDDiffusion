import numpy as np

class Grid():
    def __init__(self, a:float, b:float, N:int):
        """
        initialize a finite gird of general type [a,..., b] and divided into N-1 pieces with N vertices.
        :param a: left space boundary
        :param b: right space boundary
        :param N: number of vectices

        """
        if (a > b):
                print("wrong input b should be larger than a")
        self.len = N
        self.step = (b-a) / (self.len-1)
        self.left = a
        self.right = b
        self.grid = np.linspace(self.left, self.right, self.len)
        self.staggered = False

    def __str__(self):
        return "grid:"+str(self.grid)


class SpaceGrid(Grid):
    def __init__(self, a:float, b:float, N:int):
        """
        initialize a finite gird of general type [a,..., b] and divided into N-1 pieces with N vertices.
        :param a: left space boundary
        :param b: right space boundary
        :param N: number of vectices

        """
        super(SpaceGrid, self).__init__(a, b, N)

    def __str__(self):
        return "spacegrid:"+str(self.grid)


class StaggeredSpaceGrid(Grid):
    def __init__(self, a:float, b:float, N:int):
        """
        initialize a finite gird of general type [a-1/2,..., b+1/2] and with N+1 vertices.
        :param a: left space boundary
        :param b: right space boundary
        :param N: number of vertices - 1
        """
        super(StaggeredSpaceGrid, self).__init__(a,b, N)
        self.len = N + 1
        self.left = a - self.step * 0.5
        self.right = b+ self.step * 0.5
        self.grid = np.linspace(self.left, self.right, self.len)
        self.staggered = True

    def __str__(self):
        return "staggered grid:" + str(self.grid)


