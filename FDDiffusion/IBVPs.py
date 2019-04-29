from .FDiff import SturmLiouville
# from .discretizatoin import SpaceGrid
# from .BCs import BoundaryCondition
from scipy.sparse import csr_matrix, dia_matrix, spmatrix, identity
from scipy.sparse.linalg import spsolve
from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp
from .Integrators import BS_RK23, Ralston
import numpy as np

solver_dict={"BS_RK23":BS_RK23, "Ralston":Ralston}
MESSAGES = {0: "The solver successfully reached the end of the integration interval.",
            1: "A termination event occurred."}



class OdeResult(OptimizeResult):
    pass



class IBVP():
    def __init__(self,
                 ST:SturmLiouville, # Sturm-Liouville operator of (p(x) u_x)_x + q(x)
                 Nonlinear:callable(float), #g(u(x))
                 IC:callable(float), #u(x, 0)
                 BCs:list, #[BoundaryCondition, BoundaryCondition]
                 rhs:callable # source term on rhs callable of the form rhs(x,t)
                 ):
        self.grd = ST.grd
        self.left = ST.a
        self.right = ST.b
        self.h = ST.h
        self.p = ST.p
        self.q = ST.q
        self.fdmat = ST.fdmat
        self.nonlinear = Nonlinear
        self.initial = np.array([ IC(g) for g in self.grd])
        self.periodic = False
        if(len(BCs)==2):
            self.BCleft, self.BCright = BCs[0], BCs[1]
        elif(len(BCs)==0):
            self.periodic = True
        self.rhs = lambda t: np.array([rhs(x, t) for x in self.grd])
        self.success = False
        # self.odefunc = None # odefunction to be implemented of the form odefunc(u,t), where u(self.grd)
        self.geometry_type = None

    def compute(self):
        try:
            if(self.periodic == False):
                p = self.p

                N = p.shape[0] - 1  # get the number of points, where p is the staggered grid has one more point
                boundary_type_dict = {"Dirichlet": -1, "Neumann": 0}

                geometry_type = \
                    (boundary_type_dict[self.BCleft.type], boundary_type_dict[self.BCright.type])
                # print(geometry_type)
                self.geometry_type = geometry_type
                shape = (N + sum(geometry_type), N + sum(geometry_type))
                diagonal = -p[:-1] + -p[1:]
                off_diagonal = p[1:-1]
                data0 = diagonal[-geometry_type[0]: N + geometry_type[1]].copy() \
                        + self.q[-geometry_type[0]: N + geometry_type[1]].copy() * (self.h ** 2)
                data1 = off_diagonal[-geometry_type[0]:N + geometry_type[1] - 1].copy()
                datam1 = off_diagonal[-geometry_type[0]:N + geometry_type[1] - 1].copy()
                if (N > 3):
                    # print("called")
                    # print("raw data1",data1)
                    data1[0] += (1 + geometry_type[0]) * p[0]
                    datam1[-1] += (1 + geometry_type[1]) * p[-1]
                    # print(p[0], p[-1])
                data1 = np.hstack([[0], data1])
                datam1 = np.hstack([datam1, [0]])
                offsets = np.array([-1, 0, 1])
                data = np.vstack([datam1, data0, data1])
                # print(data[0])
                A = dia_matrix((data, offsets), shape=shape).tocsr() / (self.h ** 2)
                self.fdmat = A
                self.success = True

            elif(self.periodic):
                p = self.p
                # print(p)
                N = p.shape[0] - 1

                shape = (N - 1, N -1)
                diagonal = -p[:-1] + -p[1:] # of length N
                off_diagonal = p[1:-1].copy() # of length N-1
                data0 = diagonal[0: N-1].copy() \
                        + self.q[0: N-1].copy() * (self.h ** 2) # len= N-1
                data1 = off_diagonal[0: N - 2].copy() #len = N -2
                datam1 = off_diagonal[0: N - 2].copy() # len = N-2

                data1 = np.hstack([[0], data1]) # len = N-1
                datam1 = np.hstack([datam1, [0]]) # len = N-1
                offsets = np.array([-1, 0, 1])
                data = np.vstack([datam1, data0, data1])
                # print(data[0])
                A = dia_matrix((data, offsets), shape=shape).tocsr()
                A[0, -1]= p[0]
                A[-1, 0]= p[-2]
                A /= self.h ** 2
                self.fdmat = A.tocsr()
                # print(A)
                self.success = True

        except:
            self.success = False





        # self.rhs = self.rhs[-geometry_type[0]: N + geometry_type[1]]

    def _odefunc(self, t:float, u:np.array):
        # u must be sliced before put into this function
        if(self.success):
            if(self.periodic==False):
                p = self.p
                geometry_type = self.geometry_type
                N = p.shape[0] - 1
                rhs = self.rhs(t)
                # print(rhs)
                rhs[0] -= (geometry_type[0] + 1) * 2 * p[0] * self.BCleft.constraint(t) / (self.h)
                rhs[-1] -= (geometry_type[1] + 1) * 2 * p[-1] * self.BCright.constraint(t) / (self.h)
                rhs[1] -= (-geometry_type[0]) * p[1] * self.BCleft.constraint(t) / (self.h ** 2)
                rhs[-2] -= (-geometry_type[1]) * p[-2] * self.BCright.constraint(t) / (self.h ** 2)
                if(self.nonlinear!=None):
                    nonlinear = np.array([self.nonlinear(x) for x in u ])
                else:
                    nonlinear = np.zeros_like(u)
                return self.fdmat.dot(u)- rhs[-geometry_type[0]: N + geometry_type[1]] + nonlinear
            else:
                p = self.p
                geometry_type = self.geometry_type
                N = p.shape[0] - 1
                rhs = self.rhs(t)
                if (self.nonlinear != None):
                    nonlinear = np.array([self.nonlinear(x) for x in u])
                else:
                    nonlinear = np.zeros_like(u)
                return self.fdmat.dot(u) - rhs[0:N-1] + nonlinear


        else:
            print("discretization not specified yet, run compute first")

    def ibvp_solve(self,t_span:list=[0,1], type="RK45", max_step=0.01):
        p = self.p
        geometry_type = self.geometry_type
        N = p.shape[0] - 1


        if(self.periodic==False):
            u0 = self.initial[-geometry_type[0]: N + geometry_type[1]]
            if(type in solver_dict):
                # print("called")
                type = solver_dict[type]
            # print("ahah", self.odefunc(u0, 0))
            sol = solve_ivp(self._odefunc, t_span, u0, method=type, max_step=max_step)
            left = np.array([])
            right = np.array([])
            if(geometry_type[0]!=0):
                # add back the boundary value if it is dirichlet
                left = np.hstack([np.array([self.BCleft.constraint(t) for t in sol.t]),left])
                sol.y = np.vstack([[left], sol.y])
            if(geometry_type[1]!=0):
                right = np.hstack([right,np.array([self.BCright.constraint(t) for t in sol.t])])
                sol.y = np.vstack([sol.y, [right]])
        else:
            print("called")
            u0 = self.initial[0 : N-1]
            if (type in solver_dict):
                # print("called")
                type = solver_dict[type]
            sol = solve_ivp(self._odefunc, t_span, u0, method=type, max_step=max_step)
            head = sol.y[0,:].copy()
            sol.y = np.vstack([sol.y, [head]])


        return sol

    def _CN_step(self, t, u, dt, tf):
        """
        conduct a crank-nicolson step
        :param t: time
        :param u: functio value
        :param dt: time step
        :return: t_new,u_new, message
        """
        assert(self.success)
        assert(self.nonlinear==None)
        message = None
        try:
            p = self.p
            geometry_type = self.geometry_type
            N = p.shape[0] - 1
            rhs = self.rhs(t+dt/2)
            # print(rhs)
            if(self.periodic==False):
                rhs[0] -= (geometry_type[0] + 1) * 2 * p[0] * self.BCleft.constraint(t+dt/2) / (self.h)
                rhs[-1] -= (geometry_type[1] + 1) * 2 * p[-1] * self.BCright.constraint(t+dt/2) / (self.h)
                rhs[1] -= (-geometry_type[0]) * p[1] * self.BCleft.constraint(t+dt/2) / (self.h ** 2)
                rhs[-2] -= (-geometry_type[1]) * p[-2] * self.BCright.constraint(t+dt/2) / (self.h ** 2)

                RHS = (identity(N + sum(geometry_type)) + 0.5 * dt * self.fdmat).tocsr().dot(u) \
                      - rhs[-geometry_type[0]: N + geometry_type[1]] * dt
                u_new = spsolve((identity(N + sum(geometry_type)) - 0.5 * dt * self.fdmat).tocsr(), RHS)
                t_new = t+dt
            else:
                RHS = (identity(N -1) + 0.5 * dt * self.fdmat).tocsr().dot(u) \
                      - rhs[0: N -1] * dt
                u_new = spsolve((identity(N -1) - 0.5 * dt * self.fdmat).tocsr(), RHS)
                t_new = t + dt

            if (t_new> tf):
                message = "finished"
            return t_new, u_new, message
        except:
            print("called")
            print(" Early stopped")
            return t, u, message

    def customize_solve(self, t_span:list=[0,1], type="CN",max_step=0.01):
        """
        A solver for Crank-Nicolson only
        :param t_span:
        :param max_step: float
        :return:
        """
        assert(self.success)
        assert(type in ["CN", "BE"])
        p = self.p
        geometry_type = self.geometry_type
        N = p.shape[0] - 1
        if(type=="CN"):
            step = self._CN_step
        elif(type=="BE"):
            step = self._BE_step
        if self.periodic == False:
            u0 = self.initial[-geometry_type[0]: N + geometry_type[1]]
        else:
            u0 = self.initial[0:N-1]
        t0, tf = float(t_span[0]), float(t_span[1])
        ts = [t0]
        ys = [u0]
        status = None
        t = t0
        u = u0
        while status is None:
           t, u, message = step(t, u, max_step, tf)
           if message == 'finished':
               status = 0
           elif message == 'failed':
               status = -1
               break

           ts.append(t)
           ys.append(u)


        message = MESSAGES.get(status, message)
        ts = np.array(ts)
        ys = np.vstack(ys).T

        sol=OdeResult(t=ts, y=ys)
        left = np.array([])
        right = np.array([])
        if(self.periodic==False):
            if (geometry_type[0] != 0):
                # add back the boundary value if it is dirichlet
                left = np.hstack([np.array([self.BCleft.constraint(t) for t in sol.t]), left])
                sol.y = np.vstack([[left], sol.y])
            if (geometry_type[1] != 0):
                right = np.hstack([right, np.array([self.BCright.constraint(t) for t in sol.t])])
                sol.y = np.vstack([sol.y, [right]])

        else:
            head = sol.y[0,:].copy()
            sol.y = np.vstack([sol.y, [head]])

        return sol

    def _BE_step(self, t, u, dt, tf):
        """
        conduct a backward-Euler step
        :param t: current time
        :param u: function value
        :param dt: teme step
        :param tf: final time
        :return: t_new, u_new, message
        """
        assert (self.success)
        assert (self.nonlinear == None)
        message = None
        try:
            p = self.p
            geometry_type = self.geometry_type
            N = p.shape[0] - 1
            rhs = self.rhs(t + dt)
            # print(rhs)
            if(self.periodic==False):
                rhs[0] -= (geometry_type[0] + 1) * 2 * p[0] * self.BCleft.constraint(t + dt) / (self.h)
                rhs[-1] -= (geometry_type[1] + 1) * 2 * p[-1] * self.BCright.constraint(t + dt) / (self.h)
                rhs[1] -= (-geometry_type[0]) * p[1] * self.BCleft.constraint(t + dt) / (self.h ** 2)
                rhs[-2] -= (-geometry_type[1]) * p[-2] * self.BCright.constraint(t + dt) / (self.h ** 2)

                RHS = u - rhs[-geometry_type[0]: N + geometry_type[1]] * dt
                u_new = spsolve((identity(N + sum(geometry_type)) - dt * self.fdmat).tocsr(), RHS)
                t_new = t + dt
            else:
                RHS = u - rhs[0: N -1] * dt
                u_new = spsolve((identity(N - 1) - dt * self.fdmat).tocsr(), RHS)
                t_new = t + dt

            if (t_new > tf):
                message = "finished"
            return t_new, u_new, message
        except:
            # print("called")
            return t, u, message


    def BE_solve(self, t_span:list=[0,1], max_step=0.01):
        assert (self.success)
        p = self.p
        geometry_type = self.geometry_type
        N = p.shape[0] - 1

        u0 = self.initial[-geometry_type[0]: N + geometry_type[1]]
        t0, tf = float(t_span[0]), float(t_span[1])
        ts = [t0]
        ys = [u0]
        status = None
        t = t0
        u = u0
        while status is None:
            t, u, message = self._BE_step(t, u, max_step, tf)
            if message == 'finished':
                status = 0
            elif message == 'failed':
                status = -1
                break

            ts.append(t)
            ys.append(u)

        message = MESSAGES.get(status, message)
        ts = np.array(ts)
        ys = np.vstack(ys).T

        sol = OdeResult(t=ts, y=ys)
        left = np.array([])
        right = np.array([])
        if (geometry_type[0] != 0):
            # add back the boundary value if it is dirichlet
            left = np.hstack([np.array([self.BCleft.constraint(t) for t in sol.t]), left])
            sol.y = np.vstack([[left], sol.y])
        if (geometry_type[1] != 0):
            right = np.hstack([right, np.array([self.BCright.constraint(t) for t in sol.t])])
            sol.y = np.vstack([sol.y, [right]])

        return sol




