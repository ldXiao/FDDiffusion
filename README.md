# FDDiffusion
A 1D BVP IBVP solver for diffusion or advection type PDE, with fruitful choice of integrators and illustration tools.
Basic ussage example with discrete Green's function:

```python
from FDDiffusion.FDiff import SturmLiouville
from FDDiffusion.discretizatoin import SpaceGrid
from FDDiffusion.BCs import BoundaryCondition
from FDDiffusion.BVPs import BVP
from FDDiffusion.IBVPs import IBVP, OdeResult
import numpy as np
import matplotlib.pyplot as plt
from FDDiffusion.utils import PlotSpaceTimePDE, Animate_1d_wave, Write_sol_json, Read_sol_json

CFL = 1.5
N = 128

# space- time geometric gird
a = -1
b = 1
dx = 2/N
dt = (CFL* dx ** 2)
tf = 0.25
method = "CN" # use Crank-Nicolson method, available integrators include Ralston, SBDF2 and all stadrad integrators in scipy

# the weight functions for Sturm-Liouville operator (d(x) u_x)_x + q(x) u
def d(x):
    return 1

def q(x):
    return 0.0
# boundary control functions
def bc0(t):
    return 0

def bc1(t):
    return 0
# initial condition
def Ddelta(x):
    if (x < dx/4  and x > -dx/4):
        return 1
    else:
        return 0
# source term on rhs
def f(x, t):

    return 0

# initialize Boundary conditions, both Neumann
BC1 = BoundaryCondition(a, type="N", constraint=bc0)
BC2 = BoundaryCondition(b, type="N", constraint=bc1)

#space grid, containing staggered grids
spgrid = SpaceGrid(a, b, N + 1)
xsample = np.linspace(a, b, N + 1)

#initialize Sturm-Liouville operator, use BCs=[] is you want to specify periodic boundary condition
ST = SturmLiouville(spgrid, d, q)
#initialize initial-boundary value problem
ibvp = IBVP(ST=ST, Nonlinear=None, IC=Ddelta, BCs=[BC1,BC2], rhs=f)
# compute the dinite difference matrix in the given boundry condition
ibvp.compute()

# solve the initial value boundary problem
if(ibvp.success):
    sol = ibvp.customize_solve(t_span=[0, tf], type=method, max_step=dt)

# slice the initial state to make a better plot
sol.t = sol.t[1:]
sol.y = sol.y[:,1:]
min = sol.y.min()
#Plot a 3D plot of the solution
PlotSpaceTimePDE(sol, xsample, xcount=2, tcount=2,
                 title="Discrete Green for {},N={}, CFL={}, min={:.2g}".format(method,N, CFL,min))
```
