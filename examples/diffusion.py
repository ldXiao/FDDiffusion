from FDDiffusion.FDiff import SturmLiouville
from FDDiffusion.discretizatoin import SpaceGrid
from FDDiffusion.BCs import BoundaryCondition
from FDDiffusion.BVPs import BVP
from FDDiffusion.IBVPs import IBVP, OdeResult
from FDDiffusion.Integrators import BS_RK23
import numpy as np
from scipy.optimize import  OptimizeResult
import matplotlib.pyplot as plt
from FDDiffusion.utils import PlotSpaceTimePDE, Animate_1d_wave, \
    Write_sol_json, Read_sol_json, FunctionNormAlong, FunctionNorm1D

def function_norm_err(ysample:list, u_exact:list, norm_type:str):
    norm_dict = {"L2": 2, "L1": 1, "Linf": np.inf}
    func_norm = np.linalg.norm(ysample, norm_dict[norm_type])
    err_norm = np.linalg.norm(ysample-u_exact, norm_dict[norm_type])
    return err_norm/func_norm

def Node_num_rel_err(M=1000, a=0, b=1, norm_type="L1"):
    M=M
    spgrid = SpaceGrid(a, b, M+1)
    xsample = np.linspace(a, b , M+1)
    def d(x):
        return 2+ np.cos(np.pi * x)
    def u(x):
        return np.cos(np.pi * x /2)

    def result(x):
        return -3.0/8 *(np.pi ** 2) * (np.cos(np.pi/2 * x)+ np.cos(3/2 * np.pi * x))
    ST = SturmLiouville(spgrid,d)

    usample = u(xsample)
    ysample = ST(usample)
    u_exact = result(xsample)[1:-1]
    return function_norm_err(ysample=ysample, u_exact=u_exact, norm_type=norm_type)

def discretization_error_plot(node_nums=[8,16, 32,64,128,256,512,1024],a=0, b=1):
    norm_dict = {"L2": 2, "L1": 1, "Linf": np.inf}
    mark_dict = {"L2": 'o-', "L1": 's-', "Linf": 'h-'}
    for norm_type in norm_dict:
        err_list = [Node_num_rel_err(num, a=0., b=1.,norm_type=norm_type) for num in node_nums]
        plt.loglog(node_nums, err_list, mark_dict[norm_type],label=norm_type)
    quadr = lambda x: x ** (-2) * 10
    plt.loglog(node_nums, [quadr(x) for x in node_nums],'--', label="quadratic base line")
    plt.xlabel("node numbers")
    plt.ylabel("relative error w.r.t. exact function")
    plt.legend()
    plt.grid(True)
    plt.show()

    def result(x):
        return -3.0 / 8 * (np.pi ** 2) * (np.cos(np.pi / 2 * x) + np.cos(3 / 2 * np.pi * x))
    # BC1 = BoundaryCondition(a, type="D", constraint=lambda t: 0.)
    # BC2 = BoundaryCondition(b, type="N", constraint=lambda t: -0.5 * np.pi)
    for num in [2,4,5,6,8]:
        spgrid = SpaceGrid(a, b, num + 1)
        xsample = np.linspace(a, b, num + 1)

        def d(x):
            return 2 + np.cos(np.pi * x)

        def u(x):
            return np.cos(np.pi * x / 2)
        ST = SturmLiouville(spgrid, d)


        usample = u(xsample)
        ysample = ST(usample)
        plt.plot(xsample[1:-1], ysample, label="node num={}".format(num))
    plt.plot(np.linspace(a, b, 1000 + 1), result(np.linspace(a, b, 1000 + 1)), label="exact")
    plt.legend()
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def BVP_solve_plot(node_nums = [3,5,8,16,32,64,128,256,512,1024]):

    mark_dict = dict(zip([3,4, 5, 6, 7, 8]
    , ["p-","o-", "X-", "s-", 'x-', 'D-']))

    def d(x):
        return 2 + np.cos(np.pi * x)

    def q(x):
        return 0.0

    def u(x):
        return np.cos(np.pi * x / 2)

    def ftilde(x):
        return -3.0 / 8 * (np.pi ** 2) * (np.cos(np.pi / 2 * x) + np.cos(3 / 2 * np.pi * x))

    a = 0
    b = 1
    BC1 = BoundaryCondition(a, type="D", constraint=lambda t: 1.)
    BC2 = BoundaryCondition(b, type="N", constraint=lambda t: -0.5 * np.pi)
    geometry_dict = {"Dirichlet":-1, "Neumann":0}

    norm_errors_L1 =[]
    norm_errors_L2 =[]
    norm_errors_Linf = []
    dirichlet_errors = []
    neumann_errors = []
    for num in node_nums:
        spgrid = SpaceGrid(a, b, num + 1)
        xsample = np.linspace(a, b, num + 1)
        # xsample = xsample[-geometry_dict[BC1.type]: xsample.shape[0]+geometry_dict[BC2.type]]
        # xsample=xsample[1:]
        ST = SturmLiouville(spgrid, d, q)
        bvp = BVP(ST=ST, BCs=[BC1, BC2], rhs=ftilde)
        bvp.compute()
        # print(bvp.solve().shape, xsample.shape)
        ysample = bvp.solve()
        # print("aha",ST.fdmat.toarray())
        # print("ahaaa", ST.p / (ST.h ** 2))
        if(num < 10):
            plt.plot(xsample, bvp.solve(), 'X-',label="node number ={}".format(num))
        usample = u(xsample)
        norm_errors_L1.append(function_norm_err(ysample, usample, "L1"))
        norm_errors_L2.append(function_norm_err(ysample, usample, "L2"))
        norm_errors_Linf.append(function_norm_err(ysample, usample, "Linf"))
        neumann_errors.append(np.abs(ysample[-1]-usample[-1]))

    plt.plot(np.linspace(a, b, 1000), u(np.linspace(a, b, 1000)), label="exact")
    plt.title("Compare the exact result with small node numbers")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.loglog(node_nums, [2 * error for error in neumann_errors],'x-' , label="neumann error")
    plt.loglog(node_nums, norm_errors_Linf,'s-', label="Linf norm error")
    plt.loglog(node_nums, norm_errors_L2, 'o-',label="L2 norm error")
    plt.loglog(node_nums, norm_errors_L1,'p-', label="L1 nodem error")
    plt.loglog(node_nums, [pow(num, -2) * 10  for num in node_nums], '--', label="second order base line")
    plt.legend()
    plt.grid(True)
    plt.show()

def IBVP_solve_plot():
    def d(x):
        return 1

    def q(x):
        return 0

    def u(x):
        return np.cos(np.pi * x / 2)

    def ftilde(xsample, t):
        return  0
    def ic1(xsample):
        return np.array([pow(np.cos(np.pi * 0.5 * x) ,100) for x in xsample])


    def ic(xsample):
        def step(x):
            if x < 0.5:
                return 0
            else:
                return 1
        return np.array([step(x) for x in xsample])

    a = 0
    b = 1
    BC1 = BoundaryCondition(a, type="D", constraint=lambda t: 0)
    BC2 = BoundaryCondition(b, type="N", constraint=lambda t: 0)
    geometry_dict = {"Dirichlet": -1, "Neumann": 0}
    spgrid = SpaceGrid(a, b, 512 + 1)
    xsample = np.linspace(a, b, 512 + 1)
    # xsample = xsample[-geometry_dict[BC1.type]: xsample.shape[0]+geometry_dict[BC2.type]]
    # xsample=xsample[1:]
    ST = SturmLiouville(spgrid, d, q)
    ibvp =IBVP(ST=ST,Nonlinear=None, IC=ic, BCs=[BC1, BC2], rhs=ftilde)
    ibvp.compute()
    # print(ibvp.odefunc(ic(xsample)[0:-1], 0))
    sol = ibvp.ibvp_solve(t_span=[0,10], type="Radau", max_step=0.01)
    Write_sol_json(sol,file_name="sol.json")
    # sol = ibvp.customize_solve(t_span=[0,10], max_step=0.01)
    sol=Read_sol_json(file_name="sol.json")
    # Animate_1d_wave(sol, xsample=xsample)
    PlotSpaceTimePDE(sol, xsample)



def LTEplot3D():
    def d(x):
        return 2 + np.cos(np.pi*x)

    def q(x):
        return 0.0

    def bc0(t):
        return 1 + np.exp(- (np.pi ** 2) * t / 4)

    def bc1(t):
        return - 0.5 * np.pi * (1 + np.exp(- (np.pi ** 2) * t / 4))

    def ic(x):
        return 2 * np.cos(0.5 * np.pi * x)

    def f(x, t):
        result =  -1/ 4 * (np.pi ** 2) * np.exp(- (np.pi ** 2) * t / 4) \
                * np.cos(x * np.pi/2)*(-1 + 3 *(1 + np.exp((np.pi ** 2) * t / 4))* np.cos(x * np.pi))
        return result
    def exact_sol(x,t):
        return (1 + np.exp(- (np.pi ** 2) * t / 4)) * np.cos(np.pi * x /2)


    a = 0
    b = 1
    t_span=[0,0.25]
    N=10
    dt=0.00213
    method = "Ralston"
    BC0 = BoundaryCondition(a, type="D", constraint=bc0)
    BC1 = BoundaryCondition(b, type="N", constraint=bc1)

    spgrid = SpaceGrid(a, b, N + 1)
    xsample = np.linspace(a, b, N + 1)
    # initialize the SturmLiouville system and initial-boundary value problem
    ST = SturmLiouville(spgrid, d, q)
    ibvp = IBVP(ST=ST, Nonlinear=None, IC=ic, BCs=[BC0, BC1], rhs=f)
    # compute the discretization and matrix involved
    ibvp.compute()

    if(method in ["CN", "BE"]):
        sol = ibvp.customize_solve(t_span=t_span, type=method, max_step=dt)
    else:
        sol=ibvp.ibvp_solve(t_span=t_span, type=method, max_step=dt)
    json_name = "sol_test node={}, dt={}, for {}".format(N,dt, method)
    Write_sol_json(sol,json_name+".json")
    sol1 = Read_sol_json(json_name+".json")
    PlotSpaceTimePDE(sol1, xsample, title="Space-Time-Solution node={}, dt={}, for {}".format(N,dt, method), tcount=50, xcount=1)

    ysample= np.log10(np.abs(sol1.y -np.array([[exact_sol(x,t) for t in sol1.t] for x in xsample])))
    sol2= OdeResult(t=sol1.t[1:], y=ysample[1:,1:])
    PlotSpaceTimePDE(sol2, xsample[1:],
                     title="Space-Time-Error node={}, dt={} for {}".format(N,dt, method),
                     tcount=50, xcount=1, error_plot=True)

def LTEplot2D(grid_size=[4,8,16,32,64, 128], axis:str='t', norm_type:str="L1"):
    def d(x):
        return 2 + np.cos(np.pi*x)

    def q(x):
        return 0.0

    def bc0(t):
        return 1 + np.exp(- (np.pi ** 2) * t / 4)

    def bc1(t):
        return - 0.5 * np.pi * (1 + np.exp(- (np.pi ** 2) * t / 4))

    def ic(x):
        return 2 * np.cos(0.5 * np.pi * x)

    def f(x, t):
        result =  -1/ 4 * (np.pi ** 2) * np.exp(- (np.pi ** 2) * t / 4) \
                * np.cos(x * np.pi/2)*(-1 + 3 *(1 + np.exp((np.pi ** 2) * t / 4))* np.cos(x * np.pi))
        return result
    def exact_sol(x,t):
        return (1 + np.exp(- (np.pi ** 2) * t / 4)) * np.cos(np.pi * x /2)

    a = 0
    b = 1
    t_span = [0, 0.25]
    methods = ["CN", "Ralston"]
    markdict = {4: 'x', 8: 'x', 16: 'x', 32: 'x', 64: 'x', 128: 'x', 256:"x"}
    markoffsetdict = dict(zip(methods, ['','']))
    for N in grid_size:
        dt = 1 / (8 * N ** 2)
        BC0 = BoundaryCondition(a, type="D", constraint=bc0)
        BC1 = BoundaryCondition(b, type="N", constraint=bc1)
        spgrid = SpaceGrid(a, b, N + 1)
        xsample = np.linspace(a, b, N + 1)
        # initialize the SturmLiouville system and initial-boundary value problem
        ST = SturmLiouville(spgrid, d, q)
        ibvp = IBVP(ST=ST, Nonlinear=None, IC=ic, BCs=[BC0, BC1], rhs=f)
        # compute the discretization and matrix involved
        ibvp.compute()
        for method in methods:
            json_name = "sol node={}, dt={:.2g}, for {}.json".format(N, dt, method)
            print(json_name)
            import os
            exists = os.path.isfile('../data/{}'.format(json_name))
            if exists:
                # Store configuration file values
                sol = Read_sol_json(json_name)
            else:
                # Keep presets
                if (method in ["CN", "BE"]):
                    sol = ibvp.customize_solve(t_span=t_span, type=method, max_step=dt)
                else:
                    sol = ibvp.ibvp_solve(t_span=t_span, type=method, max_step=dt)
                Write_sol_json(sol, json_name)
            ysample = np.abs(sol.y - np.array([[exact_sol(x, t) for t in sol.t] for x in xsample]))[1:,1:]
            tsample = sol.t[1:]
            sliced_xsample = xsample[1:]

            sol1 = OdeResult(t=tsample, y=ysample)
            errors = FunctionNormAlong(sol=sol1,xsample=sliced_xsample,axis=axis, type=norm_type)

            if axis=="t":
                plt.semilogy(sliced_xsample, errors, markdict[N]+markoffsetdict[method],label=json_name[4:-5])
            elif axis=="x":
                plt.semilogy(tsample, errors, markdict[N]+markoffsetdict[method],label=json_name[4:-5])
    if axis == 't':
        plt.xlabel("x")
        plt.title("x" + "-error relation in {} norm".format(norm_type))
    else:
        plt.xlabel("t")
        plt.title("t" + "-error relation in {} norm".format(norm_type))
    plt.ylabel("log(error})")
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.8), shadow=True, ncol=1)

    plt.show()

def Verify2ndAccuracy(grid_size=[4,8,16,32,64, 128], axis:str='t', norm_type:str="L1"):
    def d(x):
        return 2 + np.cos(np.pi*x)

    def q(x):
        return 0.0

    def bc0(t):
        return 1 + np.exp(- (np.pi ** 2) * t / 4)

    def bc1(t):
        return - 0.5 * np.pi * (1 + np.exp(- (np.pi ** 2) * t / 4))

    def ic(x):
        return 2 * np.cos(0.5 * np.pi * x)

    def f(x, t):
        result =  -1/ 4 * (np.pi ** 2) * np.exp(- (np.pi ** 2) * t / 4) \
                * np.cos(x * np.pi/2)*(-1 + 3 *(1 + np.exp((np.pi ** 2) * t / 4))* np.cos(x * np.pi))
        return result
    def exact_sol(x,t):
        return (1 + np.exp(- (np.pi ** 2) * t / 4)) * np.cos(np.pi * x /2)

    a = 0
    b = 1
    t_span = [0, 0.25]
    methods = ["CN", "Ralston"]
    method = methods[0]
    markdict = {4: 'x', 8: 'x', 16: 'x', 32: 'x', 64: 'x', 128: '-', 256:"x"}
    markoffsetdict = dict(zip(methods, ['','-']))
    errors2d = {"L1":[],"L2":[],"Linf":[]}
    for N in grid_size:
        dt = 1 / (8 * N ** 2)
        BC0 = BoundaryCondition(a, type="D", constraint=bc0)
        BC1 = BoundaryCondition(b, type="N", constraint=bc1)
        spgrid = SpaceGrid(a, b, N + 1)
        xsample = np.linspace(a, b, N + 1)
        # initialize the SturmLiouville system and initial-boundary value problem
        ST = SturmLiouville(spgrid, d, q)
        ibvp = IBVP(ST=ST, Nonlinear=None, IC=ic, BCs=[BC0, BC1], rhs=f)
        # compute the discretization and matrix involved
        ibvp.compute()

        json_name = "sol node={}, dt={:.2g}, for {}.json".format(N, dt, method)
        print(json_name)
        import os
        exists = os.path.isfile('../data/{}'.format(json_name))
        if exists:
            # Store configuration file values
            sol = Read_sol_json(json_name)
        else:
            # Keep presets
            if (method in ["CN", "BE"]):
                sol = ibvp.customize_solve(t_span=t_span, type=method, max_step=dt)
            else:
                sol = ibvp.ibvp_solve(t_span=t_span, type=method, max_step=dt)
            Write_sol_json(sol, json_name)
        ysample = np.abs(sol.y - np.array([[exact_sol(x, t) for t in sol.t] for x in xsample]))[1:,1:]
        tsample = sol.t[1:]
        sliced_xsample = xsample[1:]

        sol1 = OdeResult(t=tsample, y=ysample)
        errors = FunctionNormAlong(sol=sol1,xsample=sliced_xsample,axis=axis, type=norm_type)
        errors_L1 = FunctionNormAlong(sol=sol1, xsample=sliced_xsample, axis=axis, type="L1")
        errors_L2 = FunctionNormAlong(sol=sol1,xsample=sliced_xsample,axis=axis, type="L2")
        errors_Linf = FunctionNormAlong(sol=sol1,xsample=sliced_xsample,axis=axis, type="Linf")
        errors2d["L1"].append(FunctionNorm1D(errors_L1.copy(), 1/N, type="L1"))
        errors2d["L2"].append(FunctionNorm1D(errors_L2.copy(), 1/N, type="L2"))
        errors2d["Linf"].append(FunctionNorm1D(errors_Linf.copy(), 1/N, type="Linf"))
        if axis=="t":
            offset = {4:1,8:1,16:1,32:1,64:2, 128:4}[N]
            plt.semilogy(sliced_xsample[::offset], errors[::offset], markdict[N]+markoffsetdict[method],
                         label=json_name[4:-5])
        elif axis=="x":
            plt.semilogy(tsample, errors, markdict[N]+markoffsetdict[method],label=json_name[4:-5])

    # powers = [pow(2, i) for i in range(1,5)]
    for i in [2,4,8, 16, 32]:
        plt.semilogy(sliced_xsample, errors * pow(i, 2), '--', label="{} times the node=128 error".format(i))
    if axis == 't':
        plt.xlabel("x")
        plt.title("Verification of 2nd oder accuracy in {} norm".format(norm_type))
    else:
        plt.xlabel("t")
        plt.title("Verification of 2nd oder accuracy in {} norm".format(norm_type))
    plt.ylabel("log(error})")
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.6), shadow=True, ncol=1)

    plt.show()

    for key in errors2d:
        plt.loglog(grid_size, errors2d[key], 'x-', label="total error in 2D-{} norm".format(key))
    plt.loglog(grid_size, [4 * pow(g, -2) for g in grid_size], '-', label="base line of second oder arruracy")
    # plt.loglog(grid_size, [0.01 * pow(g, -4) for g in grid_size], '-', label="base line of fourth oder arruracy")

    plt.grid(True)
    plt.legend()
    plt.xlabel("node number N")
    plt.ylabel("total errors")
    plt.title("Verification of 2nd order accuracy in 2D norms for {}".format(method))
    plt.show()

def FFT_ibvp(t_span:list, x_span:list, Nt:int, Nx:int, ic:callable(float) ):
    """
    Fast calculate exact solution
    :param t_span:
    :param x_span:
    :param Nt:
    :param Nx:
    :param ic:
    :return:
    """
    a = x_span[0]
    b = x_span[1]
    t0 = t_span[0]
    tf = t_span[1]
    c = 2 * b - a
    x_span = np.linspace(a, b, 2 * Nx + 1)[:-1]
    extend_x_span = np.linspace(a, c, 4 * Nx+1)[:-1]
    t_span = np.linspace(t0, tf, Nt)
    # xsample = np.linspace(2*a-b, b, 2* Nx +1 )[:-1]
    icsample = np.array([ic(x) for x in x_span])
    ghostsample =  np.flip(np.array([ic(x) for x in np.linspace(a, b, 2 * Nx + 1)]),0)[:-1]
    extend_icsample = np.hstack([icsample, ghostsample])

    phasesample = np.array([np.exp(-1j * np.pi * x/ 2) for x in extend_x_span])
    invphasesample = np.array([np.exp(1j * np.pi * x / 2) for x in extend_x_span])

    hsample = extend_icsample * phasesample
    hhat = np.fft.ifftshift(np.fft.fft(hsample))
    ysample = np.array([extend_icsample]).T

    # phasesample = np.array([np.exp(-1j * np.pi * x/ 2) for x in x_span])
    # invphasesample = np.array([np.exp(1j * np.pi * x/ 2) for x in x_span])
    # hsample = icsample * phasesample
    # hhat = np.fft.ifftshift(np.fft.fft(hsample))
    # ysample = np.array([icsample]).T
    print(hhat.shape)
    for t in t_span[1:]:
        evolsample = np.array([np.exp(-pow(np.pi * (n+0.5), 2) * t) for n in range(-2 * Nx, 2 * Nx)])
        htsmaple = np.fft.ifft(np.fft.fftshift(hhat * evolsample))
        utsample = np.array([invphasesample * htsmaple])
        ysample = np.hstack([ysample, utsample.T])

    print(ysample.shape)
    print(x_span.shape, t_span.shape, ysample.shape)
    sol_extended = OdeResult(t=t_span, y=np.real(ysample))
    sol = OdeResult(t=t_span[:], y= np.real(ysample[0:2 * Nx, :]))
    PlotSpaceTimePDE(sol_extended, xsample=extend_x_span,
                     title="FFT sol on extended region for dx={:.2g}".format(1/1024),
                     xcount=50, tcount=50)
    PlotSpaceTimePDE(sol, xsample=x_span,
                     title="FFT correct sol for dx={:.2g}".format(1/1024),
                     xcount=25, tcount=50)
    Write_sol_json(sol=sol, file_name="FFT for tf={}.json".format(tf))
    return sol, x_span

def CN_BE_Compare(t:float=0.01, tf:float=0.25):
    def d(x):
        return 1

    def q(x):
        return 0.0

    def bc0(t):
        return 0

    def bc1(t):
        return 0

    def ic(x):
        if x < 0.5:
            return 0
        else:
            return 1
        # return np.sin(np.pi * x /2)
    def f(x, t):
        return 0

    a = 0
    b = 1
    t_span=[0,tf]
    N = 128
    CFL = 100

    dts=[CFL / N ** 2]

    methods = ["BE", "CN"]
    markerdict = dict(zip(methods, ["x-", "--"]))
    BC0 = BoundaryCondition(a, type="D", constraint=bc0)
    BC1 = BoundaryCondition(b, type="N", constraint=bc1)

    spgrid = SpaceGrid(a, b, N + 1)
    xsample = np.linspace(a, b, N + 1)
    FFT_name = "FFT for tf={}.json".format(tf)
    import os
    exists = os.path.isfile('../data/{}'.format(FFT_name))
    if exists:
        # Store configuration file values
        print("Read existing FFT solution file")
        FFTsol = Read_sol_json(FFT_name)
        FFT_xspan = np.linspace(a, b, 1024 + 1)[:-1]
    else:
        FFTsol, FFT_xspan= FFT_ibvp(t_span,x_span=[a,b], Nt=1000, Nx=512, ic=ic)
    nth = int(t / (tf/ 1000))
    FFTsol_at_t = FFTsol.y[:, nth]
    # initialize the SturmLiouville system and initial-boundary value problem
    ST = SturmLiouville(spgrid, d, q)
    ibvp = IBVP(ST=ST, Nonlinear=None, IC=ic, BCs=[BC0, BC1], rhs=f)
    # compute the discretization and matrix involved
    ibvp.compute()
    sol_at_t_dict = {}
    for method in methods:
        for dt in dts:
            json_name = "sol_compare tspan=({},{}) node={}, dt={:.2g}, for {}.json".format(0, tf,N, dt, method)
            print(json_name)
            import os
            exists = os.path.isfile('../data/{}'.format(json_name))
            if exists:
                # Store configuration file values
                print("Read existing solution file")
                sol = Read_sol_json(json_name)
            else:
                # Keep presets
                if (method in ["CN", "BE"]):
                    sol = ibvp.customize_solve(t_span=t_span, type=method, max_step=dt)
                    Write_sol_json(sol, json_name)
                else:
                    sol = ibvp.ibvp_solve(t_span=t_span, type=method, max_step=dt)
                    Write_sol_json(sol, json_name)
            nth = int(t/dt)+1
            sol_at_t_dict[(method, dt)]= sol.y[:,nth]
            if method == "CN":
                PlotSpaceTimePDE(sol, xsample, title="Space time sol for CN N={}, CFL={}".format(N, CFL), xcount=5, tcount=50)

    for key in sol_at_t_dict:
        method = key[0]
        dt = key[1]
        plt.semilogy(xsample[10::2], np.abs(sol_at_t_dict[key])[10::2],markerdict[method],
                 label="solution at t={}, dt={}, CFL={}, method={}".format(t,dt, N * N * dt, method))
    plt.semilogy(FFT_xspan[100:], FFTsol_at_t[100:], label="FFT sol at t={}".format(t))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=True, shadow=True, ncol=1)
    plt.xlabel("x")
    plt.ylabel("u(t={})".format(t))
    plt.title("Comapre for CFL={} for grid num={}".format(N * N * dt,N))
    plt.grid(True)
    plt.show()


    # PlotSpaceTimePDE(sol1, xsample,
    #                  title="Space-Time-Solution node={}, dt={}, for {}, CFL={}".format(N,dt, method, N * N * dt),
    #                  tcount=2, xcount=2)
    # sol2 = FFT_ibvp(t_span=t_span, x_span=[a,b], Nt=1000, Nx=512, ic=ic)

def NonlinearPDE():
    a = -1
    b = 1
    u0 = 26.16
    tf = 2.12
    dt = 0.01
    N= 64
    method = "Radau"
    def d(x):
        return 2 + np.cos(2 * np.pi * x)

    def q(x):
        return 0

    def nonlinear(x):
        return x ** 2


    def f(x, t):
        return 0

    def ic(x):
        return u0 * np.cos(np.pi * x /2) ** 100


    BC1 = BoundaryCondition(a, type="D", constraint=lambda t: 0)
    BC2 = BoundaryCondition(b, type="D", constraint=lambda t: 0)
    # geometry_dict = {"Dirichlet": -1, "Neumann": 0}
    spgrid = SpaceGrid(a, b, N + 1)
    xsample = np.linspace(a, b, N + 1)
    # xsample = xsample[-geometry_dict[BC1.type]: xsample.shape[0]+geometry_dict[BC2.type]]
    # xsample=xsample[1:]
    ST = SturmLiouville(spgrid, d, q)
    ibvp =IBVP(ST=ST,Nonlinear=nonlinear, IC=ic, BCs=[BC1,BC2], rhs= lambda x, t: 0)
    ibvp.compute()
    # print(ibvp.odefunc(ic(xsample)[0:-1], 0))
    sol = ibvp.ibvp_solve(t_span=[0, tf], type=method, max_step=dt)
    Write_sol_json(sol,file_name="sol.json")
    # sol = ibvp.customize_solve(t_span=[0,10], max_step=0.01)
    sol=Read_sol_json(file_name="sol.json")
    # Animate_1d_wave(sol, xsample=xsample)
    sol_atx0 = sol.y[N//2]
    # print("ahah",sol_atx0)
    ta = 3* tf/4
    nta = int(ta//dt)
    nta = 0
    sol_atx0 = sol_atx0[nta:]
    tsample = sol.t[nta:]
    print("t",tsample)
    print("u", sol_atx0.shape)
    T= 2.125
    plt.plot(tsample, sol_atx0,'x', label="f(t)=u(x=0,t)")
    plt.plot(tsample, [1/(T-t) for t in tsample], label="1/({}-t)".format(T))
    plt.legend()
    plt.title("finite time blow up and $1/(T-t)$")
    plt.grid(True)
    plt.xlabel("t")
    plt.ylabel("u(x=0, t)")
    plt.show()

    # PlotSpaceTimePDE(sol, xsample, xcount=2, tcount=5, title="u0={}, N={}, dt={}, method:{}".format(u0, N, dt, "SBDF2"))

def DGreen(N:int, CFL:float):
    dx = 2/N
    dt = (CFL* dx ** 2)
    a = -1
    b = 1
    tf = 0.25
    method = "CN"
    def d(x):
        return 1

    def q(x):
        return 0.0

    def bc0(t):
        return 0

    def bc1(t):
        return 0

    def Ddelta(x):
        if (x < dx/4  and x > -dx/4):
            return 1
        else:
            return 0

    def f(x, t):

        return 0

    BC1 = BoundaryCondition(a, type="N", constraint=lambda t: 0)
    BC2 = BoundaryCondition(b, type="N", constraint=lambda t: 0)
    # geometry_dict = {"Dirichlet": -1, "Neumann": 0}

    spgrid = SpaceGrid(a, b, N + 1)
    xsample = np.linspace(a, b, N + 1)
    # xsample = xsample[-geometry_dict[BC1.type]: xsample.shape[0]+geometry_dict[BC2.type]]
    # xsample=xsample[1:]
    ST = SturmLiouville(spgrid, d, q)
    ibvp = IBVP(ST=ST, Nonlinear=None, IC=Ddelta, BCs=[BC1,BC2], rhs=lambda x, t: 0)
    ibvp.compute()
    # print(ibvp.odefunc(ic(xsample)[0:-1], 0))
    sol = ibvp.customize_solve(t_span=[0, tf], type=method, max_step=dt)
    sol.t = sol.t[1:]
    sol.y = sol.y[:,1:]
    min = sol.y.min()
    PlotSpaceTimePDE(sol, xsample, xcount=2, tcount=2,
                     title="Discrete Green for {},N={}, CFL={}, min={:.2g}".format(method,N, CFL,min))
    plt.plot(xsample, sol.y[:,0])
    plt.show()

    # FFT_ibvp(t_span=[0,tf], x_span=[a,b], Nt= 1000, Nx=10 *N, ic=Ddelta)

# discretization_error_plot()
# BVP_solve_plot()
# IBVP_solve_plot()
# LTEplot3D()
# LTEplot2D(axis='t', norm_type="L2")
# Verify2ndAccuracy(norm_type="Linf")

# FFT_ibvp([0,1],[0,1],100,100, ic=ic)
# CN_BE_Compare(t=8, tf=10)
NonlinearPDE()
# DGreen(64, 1.5)