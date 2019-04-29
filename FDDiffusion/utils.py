from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from .IBVPs import OdeResult
import numpy as np
import json


def PlotSpaceTimePDE(sol, xsample, title:str="Space-Time-PDE", tcount=10, xcount=10, error_plot:bool=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    time_events = sol.t

    ysamples = sol.y
    zmax, zmin = ysamples.max(), ysamples.min()
    time_events, xsample = np.meshgrid(time_events, xsample)
    norm = plt.Normalize(ysamples.min(), ysamples.max())
    colors = cm.viridis(norm(ysamples))
    rcount, ccount, _ = colors.shape
    print(rcount, ccount)
    print(xsample.shape, time_events.shape, ysamples.shape)
    surf = ax.plot_surface(time_events, xsample, ysamples,facecolors=colors,
                           rcount=rcount//xcount, ccount=ccount//tcount, shade=False)
    surf.set_facecolor((0, 0, 0, 0))
    # Customize the z axis.
    # ax.set_zlim(1.1 * zmin, 1.1 * zmax)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    if(error_plot):
        ax.set_zlabel("$log(abs(u-u_{exact})$")
    else:
        ax.set_zlabel("u")


    # Add a color bar which maps values to colors.
    # fig.colorbar(surf)
    plt.title(title)
    plt.show()

def Animate_1d_wave(sol,xsample, frames=120, interval=1, multiplier=1,filename="test.mp4"):
    """
    :param sol: OdeResult
    :param frames: int
    :param interval: int
    :param filename: string
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_xlim((xsample[0] * 1.1, xsample[-1]* 1.1))
    ax.set_ylim((0, 30))
    frames = sol.y.shape[1]
    line, = ax.plot([], [], lw=2)
    plt.grid(True)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)

    # animation function. This is called sequentially
    def animate(num):
        ysample = sol.y[:,num * multiplier]
        line.set_data(xsample, ysample)
        line.set_marker('o')
        return (line,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval, blit=True)

    anim.save(filename)

def FunctionNorm1D(ysample:np.array, step:float, type="L1"):
    """
    compute the function of discrete array given the step
    :param array:
    :param step:
    :param type:
    :return: float
    """
    ysample= ysample.copy()
    typedict = {"L2": 2, "L1": 1, "Linf": np.inf}
    assert(type in typedict)
    if type == "L1":
        ysample *= step
    elif type == "L2":
        ysample *= np.sqrt(step)
    func_norm = np.linalg.norm(ysample, typedict[type])
    return func_norm

def FunctionNormAlong(sol, xsample, axis="t", type="L1"):
    """
    :param sol:
    :param xsample:
    :param axis:
    :param type:
    :return: np.array of format sol.t or xsmaple, depending on the choice of axis
    """
    assert axis in ["t","x"]
    dt = sol.t[1]- sol.t[0]
    h = xsample[1]-xsample[0]
    ysample = sol.y.copy()
    if axis == "t":
        result = np.array([FunctionNorm1D(ysample[i,:].copy(),dt,type=type) for i in range(xsample.shape[0])])
    else:
        result = np.array([FunctionNorm1D(ysample[:, i].copy(), h, type=type) for i in range(sol.t.shape[0])])
    return result

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def Write_sol_json(sol, file_name:str):
    with open('../data/{}'.format(file_name), 'w') as f:
        # data = json.load(f)
        json.dump(sol, f, sort_keys=True, cls=NumpyEncoder)

def Read_sol_json(file_name:str):
    with open('../data/{}'.format(file_name), 'r') as f:
        dic = dict(json.load(f))
        result=OdeResult(t=np.array(dic["t"]), y=np.array(dic["y"]))

        # sol= json.dumps(sol.__dict__)
        # print(sol)
        return result

