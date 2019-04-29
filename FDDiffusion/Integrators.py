from scipy.integrate._ivp.rk import RungeKutta, rk_step
from scipy.integrate import OdeSolver
import numpy as np

# this is a copy of rk_step for illustration reason
def rk_step(fun, t, y, f, h, A, B, C, E, K):
    """Perform a single Runge-Kutta step.

    This function computes a prediction of an explicit Runge-Kutta method and
    also estimates the error of a less accurate method.

    Notation for Butcher tableau is as in [1]_.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system.
    t : float
        Current time.
    y : ndarray, shape (n,)
        Current state.
    f : ndarray, shape (n,)
        Current value of the derivative, i.e. ``fun(x, y)``.
    h : float
        Step to use.
    A : list of ndarray, length n_stages - 1
        Coefficients for combining previous RK stages to compute the next
        stage. For explicit methods the coefficients above the main diagonal
        are zeros, so `A` is stored as a list of arrays of increasing lengths.
        The first stage is always just `f`, thus no coefficients for it
        are required.
    B : ndarray, shape (n_stages,)
        Coefficients for combining RK stages for computing the final
        prediction.
    C : ndarray, shape (n_stages - 1,)
        Coefficients for incrementing time for consecutive RK stages.
        The value for the first stage is always zero, thus it is not stored.
    E : ndarray, shape (n_stages + 1,)
        Coefficients for estimating the error of a less accurate method. They
        are computed as the difference between b's in an extended tableau.
    K : ndarray, shape (n_stages + 1, n)
        Storage array for putting RK stages here. Stages are stored in rows.

    Returns
    -------
    y_new : ndarray, shape (n,)
        Solution at t + h computed with a higher accuracy.
    f_new : ndarray, shape (n,)
        Derivative ``fun(t + h, y_new)``.
    error : ndarray, shape (n,)
        Error estimate of a less accurate method.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """
    K[0] = f
    for s, (a, c) in enumerate(zip(A, C)):
        dy = np.dot(K[:s + 1].T, a) * h
        K[s + 1] = fun(t + c * h, y + dy)

    y_new = y + h * np.dot(K[:-1].T, B)
    f_new = fun(t + h, y_new)

    K[-1] = f_new
    error = np.dot(K.T, E) * h

    return y_new, f_new, error

class BS_RK23(RungeKutta):
    """
    It is basically identical to the built-in class of RK23,
    why don't we just override RK23?
    for illustration
    """
    order = 3
    n_stages = 3
    C = np.array([1 / 2, 3 / 4])
    A = [np.array([1 / 2]),
         np.array([0, 3 / 4])]
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    E = np.array([5 / 72, -1 / 12, -1 / 9, 1 / 8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2 / 3],
                  [0, 4 / 3, -8 / 9],
                  [0, -1, 1]])

    def _step_impl(self):
        # print("called")
        t = self.t
        y = self.y
        # we will stick to a max_step untill we reach the end
        h = self.max_step
        t_new = t + h
        y_new, f_new, error = rk_step(self.fun, t, y, self.f, h, self.A,
                                      self.B, self.C, self.E, self.K)
        # this error is produced by comparing two RK23 methods
        # but we will never use it
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h
        self.f = f_new
        return True, None

class Ralston(RungeKutta):
    order = 2
    n_stages = 2
    C = np.array([2/3])
    A = [np.array([2 / 3])]
    B = np.array([1/4, 3/4])
    E = np.array([5 / 72, -1 / 12, -1 / 9])
    # P = np.array([[1, -4 / 3, 5 / 9],
    #               [0, 1, -2 / 3],
    #               [0, 4 / 3, -8 / 9],
    #               [0, -1, 1]])
    def _step_impl(self):
        # print("called")
        t = self.t
        y = self.y
        # we will stick to a max_step untill we reach the end
        h = self.max_step
        t_new = t + h
        y_new, f_new, error = rk_step(self.fun, t, y, self.f, h, self.A,
                                      self.B, self.C, self.E, self.K)
        # this error is produced by comparing two RK23 methods
        # but we will never use it
        self.y_old = y
        self.t = t_new
        self.y = y_new
        self.h_abs = h
        self.f = f_new
        return True, None


