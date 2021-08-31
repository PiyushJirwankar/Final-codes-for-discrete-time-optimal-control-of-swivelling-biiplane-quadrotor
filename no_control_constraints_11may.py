import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import time
# from RK4_project import RK4 as RK4

start_time = time.time()

EPS = np.MachAr().eps


def _get_epsilon(x, s, epsilon, n):
    if epsilon is None:
        h = EPS**(1. / s) * np.maximum(np.abs(x), 0.1)
    else:
        if np.isscalar(epsilon):
            h = np.empty(n)
            h.fill(epsilon)
        else:  # pragma : no cover
            h = np.asarray(epsilon)
            if h.shape != x.shape:
                raise ValueError("If h is not a scalar it must have the same"
                                 " shape as x.")
    return h


def approx_fprime(x, f, epsilon=None, args=(), kwargs=None, centered=True):
    """
    Gradient of function, or Jacobian if function fun returns 1d array
    Parameters
    ----------
    x : array
        parameters at which the derivative is evaluated
    fun : function
        `fun(*((x,)+args), **kwargs)` returning either one value or 1d array
    epsilon : float, optional
        Stepsize, if None, optimal stepsize is used. This is _EPS**(1/2)*x for
        `centered` == False and _EPS**(1/3)*x for `centered` == True.
    args : tuple
        Tuple of additional arguments for function `fun`.
    kwargs : dict
        Dictionary of additional keyword arguments for function `fun`.
    centered : bool
        Whether central difference should be returned. If not, does forward
        differencing.
    Returns
    -------
    grad : array  
        gradient or Jacobian
    Notes
    -----
    If fun returns a 1d array, it returns a Jacobian. If a 2d array is returned
    by fun (e.g., with a value for each observation), it returns a 3d array
    with the Jacobian of each observation with shape xk x nobs x xk. I.e.,
    the Jacobian of the first observation would be [:, 0, :]
    """
    kwargs = {} if kwargs is None else kwargs
    x = np.atleast_1d(x)  # .ravel()
    n = len(x)
    f0 = f(*(x,) + args, **kwargs)
    dim = np.atleast_1d(f0).shape  # it could be a scalar
    grad = np.zeros((n,) + dim, float)
    ei = np.zeros(np.shape(x), float)
    if not centered:
        epsilon = _get_epsilon(x, 2, epsilon, n)
        for k in range(n):
            ei[k] = epsilon[k]
            grad[k, :] = (f(*(x + ei,) + args, **kwargs) - f0) / epsilon[k]
            ei[k] = 0.0
    else:
        epsilon = _get_epsilon(x, 3, epsilon, n) / 2.
        for k in range(n):
            ei[k] = epsilon[k]
            grad[k, :] = (f(*(x + ei,) + args, **kwargs) -
                          f(*(x - ei,) + args, **kwargs)) / (2 * epsilon[k])
            ei[k] = 0.0
    # return grad
    # return grad.T
    # grad = grad.squeeze()
    axes = list(range(grad.ndim))
    axes[:2] = axes[1::-1]
    return np.transpose(grad, axes=axes)


m = 0.8
g = 9.81
T0 = m*g
l = 0.42

start = 0
end = 10
N = 20*(end-start)+1
time_plot = np.linspace(start, end, N)
h = time_plot[1]-time_plot[0]

omega0 = 0.0
omegaNm1 = 0.0
c = np.pi/2.0

M0 = 0.0
Mnm1 = 0.0
d = 2.5

Md0 = 0.0
Mdnm1 = 0.0
e = 2.0

u_lim = 1.0
Izz = 0.02275
Iyy = 0.0136
Ixx = 0.01111

th0_deg = 0.0
th0 = th0_deg*np.pi/180.0
R0 = np.array([[np.cos(th0), -np.sin(th0)],
               [np.sin(th0), np.cos(th0)]])
thn_deg = 89.9999
thn = thn_deg*np.pi/180.0
Rn = np.array([[np.cos(thn), -np.sin(thn)],
               [np.sin(thn), np.cos(thn)]])

delta_arr = np.zeros(N)
delta_dot_arr = np.zeros(N)


def hat(x):
    return np.array([[0.0, -x],
                     [x, 0.0]])


def vee(X):
    return X[1, 0]


def Rotm(t):  # Defining rotation matrix as required in definition of Rho
    out = np.array([[np.cos(t), -np.sin(t)],
                    [np.sin(t), np.cos(t)]])
    return out


def F_int(t, Rk, Rkp1):
    val = np.arccos(np.dot(Rk.transpose(), np.dot(Rkp1, Rotm(t)))[0, 0])
    return hat(val)


def rho2zeta(Rk, Rkp1, rhok):  # Function that transforms rho to zeta
    eps = 1E-4
    Fv = (F_int(0.0+eps, Rk, Rkp1) - F_int(0.0-eps, Rk, Rkp1))/(2.0*eps)
    v = np.array([[0.0, -1.0], [1.0, 0.0]])
    v_scalar = vee(v)
    rho_scalar = vee(rhok)
    F_scalar = vee(Fv)
    zeta_scalar = (rho_scalar*v_scalar)/F_scalar
    zetak = hat(zeta_scalar)
    return zetak


def f(x, u):  # Function that outputs the system dynamics
    w = x[0]
    M = x[1]
    Md = x[2]
    out = np.zeros(3)
    out[0] = w + h*M/Izz
    out[1] = M + h*Md
    out[2] = Md + h*u
    return out


def ctrlk(xik):  # Bang-off-bang control function
    # if -u_lim<= h*xik[2] <= u_lim:
    #     u = h*xik[2]
    # elif h*xik[2] > u_lim:
    #     u = u_lim
    # else:
    #     u = -u_lim
    u = h*xik[2]
    return u


def jac_calc(jacobian, X):  # Jacobian calclation after numerically calculating jacobian to include generalised gradient
    Jac = jacobian
    for i in range(N-1):
        xi2 = X[6*i+5]
        if h*xi2 < -u_lim or h*xi2 > u_lim:
            NegPartialCtrlPartialXi = 0.0
        elif -u_lim <= h*xi2 <= u_lim:
            NegPartialCtrlPartialXi = -h
        Jac[6*i+2, 6*i+5] = NegPartialCtrlPartialXi
    return Jac


def alphak(xkp1, xk, xik):  # State dynamics part of multiple shooting method
    uk = ctrlk(xik)
    xkp1_calc = f(xk, uk)
    return xkp1 - xkp1_calc


def Fk(wk):  # Rotational kinematics
    out = np.array([[np.cos(wk*h), np.sin(wk*h)],
                    [-np.sin(wk*h), np.cos(wk*h)]])
    return out


def Cw(x0, xn):  # Initial and final conditons in multiple shooting method
    a1 = x0[0] - omega0
    a2 = x0[1] - M0
    a3 = x0[2] - Md0
    a4 = xn[0] - omegaNm1
    a5 = xn[1] - Mnm1
    a6 = xn[2] - Mdnm1
    out = np.array([a1, a2, a3, a4, a5, a6])
    return out


def project_w(wk):  # Projection of angular velocity onto feasible domain
    if abs(wk) < 0.0001:
        return 0.0
    else:
        return np.sign(wk)*np.minimum(c, np.absolute(wk))


def project_M(Mk):  # Projection of Moment onto feasible domain
    if abs(Mk) < 0.0001:
        return 0.0
    else:
        return np.sign(Mk)*np.minimum(d, np.absolute(Mk))


def project_Md(Mdk):  # Projection of Moment derivative onto feasible domain
    if abs(Mdk) < 0.0001:
        return 0.0
    else:
        return np.sign(Mdk)*np.minimum(e, np.absolute(Mdk))


def project_xi2(xik2, Mdkp1_tilde, Mdk_tilde):
    uk_tilde = (Mdkp1_tilde-Mdk_tilde)/h
    if uk_tilde >= u_lim:
        return xik2
    elif uk_tilde <= -u_lim:
        return xik2
    else:
        return uk_tilde/h


def Mk_tilde_calc(wkp1_tilde, wk_tilde):
    Mk_tilde = Izz*(wkp1_tilde - wk_tilde)/h
    if abs(Mk_tilde) < d:
        return Mk_tilde
    elif Mk_tilde < -d:
        return -d
    else:
        return d


def Mdk_tilde_calc(Mkp1_tilde, Mk_tilde):
    Mdk_tilde = (Mkp1_tilde - Mk_tilde)/h
    if abs(Mdk_tilde) < e:
        return Mdk_tilde
    elif Mdk_tilde < -e:
        return -e
    else:
        return e


def betak(xikm1, xik, xk, zetak):  # Adjoint equation part of  multiple shooting method
    M = xk[1]
    arr1 = np.array([0.0, -2.0*M, 0.0]).transpose()
    arr2 = np.array([2.0*h*vee(zetak), 0.0, 0.0]).transpose()
    arr3 = np.array([xik[0], h*xik[0]/Izz + xik[1],
                    h*xik[1] + xik[2]]).transpose()
    out = arr1 + arr2 + arr3
    return xikm1 - out


def rhok_calc_from_rho0(X, k):
    rho0_scalar = X[6*(N-1)+3]
    rho0 = hat(rho0_scalar)
    Fk_mat = np.eye(2)
    mat = np.eye(2)
    for i in range(k):
        omegak = X[6*(i+1)]
        Fk_mat = Fk(omegak)
        mat = np.dot(mat, Fk_mat)
    # rhok = np.dot(mat , np.dot(rho0 , mat.transpose()))
    rhok = np.dot(mat.transpose(), np.dot(rho0, mat))
    return rhok


length = 6*(N-1)+4


def shoot_func(X):
    Marr = np.zeros(length)
    for i in range(N-2):
        xk = X[6*i: 6*i+3]
        xik = X[6*i+3: 6*i+6]
        xkp1 = X[6*(i+1): 6*(i+1)+3]
        zetakp1 = rhok_calc_from_rho0(X, i+1)
        xikp1 = X[6*(i+1)+3: 6*(i+1)+6]
        alph = alphak(xkp1, xk, xik)
        beta = betak(xik, xikp1, xkp1, zetakp1)
        if abs(abs(xkp1[0]) - c) < 0:
            beta[0] = xkp1[0]**2 - c**2
        if abs(abs(xkp1[1]) - d) < 0:
            beta[1] = xkp1[1]**2 - d**2
        if abs(abs(xkp1[2]) - e) < 0:
            beta[2] = xkp1[2]**2 - e**2

        Marr[6*i: 6*i+3] = alph
        Marr[6*i+3: 6*i+6] = beta

    x0 = X[:3]
    xnm1 = X[6*(N-1): 6*(N-1)+3]
    xnm2 = X[6*(N-2): 6*(N-2)+3]
    xinm2 = X[6*(N-2)+3: 6*(N-2)+6]
    alph = alphak(xnm1, xnm2, xinm2)

    Marr[6*(N-2): 6*(N-2)+3] = alph
    Marr[6*(N-2)+3: 6*(N-1)+3] = Cw(x0, xnm1)

    rot_diff = np.dot(Rn.transpose(), R0)
    prod_fk = np.eye(2)
    for i in range(N-1):
        wk = X[6*i]
        rot_inc = Fk(wk)
        prod_fk = np.dot(prod_fk, rot_inc)

    rot_err = np.dot(rot_diff, prod_fk)
    Cr = np.arcsin(rot_err[1, 0])

    Marr[6*(N-1)+3] = Cr
    return Marr


X = np.zeros(length)
# for i in range(N-1):
#     X[6*i] = 1.5 - 1.5*np.cos(h*i*np.pi/4.0)

for j in range(N-1):
    X[6*(j)] = project_w(X[6*(j)])
    X[6*(j)+1] = project_M(X[6*(j)+1])
    X[6*(j)+2] = project_Md(X[6*(j)+2])
X[0] = project_w(X[6*(N-1)])
X[6*(N-1)+1] = project_M(X[6*(N-1)+1])
X[6*(N-1)+2] = project_Md(X[6*(N-1)+2])

func_eval = shoot_func(X)
norm = np.linalg.norm(func_eval)

# jacobian = approx_fprime(X, shoot_func)
# rank = np.linalg.matrix_rank(jacobian)

toll = 1E-4
i = 1
# if Mdnm1-Md0 > (end-start)*u_lim:
#     raise ValueError("No feasible solution")
# else:
#     while norm > toll:
#         jacobian = approx_fprime(X, shoot_func)
#         # jacobian = jac_calc(jacobian, X)
#         rank = np.linalg.matrix_rank(jacobian)
#         if rank == length:
#             print(i)
#             jacinv = np.linalg.inv(jacobian)
#             delX = np.dot(jacinv, -func_eval)
#             X = X + delX
#             # X = X - (1.0/(i+1.0))*func_eval
#             # X = scipy.optimize.fsolve(shoot_func, X)
#             for j in range(N-1):
#                 X[6*j] = project_w(X[6*j])
#             for j in range(N-1):
#                 X[6*j+1] = project_M(X[6*j+1]) #Mk_tilde_calc(X[6*(j+1)], X[6*j])
#             for j in range(N-1):
#                 X[6*j+2] = project_Md(X[6*j+2]) #Mdk_tilde_calc(X[6*(j+1)+1], X[6*j+1])
#             X[6*(N-1)] = project_w(X[6*(N-1)])
#             X[6*(N-1)+1] = project_M(X[6*(N-1)+1])
#             X[6*(N-1)+2] = project_Md(X[6*(N-1)+2])
#             # for j in range(N-1):
#             #     X[6*j+5] = project_xi2(X[6*j+5], X[6*(j+1)+2], X[6*j+2])
#             func_eval = shoot_func(X)
#             norm = np.linalg.norm(func_eval)
#             print(norm)
#             i += 1
#         else:
#             raise ValueError("Singular Jacobian")

# def func_min(X):
#     return np.linalg.norm(shoot_func(X))

lower_bound = np.zeros(length)
upper_bound = np.zeros(length)
for i in range(N):
    lower_bound[6*i] = -c
    lower_bound[6*i+1] = -d
    lower_bound[6*i+2] = -e
    upper_bound[6*i] = c
    upper_bound[6*i+1] = d
    upper_bound[6*i+2] = e
for i in range(N-1):
    lower_bound[6*i+3] = -100000.0
    lower_bound[6*i+4] = -100000.0
    lower_bound[6*i+5] = -100000.0
    upper_bound[6*i+3] = 100000.0
    upper_bound[6*i+4] = 100000.0
    upper_bound[6*i+5] = 100000.0
lower_bound[-1] = -100000.0
upper_bound[-1] = 100000.0

# list_lb = list(lower_bound)
# list_ub = list(upper_bound)
# bounds = scipy.optimize.Bounds(list_lb, list_ub)
# # print(bounds)
# # result = scipy.optimize.minimize(func_min, X, bounds=bounds)  ## Not included control constraints
# result = scipy.optimize.minimize(func_min, X)
# X = result.x

X = scipy.optimize.fsolve(shoot_func, X)


def constrainedFunction(x, f, lower, upper, minIncr=0.001):
    x = np.asarray(x)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    xBorder = np.where(x < lower, lower, x)
    xBorder = np.where(x > upper, upper, xBorder)
    fBorder = f(xBorder)
    distFromBorder = (np.sum(np.where(x < lower, lower-x, 0.))
                      + np.sum(np.where(x > upper, x-upper, 0.)))
    return (fBorder + (fBorder
                       + np.where(fBorder > 0, minIncr, -minIncr))*distFromBorder)


# result = scipy.optimize.root(constrainedFunction, X, args=(shoot_func, lower_bound, upper_bound))
# print(result)
# X = result.x
print(np.linalg.norm(shoot_func(X)))

omg_plot = np.zeros(N)
M_plot = np.zeros(N)
Md_plot = np.zeros(N)
xi_plot = np.zeros((3, N-1))
u_plot = np.zeros(N-1)
R_plot = np.zeros((2, 2, N))
R_plot[:, :, 0] = R0
R_plot_scalar = np.zeros(N)
R_plot_scalar[0] = np.arcsin(R_plot[1, 0, 0])


def tau_calc(u, delta, delta_dot, omega):
    t1 = u*np.cos(delta)*np.cos(delta)
    t2 = 4.0*l*T0*np.tan(delta)*delta_dot*delta_dot
    t3 = -2.0*l*T0
    nu = (t1+t2)/t3
    term1 = nu*Ixx
    term2 = 0.5*(Iyy-Izz)*np.sin(2.0*delta)*(omega*omega)
    tau_delta = term1 - term2
    return tau_delta


for i in range(N-1):
    omg_plot[i] = X[6*i]
    # print(omg_plot[i])
    M_plot[i] = X[6*i+1]
    Md_plot[i] = X[6*i+2]
    xi_plot[:, i] = X[6*i+3: 6*i+6]
    R_plot[:, :, i+1] = np.dot(R_plot[:, :, i], Fk(omg_plot[i]))
    R_plot_scalar[i+1] = np.arcsin(R_plot[1, 0, i+1])*180.0/np.pi
    u_plot[i] = ctrlk(X[6*i+3: 6*i+6])
    tau_delt = tau_calc(u_plot[i], delta_arr[i], delta_dot_arr[i], omg_plot[i])
    delta_arr[i+1] = delta_arr[i] + h*delta_dot_arr[i]
    delta_dot_arr[i+1] = delta_dot_arr[i] + \
        (h/Ixx)*(tau_delt - 0.5*np.sin(2.0 *
                                       delta_arr[i])*Izz*omg_plot[i]*omg_plot[i])

end_time = time.time()
total_time = end_time-start_time
print("Total time for 10 sec simulation = ", total_time)


omg_plot[N-1] = X[6*(N-1)]
M_plot[N-1] = X[6*(N-1)+1]
Md_plot[N-1] = X[6*(N-1)+2]

# for i in range(N):
#     delta[i] = np.arctan(M_plot[i]/(-2.0*l*T0))
#     delta_dot[i] = Md_plot[i]*np.cos(delta[i])*np.cos(delta[i])/(-2.0*l*T0)

np.savetxt('10secmin_omg_array.csv', omg_plot, delimiter=',')
np.savetxt('10secmin_M_array.csv', M_plot, delimiter=',')
np.savetxt('10secmin_Md_array.csv', Md_plot, delimiter=',')
np.savetxt('10secmin_xi_array.csv', xi_plot, delimiter=',')
np.savetxt('10secmin_theta_plot.csv', R_plot_scalar, delimiter=',')
np.savetxt('10secmin_u_array.csv', u_plot, delimiter=',')
np.savetxt('10secmin_delta_array.csv', delta_arr, delimiter=',')


fig, ax = plt.subplots(4, 2)
ax[0, 0].plot(time_plot, omg_plot)  # , drawstyle='steps-pre')
ax[0, 0].set_xlabel('Time (sec)')
ax[0, 0].set_ylabel('$ \\omega \\:\\:\\: (rad/\\sec)$')
# ax[0,0].legend(['X1(k)'], fontsize='xx-small')

ax[0, 1].plot(time_plot, M_plot)  # , drawstyle='steps-pre')
ax[0, 1].set_xlabel('Time (sec)')
ax[0, 1].set_ylabel('$M \\:\\:\\: (kg.m^2/{sec}^2)$')
# ax[0,1].legend(['X2(k)'], fontsize='xx-small')

# , drawstyle='steps-pre')#, drawstyle = 'steps-pre')
ax[1, 0].plot(time_plot[:-1], u_plot)
ax[1, 0].set_xlabel('Time (sec)')
ax[1, 0].set_ylabel('$u \\:\\:\\: (kg.m^2/{sec}^4)$')
# ax[1,0].legend(['X3(k)'], fontsize='xx-small')

ax[1, 1].plot(time_plot, Md_plot)  # , drawstyle='steps-pre')
ax[1, 1].set_xlabel('Time (sec)')
ax[1, 1].set_ylabel('$\\dot{M} \\:\\:\\: (kg.m^2/{sec}^3)$')

ax[2, 0].plot(time_plot, R_plot_scalar)  # , drawstyle='steps-pre')
ax[2, 0].set_xlabel('Time (sec)')
ax[2, 0].set_ylabel('$\\theta \\:\\:\\: (\\deg)$')

ax[2, 1].plot(time_plot[:-1], xi_plot[0, :], 'r')  # , drawstyle='steps-pre')
ax[2, 1].plot(time_plot[:-1], xi_plot[1, :], 'b')  # , drawstyle='steps-pre')
ax[2, 1].plot(time_plot[:-1], xi_plot[2, :], 'g')  # , drawstyle='steps-pre')
ax[2, 1].set_xlabel('Time (sec)')
ax[2, 1].set_ylabel('$\\xi$')
ax[2, 1].legend(["$\\xi_1$", "$\\xi_2$", "$\\xi_3$"])

ax[3, 0].plot(time_plot, (180.0/np.pi)*delta_arr)  # , drawstyle='steps-pre')
ax[3, 0].set_xlabel('Time (sec)')
ax[3, 0].set_ylabel('$\\delta$ (deg)')

# ax[3,1].plot(time_plot, (180.0/np.pi)*delta_dot)
# ax[3,1].set_xlabel('Time (sec)')
# ax[3,1].set_ylabel('delta_dot (deg/sec)')

plt.show()
