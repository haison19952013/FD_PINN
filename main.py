import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def UUV_3DOF(t, state, rud, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
             Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
             Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr):
    u, v, r, x, y, psi= state
    # Params
    m = 302.59
    x_g = 0.19252
    y_g = 0.00023
    I_zz = 119.83306
    rho = 1000
    A = 0.15597
    Dp = 0.2345
    L = 2.414
    Uc = 2.572
    # Thrust coefficients
    U = np.sqrt(u * u + v * v)
    np_ = 1.1857 * Uc + 0.0931
    J = U * (1 - 0.4177) / (np_ * Dp)
    XT = 3.3984 - 8.2263 * J + 5.4134 * J ** 2

    # Calculate for the velocities
    rhs_X = rho * A * L ** 2 / 2 * (Xrr * r ** 2) \
            + rho * A * L / 2 * Xvr * v * r \
            + rho * A / 2 * (Xuu * u ** 2 + Xvv * v ** 2 + u ** 2 * Xdrdr * rud ** 2) \
            + rho * np_ * np_ * Dp ** 4 * XT * 0.694 + m * (v * r + x_g * r ** 2)
    rhs_Y = rho * A * L ** 2 / 2 * (Yrar * r * np.abs(r)) \
            + rho * A * L / 2 * (Yur * u * r + Yvar * v * np.abs(r)) \
            + rho * A / 2 * (Yuv * u * v + Yvav * v * np.abs(v) + u ** 2 * (Y0 + Ydr * rud + Ydradr * rud * np.abs(rud))) - m*(u*r - y_g*r**2)
    rhs_N = rho * A * L ** 3 / 2 * (Nrar * r * np.abs(r)) \
            + rho * A * L * L / 2 * (Nur * u * r + Nvar * v * np.abs(r)) \
            + rho * A * L / 2 * (Nuv * u * v + Nvav * v * np.abs(v) + u ** 2 * (
            N0 + Ndr * rud + Ndradr * rud * np.abs(rud))) - m*(x_g*u*r + y_g *v*r)
    F = np.array([[rhs_X],
                  [rhs_Y],
                  [rhs_N]])

    M = np.array([[m - rho * A * L / 2 * Xud, 0, -m * y_g],
                  [0, m - rho * A * L / 2 * Yvd, m * x_g - rho * A * L ** 2 * Yrd],
                  [-m * y_g, m * x_g - rho * A * L ** 2 / 2 * Nvd, I_zz - rho / 2 * A * L ** 3 * Nrd]])
    M_inv = np.linalg.inv(M)
    state_dot = np.dot(M_inv, F).T.tolist()[0]

    # Calculate for the trajectories
    dot_x = np.cos(psi)*u - np.sin(psi)*v
    dot_y = np.sin(psi)*u + np.cos(psi)*v
    dot_psi = r

    return state_dot + [dot_x, dot_y, dot_psi]

Xrr = -1.3502
Xud = -0.08523
Xvr = -2.5772
Xuu = -0.10543
Xvv = 3.9561
Xdrdr = -0.0218

Yrd = 0.7695
Yrar = -0.8593
Yvd = -1.4749
Yur = 1.7231
Yvar = 0.5007
Yuv = -1.6621
Yvav = -3.9107
Y0 = 0.0000
Ydr = 1.0607
Ydradr = -2.2453

Nrd = -0.2950
Nrar = 0.4917
Nvd = 0.3479
Nur = -0.2530
Nvar = 2.6424
Nuv = -1.4352
Nvav = 2.1667
N0 = 0.0000
Ndr = -0.5337
Ndradr = 0.6899
rud = np.deg2rad(+25)
print(rud)
p = (rud, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
     Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
     Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr)  # Parameters of the system

y0 = [1.640094, 0.009943, -0.142654076, 0, 0 ,0]  # Initial state of the system

t_max = 60
t_span = (0.0,t_max)
t = np.arange(0.0, t_max , 0.01)

result_solve_ivp = solve_ivp(UUV_3DOF, t_span, y0, args=p, method ='RK23', t_eval =t)
plt.scatter(result_solve_ivp.y[3, :], result_solve_ivp.y[4, :])
plt.show()
