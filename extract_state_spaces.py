import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os


def export_evaluation_metric(pinn_model_path):
    def return_mape(data):
        from sklearn.metrics import mean_absolute_percentage_error
        u_pred = data['u_pred'].values
        v_pred = data['v_pred'].values
        r_pred = data['r_pred'].values
        u_true = data['u_true'].values
        v_true = data['v_true'].values
        r_true = data['r_true'].values
        mape_u = round(mean_absolute_percentage_error(u_pred, u_true) * 100, 4)
        mape_v = round(mean_absolute_percentage_error(v_pred, v_true) * 100, 4)
        mape_r = round(mean_absolute_percentage_error(r_pred, r_true) * 100, 4)
        return [mape_u, mape_v, mape_r]

    def return_rmse(data):
        from sklearn.metrics import mean_squared_error
        u_pred = data['u_pred'].values
        v_pred = data['v_pred'].values
        r_pred = data['r_pred'].values
        u_true = data['u_true'].values
        v_true = data['v_true'].values
        r_true = data['r_true'].values
        rmse_u = round(mean_squared_error(u_pred, u_true) ** 0.5, 4)
        rmse_v = round(mean_squared_error(v_pred, v_true) ** 0.5, 4)
        rmse_r = round(mean_squared_error(r_pred, r_true) ** 0.5, 4)
        return [rmse_u, rmse_v, rmse_r]

    def return_R2(data):
        from sklearn.metrics import r2_score
        u_pred = data['u_pred'].values
        v_pred = data['v_pred'].values
        r_pred = data['r_pred'].values
        u_true = data['u_true'].values
        v_true = data['v_true'].values
        r_true = data['r_true'].values
        R2_u = round(r2_score(u_true, u_pred), 4)
        R2_v = round(r2_score(v_true, v_pred), 4)
        R2_r = round(r2_score(r_true, r_pred), 4)
        return [R2_u, R2_v, R2_r]

    def return_r2(data):
        from scipy.stats import pearsonr
        u_pred = data['u_pred'].values
        v_pred = data['v_pred'].values
        r_pred = data['r_pred'].values
        u_true = data['u_true'].values
        v_true = data['v_true'].values
        r_true = data['r_true'].values
        r_u, _ = pearsonr(u_true, u_pred)
        r_v, _ = pearsonr(v_true, v_pred)
        r_r, _ = pearsonr(r_true, r_pred)
        return [round(r_u ** 2, 4), round(r_v ** 2, 4), round(r_r ** 2, 4)]

    evaluation_metric = pd.DataFrame()
    MAPE_u = []
    MAPE_v = []
    MAPE_r = []
    RMSE_u = []
    RMSE_v = []
    RMSE_r = []
    R2_u = []
    R2_v = []
    R2_r = []
    r2_u = []
    r2_v = []
    r2_r = []
    data_list = []

    rud_list = ['+5', '+10', '+15', '+20', '-5', '-10', '-15', '-20']
    for rud in rud_list:
        data_path = os.path.join(pinn_model_path, 'error_analysis', 'error_analysis_turning_rud%s.csv' % rud)
        data = pd.read_csv(data_path)
        mape_pinn_model = return_mape(data)
        rmse_pinn_model = return_rmse(data)
        R2_pinn_model = return_R2(data)
        r2_pinn_model = return_r2(data)

        # Save rmse_pinn_model_data_coef_apprch
        MAPE_u.append(mape_pinn_model[0])
        MAPE_v.append(mape_pinn_model[1])
        MAPE_r.append(mape_pinn_model[2])

        RMSE_u.append(rmse_pinn_model[0])
        RMSE_v.append(rmse_pinn_model[1])
        RMSE_r.append(rmse_pinn_model[2])

        R2_u.append(R2_pinn_model[0])
        R2_v.append(R2_pinn_model[1])
        R2_r.append(R2_pinn_model[2])

        r2_u.append(r2_pinn_model[0])
        r2_v.append(r2_pinn_model[1])
        r2_r.append(r2_pinn_model[2])
        data_list.append('turning_%s' % rud)

    rud_list = ["10", "10", "20"]
    yaw_list = ["10", "20", "20"]
    for rud, yaw in zip(rud_list, yaw_list):
        data_path = os.path.join(pinn_model_path, 'error_analysis',
                                 'error_analysis_zigzag_rud%s_yaw%s.csv' % (rud, yaw))
        data = pd.read_csv(data_path)
        rmse_pinn_model = return_rmse(data)
        R2_pinn_model = return_R2(data)
        r2_pinn_model = return_r2(data)
        mape_pinn_model = return_mape(data)

        # Save rmse_pinn_model_data_coef_apprch
        MAPE_u.append(mape_pinn_model[0])
        MAPE_v.append(mape_pinn_model[1])
        MAPE_r.append(mape_pinn_model[2])
        RMSE_u.append(rmse_pinn_model[0])
        RMSE_v.append(rmse_pinn_model[1])
        RMSE_r.append(rmse_pinn_model[2])
        R2_u.append(R2_pinn_model[0])
        R2_v.append(R2_pinn_model[1])
        R2_r.append(R2_pinn_model[2])
        r2_u.append(r2_pinn_model[0])
        r2_v.append(r2_pinn_model[1])
        r2_r.append(r2_pinn_model[2])
        data_list.append('zigzag_%s_%s' % (rud, yaw))

    evaluation_metric['MAPE_u'] = MAPE_u
    evaluation_metric['MAPE_v'] = MAPE_v
    evaluation_metric['MAPE_r'] = MAPE_r
    evaluation_metric['RMSE_u'] = RMSE_u
    evaluation_metric['RMSE_v'] = RMSE_v
    evaluation_metric['RMSE_r'] = RMSE_r
    evaluation_metric['R2_u'] = R2_u
    evaluation_metric['R2_v'] = R2_v
    evaluation_metric['R2_r'] = R2_r
    evaluation_metric['r2_u'] = r2_u
    evaluation_metric['r2_v'] = r2_v
    evaluation_metric['r2_r'] = r2_r
    evaluation_metric['data'] = data_list
    evaluation_metric_path = os.path.join(pinn_model_path, 'error_analysis', 'evaluation_metric_separate_data.csv')
    evaluation_metric.to_csv(evaluation_metric_path, index=False)

def force_full_predict(vel, vel_dot, rud):
    """ This class consider the 3DOF model
        Inputs X: rud, t
        Outputs Y: u, v, r, X_add, Y_add, N_add
    """

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
    # Params
    rho = 1000
    A = 0.15597
    Dp = 0.2345
    L = 2.414
    Uc = 2.572

    u, v, r = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]
    du_t, dv_t, dr_t = vel_dot[:, 0:1], vel_dot[:, 1:2], vel_dot[:, 2:3]
    # Thrust coefficients
    U = np.sqrt(u * u + v * v)
    _np = 1.1857 * Uc + 0.0931
    J = U * (1 - 0.4177) / (_np * Dp)
    XT = 3.3984 - 8.2263 * J + 5.4134 * J * J

    # Residuals
    fX_full = rho * A * L ** 2 / 2 * (Xrr * r ** 2) \
              + rho * A * L / 2 * (Xud * du_t + Xvr * v * r) \
              + rho * A / 2 * (Xuu * u ** 2 + Xvv * v ** 2 + u ** 2 * Xdrdr * rud ** 2) \
              + rho * _np * _np * Dp * Dp * Dp * Dp * XT * 0.694
    fY_full = rho * A * L ** 2 / 2 * (Yrd * dr_t + Yrar * r * np.abs(r)) \
              + rho * A * L / 2 * (Yvd * dv_t + Yur * u * r + Yvar * v * np.abs(r)) \
              + rho * A / 2 * (Yuv * u * v + Yvav * v * np.abs(v) + u ** 2 * (
            Y0 + Ydr * rud + Ydradr * rud * np.abs(rud)))
    fN_full = rho * A * L ** 3 / 2 * (Nrd * dr_t + Nrar * r * np.abs(r)) \
              + rho * A * L * L / 2 * (Nvd * dv_t + Nur * u * r + Nvar * v * np.abs(r)) \
              + rho * A * L / 2 * (Nuv * u * v + Nvav * v * np.abs(v) + u ** 2 * (
            N0 + Ndr * rud + Ndradr * rud * np.abs(rud)))
    return np.concatenate([fX_full, fY_full, fN_full], axis=1)


def lhs_predict(vel, vel_dot):
    """ This class consider the 3DOF model
            Inputs X: rud, t
            Outputs Y: u, v, r, X_add, Y_add, N_add
    """
    u, v, r = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]
    du_t, dv_t, dr_t = vel_dot[:, 0:1], vel_dot[:, 1:2], vel_dot[:, 2:3]

    # Params
    m = 302.59
    x_g = 0.19252
    y_g = 0.00023
    I_zz = 119.83306

    # lhs
    lhs_u = m * (du_t - v * r - x_g * r ** 2 - y_g * dr_t)
    lhs_v = m * (dv_t + u * r - y_g * r ** 2 + x_g * dr_t)
    lhs_r = I_zz * dr_t + m * (x_g * (dv_t + u * r) - y_g * (du_t - v * r))
    return np.concatenate([lhs_u, lhs_v, lhs_r], axis=1)


def UUV_3DOF(t, state, rud, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
             Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
             Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr, saved_vel_dot):
    u, v, r, x, y, psi = state
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
            + rho * A / 2 * (Yuv * u * v + Yvav * v * np.abs(v) + u ** 2 * (
            Y0 + Ydr * rud + Ydradr * rud * np.abs(rud))) - m * (u * r - y_g * r ** 2)
    rhs_N = rho * A * L ** 3 / 2 * (Nrar * r * np.abs(r)) \
            + rho * A * L * L / 2 * (Nur * u * r + Nvar * v * np.abs(r)) \
            + rho * A * L / 2 * (Nuv * u * v + Nvav * v * np.abs(v) + u ** 2 * (
            N0 + Ndr * rud + Ndradr * rud * np.abs(rud))) - m * (x_g * u * r + y_g * v * r)
    F = np.array([[rhs_X],
                  [rhs_Y],
                  [rhs_N]])

    M = np.array([[m - rho * A * L / 2 * Xud, 0, -m * y_g],
                  [0, m - rho * A * L / 2 * Yvd, m * x_g - rho / 2 * A * L ** 2 * Yrd],
                  [-m * y_g, m * x_g - rho * A * L ** 2 / 2 * Nvd, I_zz - rho / 2 * A * L ** 3 * Nrd]])
    M_inv = np.linalg.inv(M)
    vel_dot = np.dot(M_inv, F).T.tolist()[0]

    # Calculate for the trajectories
    dot_x = np.cos(psi) * u - np.sin(psi) * v
    dot_y = np.sin(psi) * u + np.cos(psi) * v
    dot_psi = r
    saved_vel_dot.append(vel_dot)
    return vel_dot + [dot_x, dot_y, dot_psi]


def extract_states_forces_closed_loop(target_data_path_list=None, saved_path_list=None):
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

    for target_data_path, saved_path in zip(target_data_path_list,
                                            saved_path_list):
        df = pd.read_csv(target_data_path)
        data = df[['u0', 'v0', 'r0', 'xn0', 'xe0', 'psi0', 'targ_rud', 'delta_t', 'u', 'v', 'r', 'xn', 'xe', 'psi']].values
        save_column_name = ['targ_rud', 'delta_t', 'u0', 'v0', 'r0',
                            'u_true', 'v_true', 'r_true', 'xn_true', 'xe_true', 'psi_true',
                            'u_pred', 'v_pred', 'r_pred', 'xn_pred', 'xe_pred', 'psi_pred',
                            'Xfull0_pred', 'Yfull0_pred', 'Nfull0_pred',
                            'lhs_u0_pred', 'lhs_v0_pred', 'lhs_r0_pred']
        for idx, data_i in enumerate(data):
            saved_vel_dot = []
            state0 = data_i[:6].tolist()
            rud = data_i[6:7].tolist()[0]
            delta_t = data_i[7:8].tolist()[0]
            t_span = (0.0, delta_t)
            p = (rud, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
                 Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
                 Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr, saved_vel_dot)  # Parameters of the system
            result_solve_ivp = solve_ivp(UUV_3DOF, t_span, state0, args=p, method='RK45', t_eval=[delta_t])
            if idx == 0:
                state_pred = result_solve_ivp.y
                vel0 = np.reshape(data_i[:3], (1, 3))
                saved_vel_dot = np.array(saved_vel_dot[0]).reshape(1, 3)
                lhs_pred = lhs_predict(vel0, saved_vel_dot)
                forcefull_pred = force_full_predict(vel0, saved_vel_dot, rud)
                simulation_data = np.hstack((np.reshape(data_i, (1, 14)),
                                             np.reshape(state_pred, (1, 6)),
                                             saved_vel_dot,
                                             forcefull_pred,
                                             lhs_pred))
                continue
            state_pred = result_solve_ivp.y
            vel0 = np.reshape(data_i[:3], (1, 3))
            saved_vel_dot = np.array(saved_vel_dot[0]).reshape(1, 3)
            lhs_pred = lhs_predict(vel0, saved_vel_dot)
            forcefull_pred = force_full_predict(vel0, saved_vel_dot, rud)
            simulation_data_ = np.hstack((np.reshape(data_i, (1, 14)),
                                          np.reshape(state_pred, (1, 6)),
                                          saved_vel_dot,
                                          forcefull_pred,
                                          lhs_pred))
            simulation_data = np.vstack((simulation_data, simulation_data_))
        simulation_df = pd.DataFrame(simulation_data,
                                     columns=['u0', 'v0', 'r0', 'xn0', 'xe0', 'psi0', 'targ_rud', 'delta_t',
                                              'u_true', 'v_true', 'r_true', 'xn_true', 'xe_true', 'psi_true',
                                              'u_pred', 'v_pred', 'r_pred', 'xn_pred', 'xe_pred', 'psi_pred',
                                              'du_pred', 'dv_pred', 'dr_pred',
                                              'Xfull0_pred', 'Yfull0_pred', 'Nfull0_pred',
                                              'lhs_u0_pred', 'lhs_v0_pred', 'lhs_r0_pred'])
        simulation_df.to_csv(saved_path,
                             index=False,
                             columns=save_column_name)


if False:
    target_data_folder = r'preprocess_wo_acc_w_psi\ref_folder'
    rud_list = ['+5', '+10', '+15', '+20', '-5', '-10', '-15', '-20']
    target_turning_data_path_list = [r'data\%s\3dof_data_turning_rud%s_ref.csv' % (target_data_folder, rud) for rud in rud_list]
    turning_saved_path_list = [r'error_analysis\error_analysis_turning_rud%s.csv' % (rud) for rud in rud_list]
    rud_list = ["10", "10", "20"]
    yaw_list = ["10", "20", "20"]
    target_zigzag_data_path_list = [r'data\%s\3dof_data_zigzag_rud%s_yaw%s_ref.csv' % (target_data_folder, rud, yaw) for rud, yaw in zip(rud_list, yaw_list)]
    zigzag_saved_path_list = [r'error_analysis\error_analysis_zigzag_rud%s_yaw%s.csv' % (rud, yaw) for rud, yaw in zip(rud_list, yaw_list)]
    target_data_path_list = target_turning_data_path_list + target_zigzag_data_path_list
    saved_path_list = turning_saved_path_list + zigzag_saved_path_list
    extract_states_forces_closed_loop(target_data_path_list=target_data_path_list, saved_path_list=saved_path_list)

if True:
    for sect in [1,2,3,4]:
        target_data_folder = r'preprocess_wo_acc_w_psi\ref_folder'
        rud_list = ['+5', '+10', '+15', '+20', '-5', '-10', '-15', '-20']
        target_turning_data_path_list = [r'data\%s\3dof_data_turning_rud%s_ref_sect_%s.csv' % (target_data_folder, rud, sect) for rud in rud_list]
        turning_saved_path_list = [r'error_analysis\error_analysis_turning_rud%s_sect_%s.csv' % (rud, sect) for rud in rud_list]
        rud_list = ["10", "10", "20"]
        yaw_list = ["10", "20", "20"]
        target_zigzag_data_path_list = [r'data\%s\3dof_data_zigzag_rud%s_yaw%s_ref_sect_%s.csv' % (target_data_folder, rud, yaw, sect) for rud, yaw in zip(rud_list, yaw_list)]
        zigzag_saved_path_list = [r'error_analysis\error_analysis_zigzag_rud%s_yaw%s_sect_%s.csv' % (rud, yaw, sect) for rud, yaw in zip(rud_list, yaw_list)]
        target_data_path_list = target_turning_data_path_list + target_zigzag_data_path_list
        saved_path_list = turning_saved_path_list + zigzag_saved_path_list
        extract_states_forces_closed_loop(target_data_path_list=target_data_path_list, saved_path_list=saved_path_list)

if False:
    model_path = '.'
    export_evaluation_metric(model_path)
