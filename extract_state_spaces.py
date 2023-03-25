import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os


def convert_psi(psi):
    cycle_min = - np.pi
    cycle_max = + np.pi
    cycle_width = cycle_max - cycle_min
    psi = psi % cycle_width
    y = psi.copy()
    y_min_idx = y < cycle_min
    y_max_idx = y > cycle_max
    y[y_min_idx] = y[y_min_idx] + cycle_width
    y[y_max_idx] = y[y_max_idx] - cycle_width
    psi = y
    return psi


def export_evaluation_metric_3dof(target_data_folder, saved_path):
    def return_metric(data, terms=None, metric_name='rmse'):
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
        from scipy.stats import pearsonr
        from astropy.stats import circcorrcoef
        if terms is None:
            terms = ['du', 'dv', 'dr', 'u', 'v', 'r', 'x', 'y','psi']
        metrics = []
        for term in terms:
            if isinstance(term, list):
                true_name = [i + '_true' for i in term]
                pred_name = [i + '_pred' for i in term]
            else:
                true_name = term + '_true'
                pred_name = term + '_pred'
                if term == 'X':
                    pred_name = 'lhs_u0_pred'
                if term == 'Y':
                    pred_name = 'lhs_v0_pred'
                if term == 'Z':
                    pred_name = 'lhs_w0_pred'
                if term == 'K':
                    pred_name = 'lhs_p0_pred'
                if term == 'M':
                    pred_name = 'lhs_q0_pred'
                if term == 'N':
                    pred_name = 'lhs_r0_pred'
            true_data = data[true_name].values
            pred_data = data[pred_name].values
            if metric_name == 'rmse':
                metric = round(mean_squared_error(true_data, pred_data) ** 0.5, 4)
            elif metric_name == 'R2':
                metric = round(r2_score(true_data, pred_data), 4)
            elif metric_name == 'mape':
                metric = round(mean_absolute_percentage_error(true_data, pred_data) * 100, 4)
            elif metric_name == 'r2':
                metric, _ = pearsonr(true_data, pred_data)
                metric = round(metric ** 2, 4)
            elif metric_name == 'circ_coef':
                metric = round(circcorrcoef(true_data, pred_data), 4)
            elif metric_name == 'relative_norm':
                if term == ['x', 'y', 'z'] or term == ['x', 'y']:
                    origin = data[true_name].values[0]
                    pos_pred_trans = data[pred_name].values - origin
                    pos_true_trans = data[true_name].values - origin
                    metric = round(np.linalg.norm(pos_pred_trans - pos_true_trans) / np.linalg.norm(pos_true_trans) * 100,4)
                else:
                    metric = round(np.linalg.norm(true_data - pred_data) / np.linalg.norm(true_data) * 100,4)
            metrics.append(metric)
        return metrics

    metric_list = []
    for file_name in os.listdir(target_data_folder):
        if file_name.split("_")[0] == 'Turn':
            if file_name.split("_")[3] == 'Drm40':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'ZZH':
            if file_name.split("_")[3] == '20to20':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'ZZV':
            if file_name.split("_")[3] == '20to20':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'VM':
            if file_name.split("_")[3] == '20to10':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'HM':
            if file_name.split("_")[3] == '30to20':
                data_set = ['testing']
            else:
                data_set = ['training']
        data_path = os.path.join(target_data_folder, file_name)
        data = pd.read_csv(data_path)
        mape = return_metric(data, terms = ['du', 'dv', 'dr',
                                            'u', 'v', 'r',
                                            'x', 'y','psi','X','Y','N'], metric_name='mape')
        rmse = return_metric(data, terms = ['du', 'dv', 'dr',
                                            'u', 'v', 'r',
                                            'x', 'y','psi','X','Y','N'], metric_name='rmse')
        r2 = return_metric(data, terms = ['du', 'dv', 'dr',
                                          'u', 'v', 'r',
                                          'x', 'y','psi','X','Y','N'], metric_name='r2')
        R2 = return_metric(data, terms = ['du', 'dv', 'dr',
                                          'u', 'v', 'r',
                                          'x', 'y','psi','X','Y','N'], metric_name='R2')
        circular_coef = return_metric(data, terms = ['psi'], metric_name='circ_coef')
        rel_norm_xyz = return_metric(data, terms = [['x', 'y']], metric_name='relative_norm')
        rel_norm_uvw = return_metric(data, terms = [['u', 'v']], metric_name='relative_norm')
        rel_norm_pqr = return_metric(data, terms = [['r']], metric_name='relative_norm')
        rel_norm_dudvdw = return_metric(data, terms = [['du', 'dv']], metric_name='relative_norm')
        rel_norm_dpdqdr = return_metric(data, terms = [['dr']], metric_name='relative_norm')
        metric_list.append(mape + rmse + r2 + R2 + circular_coef + rel_norm_xyz + rel_norm_uvw + rel_norm_pqr + rel_norm_dudvdw + rel_norm_dpdqdr + data_set + [file_name])
    mape_name = ['mape_' + term for term in ['du', 'dv', 'dr',
                                             'u', 'v', 'r',
                                             'x', 'y','psi','X','Y','N']]
    rmse_name = ['rmse_' + term for term in ['du', 'dv', 'dr',
                                             'u', 'v', 'r',
                                             'x', 'y','psi','X','Y','N']]
    r2_name = ['r2_' + term for term in ['du', 'dv', 'dr',
                                         'u', 'v', 'r',
                                         'x', 'y','psi','X','Y','N']]
    R2_name = ['R2_' + term for term in ['du', 'dv', 'dr',
                                         'u', 'v', 'r',
                                         'x', 'y','psi','X','Y','N']]
    cc_name = ['ccc_' + term for term in ['psi']]
    rel_xyz_name = ['rel_traj']
    rel_uvw_name = ['rel_uvw']
    rel_pqr_name = ['rel_pqr']
    rel_dudvdw_name = ['rel_dudvdw']
    rel_dpdqdr_name = ['rel_dpdqdr']
    name = mape_name + rmse_name + r2_name + R2_name + cc_name +  rel_xyz_name + rel_uvw_name + rel_pqr_name + rel_dudvdw_name + rel_dpdqdr_name + ['data'] + ['dataset']
    metric_array = np.array(metric_list)
    metric_df = pd.DataFrame(metric_array, columns = name)
    evaluation_metric_path = os.path.join(saved_path)
    metric_df.to_csv(evaluation_metric_path, index=False)

def export_evaluation_metric_6dof(target_data_folder, saved_path):
    def return_metric(data, terms=None, metric_name='rmse'):
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
        from scipy.stats import pearsonr
        from astropy.stats import circcorrcoef
        if terms is None:
            terms = ['du', 'dv', 'dw', 'dp', 'dq', 'dr', 'u', 'v', 'w', 'p', 'q', 'r', 'x', 'y', 'z', 'phi', 'theta','psi']
        metrics = []
        for term in terms:
            if isinstance(term, list):
                true_name = [i + '_true' for i in term]
                pred_name = [i + '_pred' for i in term]
            else:
                true_name = term + '_true'
                pred_name = term + '_pred'
                if term == 'X':
                    pred_name = 'lhs_u0_pred'
                if term == 'Y':
                    pred_name = 'lhs_v0_pred'
                if term == 'Z':
                    pred_name = 'lhs_w0_pred'
                if term == 'K':
                    pred_name = 'lhs_p0_pred'
                if term == 'M':
                    pred_name = 'lhs_q0_pred'
                if term == 'N':
                    pred_name = 'lhs_r0_pred'
            true_data = data[true_name].values
            pred_data = data[pred_name].values
            if metric_name == 'rmse':
                metric = round(mean_squared_error(true_data, pred_data) ** 0.5, 4)
            elif metric_name == 'R2':
                metric = round(r2_score(true_data, pred_data), 4)
            elif metric_name == 'mape':
                metric = round(mean_absolute_percentage_error(true_data, pred_data) * 100, 4)
            elif metric_name == 'r2':
                metric, _ = pearsonr(true_data, pred_data)
                metric = round(metric ** 2, 4)
            elif metric_name == 'circ_coef':
                metric = round(circcorrcoef(true_data, pred_data), 4)
            elif metric_name == 'relative_norm':
                if term == ['x', 'y', 'z']:
                    origin = data[true_name].values[0]
                    pos_pred_trans = data[pred_name].values - origin
                    pos_true_trans = data[true_name].values - origin
                    metric = round(np.linalg.norm(pos_pred_trans - pos_true_trans) / np.linalg.norm(pos_true_trans) * 100,4)
                else:
                    metric = round(np.linalg.norm(true_data - pred_data) / np.linalg.norm(true_data) * 100,4)
            metrics.append(metric)
        return metrics

    metric_list = []
    for file_name in os.listdir(target_data_folder):
        if file_name.split("_")[0] == 'Turn':
            if file_name.split("_")[3] == 'Drm40':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'ZZH':
            if file_name.split("_")[3] == '20to20':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'ZZV':
            if file_name.split("_")[3] == '20to20':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'VM':
            if file_name.split("_")[3] == '20to10':
                data_set = ['testing']
            else:
                data_set = ['training']
        elif file_name.split("_")[0] == 'HM':
            if file_name.split("_")[3] == '30to20':
                data_set = ['testing']
            else:
                data_set = ['training']
        data_path = os.path.join(target_data_folder, file_name)
        data = pd.read_csv(data_path)
        mape = return_metric(data, terms = ['du', 'dv', 'dw', 'dp', 'dq', 'dr',
                                            'u', 'v', 'w', 'p', 'q', 'r',
                                            'x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N'], metric_name='mape')
        rmse = return_metric(data, terms = ['du', 'dv', 'dw', 'dp', 'dq', 'dr',
                                            'u', 'v', 'w', 'p', 'q', 'r',
                                            'x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N'], metric_name='rmse')
        r2 = return_metric(data, terms = ['du', 'dv', 'dw', 'dp', 'dq', 'dr',
                                          'u', 'v', 'w', 'p', 'q', 'r',
                                          'x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N'], metric_name='r2')
        R2 = return_metric(data, terms = ['du', 'dv', 'dw', 'dp', 'dq', 'dr',
                                          'u', 'v', 'w', 'p', 'q', 'r',
                                          'x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N'], metric_name='R2')
        circular_coef = return_metric(data, terms = ['phi', 'theta','psi'], metric_name='circ_coef')
        rel_norm_xyz = return_metric(data, terms = [['x', 'y','z']], metric_name='relative_norm')
        rel_norm_uvw = return_metric(data, terms = [['u', 'v','w']], metric_name='relative_norm')
        rel_norm_pqr = return_metric(data, terms = [['p', 'q','r']], metric_name='relative_norm')
        rel_norm_dudvdw = return_metric(data, terms = [['du', 'dv', 'dw']], metric_name='relative_norm')
        rel_norm_dpdqdr = return_metric(data, terms = [['dp', 'dq', 'dr']], metric_name='relative_norm')
        metric_list.append(mape + rmse + r2 + R2 + circular_coef + rel_norm_xyz + rel_norm_uvw + rel_norm_pqr + rel_norm_dudvdw + rel_norm_dpdqdr +  data_set + [file_name])
    mape_name = ['mape_' + term for term in ['du', 'dv', 'dw', 'dp', 'dq', 'dr','u', 'v', 'w', 'p', 'q', 'r','x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N']]
    rmse_name = ['rmse_' + term for term in ['du', 'dv', 'dw', 'dp', 'dq', 'dr','u', 'v', 'w', 'p', 'q', 'r','x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N']]
    r2_name = ['r2_' + term for term in ['du', 'dv', 'dw', 'dp', 'dq', 'dr','u', 'v', 'w', 'p', 'q', 'r','x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N']]
    R2_name = ['R2_' + term for term in ['du', 'dv', 'dw', 'dp', 'dq', 'dr','u', 'v', 'w', 'p', 'q', 'r','x', 'y', 'z', 'phi', 'theta','psi','X','Y','Z','K','M','N']]
    cc_name = ['ccc_' + term for term in ['phi', 'theta','psi']]
    rel_xyz_name = ['rel_traj']
    rel_uvw_name = ['rel_uvw']
    rel_pqr_name = ['rel_pqr']
    rel_dudvdw_name = ['rel_dudvdw']
    rel_dpdqdr_name = ['rel_dpdqdr']
    name = mape_name + rmse_name + r2_name + R2_name + cc_name +  rel_xyz_name + rel_uvw_name + rel_pqr_name  + rel_dudvdw_name + rel_dpdqdr_name  +['data'] + ['dataset']
    metric_array = np.array(metric_list)
    metric_df = pd.DataFrame(metric_array, columns = name)
    evaluation_metric_path = os.path.join(saved_path)
    metric_df.to_csv(evaluation_metric_path, index=False)


def force_full_predict(vel, vel_dot, rud, RPM, external_trainable_variables):
    """ This class consider the 3DOF model
        Inputs X: rud, t
        Outputs Y: u, v, r, X_add, Y_add, N_add
    """
    u, v, r = vel[0:1], vel[1:2], vel[2:3]
    du_t, dv_t, dr_t = vel_dot[0:1], vel_dot[1:2], vel_dot[2:3]

    [Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
     Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
     Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr] = external_trainable_variables
    # Params
    rho = 1000
    L = 3.826

    # Thrust coefficients
    U = np.sqrt(u * u + v * v)
    rps = RPM / 60
    Dp = 0.2725
    if rps <= 1e-5:
        J = 0
    else:
        J = U * 0.6688 / (rps * Dp)
    # J = np.where(rps <= 0.0, np.zeros_like(rps), U * 0.6688 / (rps * Dp))
    KT = 0.5096 - 0.4169 * J - 0.0495 * J ** 2 - 0.0586 * J ** 3

    # Residuals
    fX_full = rho * L ** 4 / 2 * (Xrr * r ** 2) \
              + rho * L ** 3 / 2 * (Xud * du_t + Xvr * v * r) \
              + rho * L ** 2 / 2 * (Xuu * u ** 2 + Xvv * v ** 2 + u ** 2 * Xdrdr * rud ** 2) \
              + (rho * rps ** 2 * Dp ** 4 * KT) * 0.8334
    fY_full = rho * L ** 4 / 2 * (Yrd * dr_t + Yrar * r * np.abs(r)) \
              + rho * L ** 3 / 2 * (Yvd * dv_t + Yur * u * r + Yvar * v * np.abs(r)) \
              + rho * L ** 2 / 2 * (
                      Yuv * u * v + Yvav * v * np.abs(v) + u ** 2 * (Y0 + Ydr * rud + Ydradr * rud * np.abs(rud)))
    fN_full = rho * L ** 5 / 2 * (Nrd * dr_t + Nrar * r * np.abs(r)) \
              + rho * L ** 4 / 2 * (Nvd * dv_t + Nur * u * r + Nvar * v * np.abs(r)) \
              + rho * L ** 3 / 2 * (
                      Nuv * u * v + Nvav * v * np.abs(v) + u ** 2 * (N0 + Ndr * rud + Ndradr * rud * np.abs(rud)))
    return [fX_full[0], fY_full[0], fN_full[0]]


def lhs_predict(vel, vel_dot):
    """ This class consider the 3DOF model
            Inputs X: rud, t
            Outputs Y: u, v, r, X_add, Y_add, N_add
    """
    u, v, r = vel[0:1], vel[1:2], vel[2:3]
    du_t, dv_t, dr_t = vel_dot[0:1], vel_dot[1:2], vel_dot[2:3]

    # Params
    m = 703.957
    x_g = 0.156285
    y_g = 0.00063
    I_zz = 673.57

    # lhs
    lhs_u = m * (du_t - v * r - x_g * r ** 2 - y_g * dr_t)
    lhs_v = m * (dv_t + u * r - y_g * r ** 2 + x_g * dr_t)
    lhs_r = I_zz * dr_t + m * (x_g * (dv_t + u * r) - y_g * (du_t - v * r))
    return [lhs_u[0], lhs_v[0], lhs_r[0]]


def UUV_3DOF(t, state, rud, RPM, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
             Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
             Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr, saved_vel_dot):
    u, v, r, x, y, psi = state
    # Params
    m = 703.957
    x_g = 0.156285
    y_g = 0.00063
    I_zz = 673.57
    rho = 1000
    L = 3.826
    # Thrust coefficients
    U = np.sqrt(u * u + v * v)
    rps = RPM / 60
    Dp = 0.2725
    if rps <= 1e-5:
        J = 0
    else:
        J = U * 0.6688 / (rps * Dp)
    # J = np.where(rps == 0.0, np.zeros_like(rps), U * 0.6688 / (rps * Dp))
    KT = 0.5096 - 0.4169 * J - 0.0495 * J ** 2 - 0.0586 * J ** 3

    # Calculate for the velocities
    rhs_X = rho * L ** 4 / 2 * (Xrr * r ** 2) \
            + rho * L ** 3 / 2 * Xvr * v * r \
            + rho * L ** 2 / 2 * (Xuu * u ** 2 + Xvv * v ** 2 + u ** 2 * Xdrdr * rud ** 2) \
            + (rho * rps ** 2 * Dp ** 4 * KT) * 0.8334 + m * (v * r + x_g * r ** 2)
    rhs_Y = rho * L ** 4 / 2 * (Yrar * r * np.abs(r)) \
            + rho * L ** 3 / 2 * (Yur * u * r + Yvar * v * np.abs(r)) \
            + rho * L ** 2 / 2 * (Yuv * u * v + Yvav * v * np.abs(v) + u ** 2 * (
            Y0 + Ydr * rud + Ydradr * rud * np.abs(rud))) - m * (u * r - y_g * r ** 2)
    rhs_N = rho * L ** 5 / 2 * (Nrar * r * np.abs(r)) \
            + rho * L ** 4 / 2 * (Nur * u * r + Nvar * v * np.abs(r)) \
            + rho * L ** 3 / 2 * (Nuv * u * v + Nvav * v * np.abs(v) + u ** 2 * (
            N0 + Ndr * rud + Ndradr * rud * np.abs(rud))) - m * (x_g * u * r + y_g * v * r)
    F = np.array([[rhs_X],
                  [rhs_Y],
                  [rhs_N]])

    M = np.array([[m - rho * L ** 3 / 2 * Xud, 0, -m * y_g],
                  [0, m - rho * L ** 3 / 2 * Yvd, m * x_g - rho / 2 * L ** 4 * Yrd],
                  [-m * y_g, m * x_g - rho * L ** 4 / 2 * Nvd, I_zz - rho / 2 * L ** 5 * Nrd]])
    M_inv = np.linalg.inv(M)
    vel_dot = np.dot(M_inv, F).T.tolist()[0]

    # Calculate for the trajectories
    dot_x = np.cos(psi) * u - np.sin(psi) * v
    dot_y = np.sin(psi) * u + np.cos(psi) * v
    dot_psi = r
    saved_vel_dot.append(vel_dot)
    return vel_dot + [dot_x, dot_y, dot_psi]


def UUV_6DOF(t, state, rud, ele, RPM, Xuu, Xvv, Xww, Xud, Xvr, Xwq, Xqq, Xrr, Xdrdr, Xdede,
             Yuv, Yvav, Yvw, Y0, Yvd, Yvq, Ywp, Ywr, Yur, Yup, Yvar, Yrd, Ypq, Yqr, Yrar, Ydr, Ydradr,
             Zuw, Zwaw, Zvw, Z0, Zwd, Zwr, Zvp, Zvq, Zuq, Zup, Zwaq, Zqd, Zrp, Zqr, Zqaq, Zde, Zdeade,
             K0, Kvw, Kup, Kvq, Kwr, Kpd, Kqr, Kpq, Kpap, Kdr, Kde,
             Muw, Mwaw, Mvw, M0, Mwd, Mwr, Mvp, Muq, Mup, Mwaq, Mqd, Mrp, Mqr, Mqaq, Mde, Mdeade,
             Nuv, Nvav, Nvw, N0, Nvd, Nvq, Nwp, Nur, Nup, Nvar, Nrd, Npq, Nqr, Nrar, Ndr, Ndradr, saved_vel_dot):
    u, v, w, p, q, r, x, y, z, phi, theta, psi = state
    # Params
    m = 703.957
    B = 705.216 * 9.80665
    W = m * 9.80665
    x_g = 0.156285
    y_g = 0.00063
    z_g = -0.001072
    x_b = 0.150576
    y_b = 0.000097298
    z_b = -0.025174
    I_xx = 29.66
    I_yy = 683.42
    I_zz = 673.57
    I_xy = Iyx = 0.044
    I_yz = I_zy = -0.01
    I_zx = I_xz = -0.353
    rho = 1000
    L = 3.826
    # Thrust coefficients
    U = np.sqrt(u * u + v * v + w * w)
    rps = RPM / 60
    Dp = 0.2725
    if rps <= 1e-5:
        J = 0
    else:
        J = U * 0.6688 / (rps * Dp)
    # J = np.where(rps == 0.0, np.zeros_like(rps), U * 0.6688 / (rps * Dp))
    KT = 0.5096 - 0.4169 * J - 0.0495 * J ** 2 - 0.0586 * J ** 3

    # Calculate for the velocities
    rhs_X = - m * (- v * r + w * q - x_g * (q ** 2 + r ** 2) + y_g * (p * q) + z_g * (p * r)) \
            + rho * L ** 4 / 2 * (Xqq * q ** 2 + Xrr * r ** 2) \
            + rho * L ** 3 / 2 * (Xvr * v * r + Xwq * w * q) \
            + rho * L ** 2 / 2 * (Xuu * u ** 2 + Xvv * v ** 2 + Xww * w ** 2 + u ** 2 * (Xdrdr * rud ** 2 + Xdede * ele ** 2)) \
            - (W - B) * np.sin(theta) + (rho * rps ** 2 * Dp ** 4 * KT) * 0.8334

    rhs_Y = - m * (- w * p + u * r - y_g * (r ** 2 + p ** 2) + z_g * (q * r) + x_g * (q * p)) \
            + rho * L ** 4 / 2 * (Ypq * p * q + Yqr * q * r + Yrar * r * np.abs(r)) \
            + rho * L ** 3 / 2 * (Yvq * v * q + Ywp * w * p + Ywr * w * r + Yur * u * r + Yup * u * p + Yvar * v * np.abs(r)) \
            + rho * L ** 2 / 2 * (Yuv * u * v + Yvav * v * np.abs(v) + Yvw * v * w + u ** 2 * (Y0 + Ydr * rud + Ydradr * rud * np.abs(rud))) \
            + (W - B) * np.cos(theta) * np.sin(phi)

    rhs_Z = - m * (- u * q + v * p - z_g * (p ** 2 + q ** 2) + x_g * (r * p) + y_g * (r * q)) \
            + rho * L ** 4 / 2 * (Zqr * q * r + Zrp * r * p + Zqaq * q * np.abs(q)) \
            + rho * L ** 3 / 2 * (Zwr * w * r + Zvp * v * p + Zvq * v * q + Zuq * u * q + Zup * u * p + Zwaq * w * np.abs(q)) \
            + rho * L ** 2 / 2 * (Zuw * u * w + Zwaw * w * np.abs(w) + Zvw * v * w + u ** 2 * (Z0 + Zde * ele + Zdeade * ele * np.abs(ele))) \
            + (W - B) * np.cos(theta) * np.cos(phi)

    rhs_K = - (I_zz - I_yy) * q * r + (p * q) * I_xz - (r * r - q * q) * I_yz - (p * r) * I_xy - m * (y_g * (- u * q + v * p) - z_g * (- w * p + u * r)) \
            + rho * L ** 5 / 2 * (Kpq * p * q + Kqr * q * r + Kpap * p * np.abs(p)) \
            + rho * L ** 4 / 2 * (Kup * u * p + Kvq * v * q + Kwr * w * r) \
            + rho * L ** 3 / 2 * (Kvw * v * w + u ** 2 * (Kdr * rud + Kde * ele)) \
            + rho * rps ** 2 * Dp ** 5 * K0 \
            + (y_g * W - y_b * B) * np.cos(theta) * np.cos(phi) - (z_g * W - z_b * B) * np.cos(theta) * np.sin(phi)

    rhs_M = - (I_xx - I_zz) * r * p + (q * r) * I_xy - (p ** 2 - r ** 2) * I_xz - (q * p) * I_yz - m * (
                z_g * (- v * r + w * q) - x_g * (- u * q + v * p)) \
            + rho * L ** 5 / 2 * (Mrp * r * p + Mqr * q * r + Mqaq * q * np.abs(q)) \
            + rho * L ** 4 / 2 * (Mwr * w * r + Mvp * v * p + Muq * u * q + Mup * u * p + Mwaq * w * np.abs(q)) \
            + rho * L ** 3 / 2 * (Muw * u * w + Mwaw * w * np.abs(w) + Mvw * v * w + u ** 2 * (
                M0 + Mde * ele + Mdeade * ele * np.abs(ele))) \
            - (z_g * W - z_b * B) * np.sin(theta) - (x_g * W - x_b * B) * np.cos(theta) * np.cos(phi)

    rhs_N = - (I_yy - I_xx) * p * q + (r * p) * I_yz - (q ** 2 - p ** 2) * I_xy - (r * q) * I_xz - m * (
                x_g * (- w * p + u * r) - y_g * (- v * r + w * q)) \
            + rho * L ** 5 / 2 * (Npq * p * q + Nqr * q * r + Nrar * r * np.abs(r)) \
            + rho * L ** 4 / 2 * (Nvq * v * q + Nwp * w * p + Nur * u * r + Nup * u * p + Nvar * v * np.abs(r)) \
            + rho * L ** 3 / 2 * (Nuv * u * v + Nvav * v * np.abs(v) + Nvw * v * w + u ** 2 * (
                N0 + Ndr * rud + Ndradr * rud * np.abs(rud))) \
            + (x_g * W - x_b * B) * np.cos(theta) * np.sin(phi) + (y_g * W - y_b * B) * np.sin(theta)

    F = np.array([[rhs_X],
                  [rhs_Y],
                  [rhs_Z],
                  [rhs_K],
                  [rhs_M],
                  [rhs_N]])

    M = np.array([[m - rho * L ** 3 / 2 * Xud, 0, 0, 0, m * z_g, -m * y_g],
                  [0, m - rho * L ** 3 / 2 * Yvd, 0, m * z_g, 0, m * x_g - rho / 2 * L ** 4 * Yrd],
                  [0, 0, m - rho * L ** 3 / 2 * Zwd, m * y_g, -(m * x_g - rho / 2 * L ** 4 * Zqd), 0],
                  [0, -m * z_g, m * y_g, (I_xx - rho * L ** 5 / 2 * Kpd), -I_xy, -I_xz],
                  [m * z_g, 0, -(m * x_g + rho * L ** 4 / 2 * Mwd), -I_xy, I_yy - rho * L ** 5 / 2 * Mqd,- (I_yz + rho * L ** 5 / 2)],
                  [-m * y_g, m * x_g - rho * L ** 4 / 2 * Nvd, 0, -I_zx, -I_yz, I_zz - rho * L ** 5 / 2 * Nrd]])
    M_inv = np.linalg.inv(M)
    vel_dot = np.dot(M_inv, F).T.tolist()[0]

    # Calculate for the trajectories
    x_dot = np.cos(psi) * np.cos(theta) * u \
            + (- np.sin(psi) * np.cos(phi) + np.cos(psi) * np.sin(theta) * np.sin(phi)) * v \
            + (np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(phi) * np.sin(theta)) * w
    y_dot = np.sin(psi) * np.cos(theta) * u \
            + (np.cos(psi) * np.cos(phi) + np.sin(phi) * np.sin(theta) * np.sin(psi)) * v \
            + (-np.cos(psi) * np.sin(phi) + np.sin(theta) * np.sin(psi) * np.cos(phi)) * w
    z_dot = -np.sin(theta) * u \
            + (np.cos(theta) * np.sin(phi)) * v \
            + (np.cos(theta) * np.cos(phi)) * w
    phi_dot = p \
              + np.sin(phi) * np.tan(theta) * q \
              + np.cos(phi) * np.tan(theta) * r
    theta_dot = np.cos(phi) * q \
                - np.sin(phi) * r
    psi_dot = np.sin(phi) / np.cos(theta) * q \
              + np.cos(phi) / np.cos(theta) * r
    saved_vel_dot.append(vel_dot)
    return vel_dot + [x_dot, y_dot, z_dot, phi_dot,theta_dot,psi_dot]


def extract_3dof_states_forces_closedloop(target_data_path=None, saved_path=None, external_trainable_variables=None):
    [Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
     Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
     Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr] = external_trainable_variables

    target_data = pd.read_csv(target_data_path)
    dt = target_data[['delta_t']].values
    dr = target_data[['dr']].values
    RPM = target_data[['RPM']].values
    vel0 = target_data[['u0', 'v0', 'r0']].values
    pos0 = target_data[['x0', 'y0', 'psi0']].values
    vel_pred = []
    dvel_pred = []
    pos_pred = []
    lhs_pred = []
    forcefull_pred = []
    # online_time = []
    for dr_i, RPM_i, dt_i, vel0_i, pos0_i in zip(dr, RPM, dt, vel0, pos0):
        dvel_pred_i = []
        dr_i = dr_i[0]
        RPM_i = RPM_i[0]
        dt_i = dt_i[0]
        state0_i = vel0_i.tolist() + pos0_i.tolist()
        t_span = (0.0, dt_i)
        p = (dr_i, RPM_i, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
             Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
             Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr, dvel_pred_i)  # Parameters of the system
        import timeit
        # start = timeit.default_timer()
        result_solve_ivp = solve_ivp(UUV_3DOF, t_span, state0_i, args=p, method='RK45', t_eval=[dt_i])
        # end = timeit.default_timer()
        # online_time.append(end-start)
        state_pred_i = np.squeeze(result_solve_ivp.y).tolist()
        lhs_pred_i = lhs_predict(np.squeeze(vel0_i), np.squeeze(dvel_pred_i[0]))
        forcefull_pred_i = force_full_predict(np.squeeze(vel0_i), np.squeeze(dvel_pred_i[0]), dr_i, RPM_i,
                                              external_trainable_variables)
        vel_pred.append(state_pred_i[0:3])
        pos_pred.append(state_pred_i[3:])
        dvel_pred.append(dvel_pred_i[0])
        lhs_pred.append(lhs_pred_i)
        forcefull_pred.append(forcefull_pred_i)
    vel_pred = np.array(vel_pred).reshape((len(target_data), 3))
    dvel_pred = np.array(dvel_pred).reshape((len(target_data), 3))
    pos_pred = np.array(pos_pred).reshape((len(target_data), 3))
    lhs_pred = np.array(lhs_pred).reshape((len(target_data), 3))
    forcefull_pred = np.array(forcefull_pred).reshape((len(target_data), 3))
    data = np.hstack((target_data[['u0', 'v0', 'r0', 'x0', 'y0', 'psi0', 'dr', 'RPM', 'delta_t']].values,
                      target_data[['u', 'v', 'r', 'x', 'y', 'psi']].values,
                      vel_pred, pos_pred,
                      dvel_pred,
                      forcefull_pred,
                      lhs_pred))
    # print(np.mean(online_time))

    simulation_df = pd.DataFrame(data,
                                 columns=['u0', 'v0', 'r0', 'x0', 'y0', 'psi0', 'targ_rud', 'RPM', 'delta_t',
                                          'u_true', 'v_true', 'r_true', 'x_true', 'y_true', 'psi_true',
                                          'u_pred', 'v_pred', 'r_pred', 'x_pred', 'y_pred', 'psi_pred',
                                          'du_pred', 'dv_pred', 'dr_pred',
                                          'Xfull0_pred', 'Yfull0_pred', 'Nfull0_pred',
                                          'lhs_u0_pred', 'lhs_v0_pred', 'lhs_r0_pred'])
    simulation_df['psi_true'] = convert_psi(simulation_df['psi_true'].values).tolist()
    simulation_df['psi_pred'] = convert_psi(simulation_df['psi_pred'].values).tolist()
    simulation_df['du_true'] = target_data['udot'].values.tolist()
    simulation_df['dv_true'] = target_data['vdot'].values.tolist()
    simulation_df['dr_true'] = target_data['rdot'].values.tolist()
    simulation_df['X_true'] = target_data['X'].values.tolist()
    simulation_df['Y_true'] = target_data['Y'].values.tolist()
    simulation_df['N_true'] = target_data['N'].values.tolist()
    simulation_df.to_csv(saved_path, index=False)


def extract_3dof_states_forces_openloop(target_data_path=None, saved_path=None, external_trainable_variables=None):
    [Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
     Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
     Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr] = external_trainable_variables

    target_data = pd.read_csv(target_data_path)
    dt = target_data[['delta_t']].values
    dr = target_data[['dr']].values
    RPM = target_data[['RPM']].values
    vel0_i = target_data[['u0', 'v0', 'r0']].values[0]
    pos0_i = target_data[['x0', 'y0', 'psi0']].values[0]
    state0_i = vel0_i.tolist() + pos0_i.tolist()
    state0_pred = []
    vel_pred = []
    dvel_pred = []
    pos_pred = []
    lhs_pred = []
    forcefull_pred = []
    # online_time = []
    for dr_i, RPM_i, dt_i, in zip(dr, RPM, dt):
        dvel_pred_i = []
        dr_i = dr_i[0]
        RPM_i = RPM_i[0]
        dt_i = dt_i[0]
        t_span = (0.0, dt_i)
        p = (dr_i, RPM_i, Xrr, Xud, Xvr, Xuu, Xvv, Xdrdr,
             Yrd, Yrar, Yvd, Yur, Yvar, Yuv, Yvav, Y0, Ydr, Ydradr,
             Nrd, Nrar, Nvd, Nur, Nvar, Nuv, Nvav, N0, Ndr, Ndradr, dvel_pred_i)  # Parameters of the system
        import timeit
        # start = timeit.default_timer()
        try:
            result_solve_ivp = solve_ivp(UUV_3DOF, t_span, state0_i, args=p, method='RK45', t_eval=[dt_i])
        except:
            print('Divergence: %s' % saved_path)
            break
        # end = timeit.default_timer()
        # online_time.append(end-start)
        state0_pred.append(state0_i)
        state_pred_i = np.squeeze(result_solve_ivp.y).tolist()
        lhs_pred_i = lhs_predict(np.squeeze(vel0_i), np.squeeze(dvel_pred_i[0]))
        forcefull_pred_i = force_full_predict(np.squeeze(vel0_i), np.squeeze(dvel_pred_i[0]), dr_i, RPM_i,
                                              external_trainable_variables)
        vel_pred.append(state_pred_i[0:3])
        pos_pred.append(state_pred_i[3:])
        dvel_pred.append(dvel_pred_i[0])
        lhs_pred.append(lhs_pred_i)
        forcefull_pred.append(forcefull_pred_i)
        state0_i = state_pred_i

    state0_pred = np.array(state0_pred).reshape((-1, 6))
    vel_pred = np.array(vel_pred).reshape((-1, 3))
    dvel_pred = np.array(dvel_pred).reshape((-1, 3))
    pos_pred = np.array(pos_pred).reshape((-1, 3))
    lhs_pred = np.array(lhs_pred).reshape((-1, 3))
    forcefull_pred = np.array(forcefull_pred).reshape((-1, 3))
    data = np.hstack((state0_pred, target_data[['dr', 'RPM', 'delta_t']].values[:len(state0_pred)],
                      target_data[['u', 'v', 'r', 'x', 'y', 'psi']].values[:len(state0_pred)],
                      vel_pred, pos_pred,
                      dvel_pred,
                      forcefull_pred,
                      lhs_pred))
    # print(np.mean(online_time))

    simulation_df = pd.DataFrame(data,
                                 columns=['u0', 'v0', 'r0', 'x0', 'y0', 'psi0', 'targ_rud', 'RPM', 'delta_t',
                                          'u_true', 'v_true', 'r_true', 'x_true', 'y_true', 'psi_true',
                                          'u_pred', 'v_pred', 'r_pred', 'x_pred', 'y_pred', 'psi_pred',
                                          'du_pred', 'dv_pred', 'dr_pred',
                                          'Xfull0_pred', 'Yfull0_pred', 'Nfull0_pred',
                                          'lhs_u0_pred', 'lhs_v0_pred', 'lhs_r0_pred'])
    simulation_df['psi_true'] = convert_psi(simulation_df['psi_true'].values[:len(data)]).tolist()
    simulation_df['psi_pred'] = convert_psi(simulation_df['psi_pred'].values[:len(data)]).tolist()
    simulation_df['du_true'] = target_data['udot'].values[:len(data)].tolist()
    simulation_df['dv_true'] = target_data['vdot'].values[:len(data)].tolist()
    simulation_df['dr_true'] = target_data['rdot'].values[:len(data)].tolist()
    simulation_df['X_true'] = target_data['X'].values[:len(data)].tolist()
    simulation_df['Y_true'] = target_data['Y'].values[:len(data)].tolist()
    simulation_df['N_true'] = target_data['N'].values[:len(data)].tolist()
    simulation_df.to_csv(saved_path, index=False)


def extract_6dof_states_forces_closedloop(target_data_path=None, saved_path=None, external_trainable_variables=None):
    [Xuu, Xvv, Xww, Xud, Xvr, Xwq, Xqq, Xrr, Xdrdr, Xdede,
     Yuv, Yvav, Yvw, Y0, Yvd, Yvq, Ywp, Ywr, Yur, Yup, Yvar, Yrd, Ypq, Yqr, Yrar, Ydr, Ydradr,
     Zuw, Zwaw, Zvw, Z0, Zwd, Zwr, Zvp, Zvq, Zuq, Zup, Zwaq, Zqd, Zrp, Zqr, Zqaq, Zde, Zdeade,
     K0, Kvw, Kup, Kvq, Kwr, Kpd, Kqr, Kpq, Kpap, Kdr, Kde,
     Muw, Mwaw, Mvw, M0, Mwd, Mwr, Mvp, Muq, Mup, Mwaq, Mqd, Mrp, Mqr, Mqaq, Mde, Mdeade,
     Nuv, Nvav, Nvw, N0, Nvd, Nvq, Nwp, Nur, Nup, Nvar, Nrd, Npq, Nqr, Nrar, Ndr, Ndradr] = external_trainable_variables

    target_data = pd.read_csv(target_data_path)
    dt = target_data[['delta_t']].values
    dr = target_data[['dr']].values
    de = target_data[['de']].values
    RPM = target_data[['RPM']].values
    vel0 = target_data[['u0', 'v0', 'w0','p0', 'q0', 'r0']].values
    pos0 = target_data[['x0', 'y0','z0' ,'phi0','theta0','psi0']].values
    vel_pred = []
    dvel_pred = []
    pos_pred = []
    lhs_pred = []
    forcefull_pred = []
    # online_time = []
    for dr_i,de_i ,RPM_i, dt_i, vel0_i, pos0_i in zip(dr,de ,RPM, dt, vel0, pos0):
        dvel_pred_i = []
        dr_i = dr_i[0]
        de_i = de_i[0]
        RPM_i = RPM_i[0]
        dt_i = dt_i[0]
        state0_i = vel0_i.tolist() + pos0_i.tolist()
        t_span = (0.0, dt_i)
        p = (dr_i,de_i ,RPM_i, Xuu, Xvv, Xww, Xud, Xvr, Xwq, Xqq, Xrr, Xdrdr, Xdede,
             Yuv, Yvav, Yvw, Y0, Yvd, Yvq, Ywp, Ywr, Yur, Yup, Yvar, Yrd, Ypq, Yqr, Yrar, Ydr, Ydradr,
             Zuw, Zwaw, Zvw, Z0, Zwd, Zwr, Zvp, Zvq, Zuq, Zup, Zwaq, Zqd, Zrp, Zqr, Zqaq, Zde, Zdeade,
             K0, Kvw, Kup, Kvq, Kwr, Kpd, Kqr, Kpq, Kpap, Kdr, Kde,
             Muw, Mwaw, Mvw, M0, Mwd, Mwr, Mvp, Muq, Mup, Mwaq, Mqd, Mrp, Mqr, Mqaq, Mde, Mdeade,
             Nuv, Nvav, Nvw, N0, Nvd, Nvq, Nwp, Nur, Nup, Nvar, Nrd, Npq, Nqr, Nrar, Ndr, Ndradr, dvel_pred_i)  # Parameters of the system
        import timeit
        # start = timeit.default_timer()
        result_solve_ivp = solve_ivp(UUV_6DOF, t_span, state0_i, args=p, method='RK45', t_eval=[dt_i])
        # end = timeit.default_timer()
        # online_time.append(end-start)
        state_pred_i = np.squeeze(result_solve_ivp.y).tolist()
        lhs_pred_i = lhs_predict(np.squeeze(vel0_i), np.squeeze(dvel_pred_i[0]))
        forcefull_pred_i = force_full_predict(np.squeeze(vel0_i), np.squeeze(dvel_pred_i[0]), dr_i, RPM_i,
                                              external_trainable_variables)
        vel_pred.append(state_pred_i[0:6])
        pos_pred.append(state_pred_i[6:])
        dvel_pred.append(dvel_pred_i[0])
        lhs_pred.append(lhs_pred_i)
        forcefull_pred.append(forcefull_pred_i)
    vel_pred = np.array(vel_pred).reshape((len(target_data), 6))
    dvel_pred = np.array(dvel_pred).reshape((len(target_data), 6))
    pos_pred = np.array(pos_pred).reshape((len(target_data), 6))
    lhs_pred = np.array(lhs_pred).reshape((len(target_data), 6))
    forcefull_pred = np.array(forcefull_pred).reshape((len(target_data), 6))
    data = np.hstack((target_data[['u0', 'v0', 'w0','p0', 'q0', 'r0' ,'x0', 'y0', 'z0','phi0', 'theta0', 'psi0' ,'dr','de', 'RPM', 'delta_t']].values,
                      target_data[['u', 'v', 'w','p', 'q', 'r' ,'x', 'y', 'z','phi', 'theta', 'psi']].values,
                      vel_pred, pos_pred,
                      dvel_pred,
                      forcefull_pred,
                      lhs_pred))
    # print(np.mean(online_time))

    simulation_df = pd.DataFrame(data,
                                 columns=['u0', 'v0', 'w0','p0', 'q0', 'r0' ,'x0', 'y0', 'z0','phi0', 'theta0', 'psi0', 'targ_rud','targ_ele', 'RPM', 'delta_t',
                                          'u_true', 'v_true', 'w_true','p_true', 'q_true', 'r_true' ,'x_true', 'y_true', 'z_true','phi_true', 'theta_true', 'psi_true',
                                          'u_pred', 'v_pred', 'w_pred','p_pred', 'q_pred', 'r_pred' ,'x_pred', 'y_pred', 'z_pred','phi_pred', 'theta_pred', 'psi_pred',
                                          'du_pred', 'dv_pred', 'dw_pred','dp_pred', 'dq_pred', 'dr_pred',
                                          'Xfull0_pred', 'Yfull0_pred', 'Zfull0_pred','Kfull0_pred', 'Mfull0_pred', 'Nfull0_pred',
                                          'lhs_u0_pred', 'lhs_v0_pred', 'lhs_w0_pred','lhs_p0_pred', 'lhs_q0_pred', 'lhs_r0_pred'])
    simulation_df['psi_true'] = convert_psi(simulation_df['psi_true'].values).tolist()
    simulation_df['psi_pred'] = convert_psi(simulation_df['psi_pred'].values).tolist()
    simulation_df['du_true'] = target_data['udot'].values.tolist()
    simulation_df['dv_true'] = target_data['vdot'].values.tolist()
    simulation_df['dw_true'] = target_data['wdot'].values.tolist()
    simulation_df['dp_true'] = target_data['pdot'].values.tolist()
    simulation_df['dq_true'] = target_data['qdot'].values.tolist()
    simulation_df['dr_true'] = target_data['rdot'].values.tolist()
    simulation_df['X_true'] = target_data['X'].values.tolist()
    simulation_df['Y_true'] = target_data['Y'].values.tolist()
    simulation_df['Z_true'] = target_data['N'].values.tolist()
    simulation_df['K_true'] = target_data['X'].values.tolist()
    simulation_df['M_true'] = target_data['Y'].values.tolist()
    simulation_df['N_true'] = target_data['N'].values.tolist()
    simulation_df.to_csv(saved_path, index=False)


dof = 3
setting = 'setting2_fnn_standard_MAE_new'
coefficients_path = r'E:\WORK\Project\PINN_project\Main_project\UUV_PINN\220801_main\PINN_UUV_discretize_wo_dt_acc_complex\pretrained_model\%sdof\%s\coefficients.csv' % (
dof, setting)
# Extract coefficients
df = pd.read_csv(coefficients_path)
coefficients = df.values[-1].tolist()[1:]

# Closed-loop
if False:
    target_data_folder = r'data\preprocess_%sdof\ref_folder' % dof
    saved_folder = 'pretrained_model/%sdof/%s/error_analysis_closedloop' % (dof, setting)
    if os.path.isdir(saved_folder):
        pass
    else:
        os.makedirs(saved_folder)
    for file_name in os.listdir(target_data_folder):
        file_path = os.path.join(target_data_folder, file_name)
        saved_path = os.path.join(saved_folder, file_name)
        if dof == 3:
            extract_3dof_states_forces_closedloop(file_path, saved_path, external_trainable_variables=coefficients)
        if dof == 6:
            extract_6dof_states_forces_closedloop(file_path, saved_path, external_trainable_variables=coefficients)

# Open-loop
if False:
    target_data_folder = r'data\preprocess_%sdof\ref_folder' % dof
    saved_folder = 'pretrained_model/%sdof/%s/error_analysis_openloop' % (dof, setting)
    if os.path.isdir(saved_folder):
        pass
    else:
        os.makedirs(saved_folder)
    for file_name in os.listdir(target_data_folder):
        file_path = os.path.join(target_data_folder, file_name)
        saved_path = os.path.join(saved_folder, file_name)
        if dof == 3:
            extract_3dof_states_forces_openloop(file_path, saved_path, external_trainable_variables=coefficients)
        # if dof == 6:
        #     extract_6dof_states_forces_closedloop(model_path, model_name, base_data_path, file_path, saved_path,
        #                                           use_psi=False)

if True:
    target_data_folder = 'pretrained_model/%sdof/%s/error_analysis_closedloop' % (dof, setting)
    saved_path = os.path.join('pretrained_model/%sdof/%s' % (dof, setting),
                              r'evaluation_metric_separate_data_closedloop.csv')
    if dof == 3:
        export_evaluation_metric_3dof(target_data_folder, saved_path)
    elif dof == 6:
        export_evaluation_metric_6dof(target_data_folder, saved_path)

if True:
    target_data_folder = 'pretrained_model/%sdof/%s/error_analysis_openloop' % (dof, setting)
    saved_path = os.path.join('pretrained_model/%sdof/%s' % (dof, setting),
                              r'evaluation_metric_separate_data_openloop.csv')
    if dof == 3:
        export_evaluation_metric_3dof(target_data_folder, saved_path)
    elif dof == 6:
        export_evaluation_metric_6dof(target_data_folder, saved_path)
