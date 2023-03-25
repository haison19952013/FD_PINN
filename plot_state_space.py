import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib
import seaborn as sns


def plot_position_vars_6dof(data_path, saved_fig_path_xyz, saved_fig_path_phithetapsi):
    # plot x,y,z
    position_true = ['x_true', 'y_true', 'z_true', '_']
    position_pred = ['x_pred', 'y_pred', 'z_pred', '_']
    ICs = ['x0', 'y0', 'z0', '_']
    ylabels = ['$N/L$', '$E/L$', '$D/L$', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t_pred = np.cumsum(data['delta_t'].values)
    t_true = [0] + np.cumsum(data['delta_t'].values).tolist()
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 12))
    i = 1
    for ax, pos_true, pos_pred, IC, ylabel in zip(axes, position_true, position_pred, ICs, ylabels):
        if i == 4:
            rud_plot = ax.plot(t_pred, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t_pred, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            if i == 3:
                L = 1
            else:
                L = 3.826
            pred_plot = ax.plot(t_pred, data[pos_pred] / L, 'r--', linewidth=4, alpha=0.8, label='PINN')
            pos_true = np.array([data[IC].values.tolist()[0]] + data[pos_true].values.tolist())
            true_plot = ax.plot(t_true, pos_true / L, 'k-', label='Actual data', linewidth=2, alpha=0.8)
            lns = true_plot + pred_plot
        ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        # added these three lines
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path_xyz)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()

    # plot x,y,z
    position_true = ['phi_true', 'theta_true', 'psi_true', '_']
    position_pred = ['phi_pred', 'theta_pred', 'psi_pred', '_']
    ICs = ['phi0', 'theta0', 'psi0', '_']
    ylabels = ['$\phi$ [rad]', r'$\theta$ [rad]', '$\psi$ [rad]', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t_pred = np.cumsum(data['delta_t'].values)
    t_true = [0] + np.cumsum(data['delta_t'].values).tolist()
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 12))
    i = 1
    for ax, pos_true, pos_pred, IC, ylabel in zip(axes, position_true, position_pred, ICs, ylabels):
        if i == 4:
            rud_plot = ax.plot(t_pred, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t_pred, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            if i == 3:
                L = 1
            else:
                L = 3.826
            pred_plot = ax.plot(t_pred, data[pos_pred] / L, 'r--', linewidth=4, alpha=0.8, label='PINN')
            pos_true = np.array([data[IC].values.tolist()[0]] + data[pos_true].values.tolist())
            true_plot = ax.plot(t_true, pos_true / L, 'k-', label='Actual data', linewidth=2, alpha=0.8)
            lns = true_plot + pred_plot
        ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        # added these three lines
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path_phithetapsi)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()


def plot_states_6dof(data_path, saved_fig_path):
    velocity_true = ['u_true', 'v_true', 'r_true', '_']
    velocity_pred = ['u_pred', 'v_pred', 'r_pred', '_']
    ICs = ['u0', 'v0', 'r0', '_']
    ylabels = ['$u$ [m/s]', '$v$ [m/s]', '$r$ [rad/s]', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t_pred = np.cumsum(data['delta_t'].values)
    t_true = [0] + np.cumsum(data['delta_t'].values).tolist()
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 12))
    i = 1
    for ax, vel_true, vel_pred, IC, ylabel in zip(axes, velocity_true, velocity_pred, ICs, ylabels):
        if i == 4:
            rud_plot = ax.plot(t_pred, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t_pred, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            pred_plot_1 = ax.plot(t_pred, data[vel_pred], 'r--', linewidth=4, alpha=0.8, label='PINN')
            vel_true = [data[IC].values.tolist()[0]] + data[vel_true].values.tolist()
            true_plot = ax.plot(t_true, vel_true, 'k-', label='Actual data', linewidth=2, alpha=0.8)
            lns = true_plot + pred_plot_1
        # added these three lines
        ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_forces_6dof(data_path, saved_fig_path):
    force_true = ['X_true', 'Y_true', 'N_true', '-']
    force_pred = ['lhs_u0_pred', 'lhs_v0_pred', 'lhs_r0_pred', '-']
    ylabels = ['$X$ [N]', '$Y$ [N]', '$N$ [N.m]', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t = [0] + np.cumsum(data['delta_t'].values).tolist()[:-1]
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 10))
    i = 1
    for ax, f_true, f_pred, ylabel, in zip(axes, force_true, force_pred, ylabels):
        if i == 4:
            rud_plot = ax.plot(t, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            true_plot = ax.plot(t, data[f_true], 'k-', label='Actual data', linewidth=2, alpha=0.8)
            pred_plot = ax.plot(t, data[f_pred], 'r--', linewidth=3, alpha=0.8, label='PINN')
            lns = true_plot + pred_plot
            ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        # added these three lines
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_position_vars_3dof(data_path, saved_fig_path):
    position_true = ['x_true', 'y_true', 'psi_true', '_']
    position_pred = ['x_pred', 'y_pred', 'psi_pred', '_']
    ICs = ['x0', 'y0', 'psi0', '_']
    ylabels = ['$N/L$', '$E/L$', '$\psi$ [rad]', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t_pred = np.cumsum(data['delta_t'].values)
    t_true = [0] + np.cumsum(data['delta_t'].values).tolist()
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 12))
    i = 1
    for ax, pos_true, pos_pred, IC, ylabel in zip(axes, position_true, position_pred, ICs, ylabels):
        if i == 4:
            rud_plot = ax.plot(t_pred, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t_pred, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            if i == 3:
                L = 1
            else:
                L = 3.826
            pred_plot = ax.plot(t_pred, data[pos_pred] / L, 'r--', linewidth=4, alpha=0.8, label='PINN')
            pos_true = np.array([data[IC].values.tolist()[0]] + data[pos_true].values.tolist())
            true_plot = ax.plot(t_true, pos_true / L, 'k-', label='Actual data', linewidth=2, alpha=0.8)
            lns = true_plot + pred_plot
        ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        # added these three lines
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    plt.close()


def plot_states_3dof(data_path, saved_fig_path):
    velocity_true = ['u_true', 'v_true', 'r_true', '_']
    velocity_pred = ['u_pred', 'v_pred', 'r_pred', '_']
    ICs = ['u0', 'v0', 'r0', '_']
    ylabels = ['$u$ [m/s]', '$v$ [m/s]', '$r$ [rad/s]', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t_pred = np.cumsum(data['delta_t'].values)
    t_true = [0] + np.cumsum(data['delta_t'].values).tolist()
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 12))
    i = 1
    for ax, vel_true, vel_pred, IC, ylabel in zip(axes, velocity_true, velocity_pred, ICs, ylabels):
        if i == 4:
            rud_plot = ax.plot(t_pred, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t_pred, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            pred_plot_1 = ax.plot(t_pred, data[vel_pred], 'r--', linewidth=4, alpha=0.8, label='PINN')
            vel_true = [data[IC].values.tolist()[0]] + data[vel_true].values.tolist()
            true_plot = ax.plot(t_true, vel_true, 'k-', label='Actual data', linewidth=2, alpha=0.8)
            lns = true_plot + pred_plot_1
        # added these three lines
        ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_forces_3dof(data_path, saved_fig_path):
    force_true = ['X_true', 'Y_true', 'N_true', '-']
    force_pred = ['lhs_u0_pred', 'lhs_v0_pred', 'lhs_r0_pred', '-']
    ylabels = ['$X$ [N]', '$Y$ [N]', '$N$ [N.m]', '$\delta_r$ [rad]']
    data = pd.read_csv(data_path)
    t = [0] + np.cumsum(data['delta_t'].values).tolist()[:-1]
    (fig, axes) = plt.subplots(4, 1, figsize=(18, 10))
    i = 1
    for ax, f_true, f_pred, ylabel, in zip(axes, force_true, force_pred, ylabels):
        if i == 4:
            rud_plot = ax.plot(t, data['targ_rud'], 'g-', label='$\delta_r$', linewidth=1.5, alpha=0.8)
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            rpm_plot = ax2.plot(t, data['RPM'], 'k-', label='RPM', linewidth=2, alpha=0.8)
            ax2.set_ylabel('RPM', fontsize=25)
            ax2.tick_params(labelsize=20)
            lns = rud_plot + rpm_plot
        else:
            true_plot = ax.plot(t, data[f_true], 'k-', label='Actual data', linewidth=2, alpha=0.8)
            pred_plot = ax.plot(t, data[f_pred], 'r--', linewidth=3, alpha=0.8, label='PINN')
            lns = true_plot + pred_plot
            ax.set_xlim(0, None)
        ax.set_ylabel(ylabel, fontsize=25)
        ax.set_xlabel("Time [s]", fontsize=25)
        ax.grid(True)
        # added these three lines
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, fontsize=25, loc='center left', bbox_to_anchor=(1.1, 0.5),
                  ncol=1, fancybox=True, shadow=True)
        ax.tick_params(labelsize=20)
        i += 1
    fig_path = os.path.join(saved_fig_path)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_trajectories(data_path, saved_fig_path, square_fig=True):
    data = pd.read_csv(data_path)
    L = 3.826
    (fig, ax) = plt.subplots(1, 1, figsize=(12, 8))
    xn_true = data['x_true'].values
    xe_true = data['y_true'].values
    ax.scatter(xe_true[0] / L, xn_true[0] / L, s=550, marker='^', alpha=1, label="Start point", c="pink",
               edgecolors='face', linewidths=2.5)
    ax.plot(data['y_pred'] / L, data['x_pred'] / L, '--', label='PINN', linewidth=3., color='red')
    ax.plot(xe_true / L, xn_true / L, label='Actual Data', color='black')
    ax.set_xlabel("$E/L$", fontsize=20)
    ax.set_ylabel("$N/L$", fontsize=20)
    ax.grid(True)
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=24)
    # ax.set_aspect('equal', adjustable='box')
    # ax.set_box_aspect(1)
    # if square_fig:
    #     ax.set_aspect('equal', adjustable='box')
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    if dx >= dy:
        d_2 = (dx - dy) / 2
        y0_new = y0 - d_2
        y1_new = y1 + d_2
        ax.set_ylim(y0_new, y1_new)
    else:
        d_2 = (dy - dx) / 2
        x0_new = x0 - d_2
        x1_new = x1 + d_2
        ax.set_xlim(x0_new, x1_new)
    ax.set_aspect('equal', adjustable='box')
    ax.locator_params(axis='both', nbins=8)
    fig.tight_layout()
    fig.savefig(saved_fig_path, dpi=300, bbox_inches='tight')
    plt.close()


dof = 3
# Figures for open-loop: state
if False:
    data_saved_folder = os.path.join('error_analysis_openloop')

    figsaved_folder_train = os.path.join( 'error_analysis_figs_openloop', 'state_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join( 'error_analysis_figs_openloop', 'state_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_state.png'
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_states_3dof(file_path, fig_saved_path)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_states_3dof(file_path, fig_saved_path)

# Figures for open-loop: trajectory
if False:
    data_saved_folder = os.path.join('error_analysis_openloop')
    figsaved_folder_train = os.path.join('error_analysis_figs_openloop', 'trajectory_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_openloop', 'trajectory_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_trajectory.png'
        if data_file_name.split("_")[0] == 'ZZH' or data_file_name.split("_")[0] == 'HM':
            square_fig = False
        else:
            square_fig = True
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_trajectories(file_path, fig_saved_path, square_fig)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_trajectories(file_path, fig_saved_path, square_fig)

# Figures for open-loop: position
if False:
    data_saved_folder = os.path.join('error_analysis_openloop')

    figsaved_folder_train = os.path.join('error_analysis_figs_openloop', 'trajectory_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_openloop', 'trajectory_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_position.png'
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_position_vars_3dof(file_path, fig_saved_path)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_position_vars_3dof(file_path, fig_saved_path)

# Figures for open-loop: force
if False:
    data_saved_folder = os.path.join('error_analysis_openloop')

    figsaved_folder_train = os.path.join('error_analysis_figs_openloop', 'force_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_openloop', 'force_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_state.png'
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_forces_3dof(file_path, fig_saved_path)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_forces_3dof(file_path, fig_saved_path)

# Figures for closed-loop: state
if True:
    data_saved_folder = os.path.join('error_analysis_closedloop')

    figsaved_folder_train = os.path.join( 'error_analysis_figs_closedloop', 'state_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_closedloop', 'state_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_state.png'
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_states_3dof(file_path, fig_saved_path)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_states_3dof(file_path, fig_saved_path)

# Figures for closed-loop: trajectory
if True:
    data_saved_folder = os.path.join('error_analysis_closedloop')

    figsaved_folder_train = os.path.join('error_analysis_figs_closedloop', 'trajectory_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_closedloop', 'trajectory_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_trajectory.png'
        if data_file_name.split("_")[0] == 'ZZH' or data_file_name.split("_")[0] == 'HM':
            square_fig = False
        else:
            square_fig = True
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_trajectories(file_path, fig_saved_path, square_fig)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_trajectories(file_path, fig_saved_path, square_fig)

# Figures for closed-loop: position
if True:
    data_saved_folder = os.path.join('error_analysis_closedloop')

    figsaved_folder_train = os.path.join('error_analysis_figs_closedloop', 'trajectory_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_closedloop', 'trajectory_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_position.png'
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_position_vars_3dof(file_path, fig_saved_path)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_position_vars_3dof(file_path, fig_saved_path)

# Figures for closed-loop: force
if True:
    data_saved_folder = os.path.join('error_analysis_closedloop')

    figsaved_folder_train = os.path.join( 'error_analysis_figs_closedloop', 'force_train')
    if os.path.isdir(figsaved_folder_train):
        pass
    else:
        os.makedirs(figsaved_folder_train)

    figsaved_folder_test = os.path.join('error_analysis_figs_closedloop', 'force_test')
    if os.path.isdir(figsaved_folder_test):
        pass
    else:
        os.makedirs(figsaved_folder_test)

    for data_file_name in os.listdir(data_saved_folder):
        file_path = os.path.join(data_saved_folder, data_file_name)
        fig_file_name = data_file_name.split(".")[0] + '_state.png'
        if data_file_name.split("_")[3] == '20to20' \
                or data_file_name.split("_")[3] == 'Drm40' \
                or data_file_name.split("_")[3] == 'Dr40' \
                or data_file_name.split("_")[3] == '30to20':
            fig_saved_path = os.path.join(figsaved_folder_test, fig_file_name)
            plot_forces_3dof(file_path, fig_saved_path)
        else:
            fig_saved_path = os.path.join(figsaved_folder_train, fig_file_name)
            plot_forces_3dof(file_path, fig_saved_path)