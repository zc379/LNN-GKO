import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from pure_GNN import GNNWithInjectedTemp
from RNN_GNN import RNN_GKO_GNN_Model
from LNN_GNN import GKO_GNN_Model
# from GKO_GNN import GKO_GNN_Model
# from Multi_kernel_GKO_GNN import GKO_GNN_Model



# model = GNNWithInjectedTemp()
# model = RNN_GKO_GNN_Model()
model = GKO_GNN_Model()
model.load_state_dict(torch.load("training_logs/LNNGKO/best_model_GNN_temperature_prediction.pth"))#Naive_GNN/0.5_training/

model.to('cuda')
model.eval()

data_list = torch.load("E:/PIML machine learning/piml/piml_bulging_distortion/dataset/"
                       "training_dataloader/deepONet_training_dataset1223.pt")
data = data_list[370].to('cuda')


def autoscale(ax, xyz):
    xlim = [xyz[:, 0].min(), xyz[:, 0].max()]
    ylim = [xyz[:, 1].min(), xyz[:, 1].max()]
    zlim = [xyz[:, 2].min(), xyz[:, 2].max()]
    max_range = max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    x_center = np.mean(xlim)
    y_center = np.mean(ylim)
    z_center = np.mean(zlim)
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
    ax.set_zlim(z_center - max_range / 2, z_center + max_range / 2)


def plot_deeponet_prediction(data, model):
    with torch.no_grad():
        pred = model(data)

    pred = 0.5 * (pred.cpu().numpy() + 1) * (2100 - 25) + 25
    # true = 0.5 * (data.temp_GNN_y.cpu().numpy() + 1) * (2100 - 25) + 25
    true = 0.5 * (data.temp_GNN_y.cpu().numpy().squeeze(-1) + 1) * (2100 - 25) + 25
    coords = data.x[:, :3].cpu().numpy()
    coords[:, 0] = 0.5 * coords[:, 0] * 300.0 + 150.0
    coords[:, 1] = 0.5 * coords[:, 1] * 150.0 + 75.0
    coords[:, 2] = 0.5 * coords[:, 2] * 40.0 + 20.0

    coords_branch = data.x_branch[:, :3].cpu().numpy()
    coords_branch[:, 0] = 0.5 * coords_branch[:, 0] * 300.0 + 150.0
    coords_branch[:, 1] = 0.5 * coords_branch[:, 1] * 150.0 + 75.0
    coords_branch[:, 2] = 0.5 * coords_branch[:, 2] * 40.0 + 20.0

    # # === 0：branch nodes ===
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
    #                       color='lightgrey', s=3, alpha=0.1,
    #                       label='All nodes', depthshade=False)
    # sc = ax.scatter(coords_branch[:, 0], coords_branch[:, 1], coords_branch[:, 2],
    #                 facecolors='red', edgecolors='black', linewidths=0.4, s=10,
    #            label='Sampled nodes', depthshade=False)
    # cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    # cbar.ax.tick_params(labelsize=12)
    # ax.set_xlabel('X (mm)', fontsize=14)
    # ax.set_ylabel('Y (mm)', fontsize=14)
    # ax.set_zlabel('Z (mm)', fontsize=14)
    # autoscale(ax, coords)
    # ax.set_title(f'Sampled nodes', fontsize=14)
    # plt.tight_layout()
    # plt.legend()
    # # plt.savefig("I:/LNN_GKO paper/Branch nodes.png", dpi=300)
    # plt.show()
    #
    # === 0：FEM ===
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=true, cmap='inferno', s=1)#c=true, cmap='inferno',
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('True Temperature (°C)', fontsize=14)
    ax.set_xlabel('X (mm)', fontsize=14)
    ax.set_ylabel('Y (mm)', fontsize=14)
    ax.set_zlabel('Z (mm)', fontsize=14)
    autoscale(ax, coords)
    ax.set_title(f'FEM Simulation Temperature', fontsize=14)
    plt.tight_layout()
    # plt.savefig("I:/LNN_GKO paper/FEM Simulation16.png", dpi=300)
    plt.show()

    # 1：PRED
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc1 = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=pred, cmap='viridis', s=10)
    cbar = plt.colorbar(sc1, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Predicted Temperature (°C)', fontsize=14)
    ax.set_title('LNN-GKO Predicted Temperature ', fontsize=14)
    ax.set_xlabel('X (mm)', fontsize=14)
    ax.set_ylabel('Y (mm)', fontsize=14)
    ax.set_zlabel('Z (mm)', fontsize=14)
    autoscale(ax, coords)
    plt.tight_layout()
    # plt.savefig("I:/LNN_GKO paper/Vanilla GNN Predicted16.png", dpi=300)
    plt.show()


    # 2：ERROR
    error = np.abs(pred - true)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc2 = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=error, cmap='turbo', s=10)
    cbar = plt.colorbar(sc2, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Absolute Error (°C)', fontsize=14)
    ax.set_title('LNN-GKO Predicted Error Map', fontsize=14)
    ax.set_xlabel('X (mm)', fontsize=14)
    ax.set_ylabel('Y (mm)', fontsize=14)
    ax.set_zlabel('Z (mm)', fontsize=14)
    autoscale(ax, coords)
    plt.tight_layout()
    # plt.savefig("I:/LNN_GKO paper/Vanilla GNN Predicted Error Map16.png", dpi=300)
    plt.show()

    # === 3：ERROR sub removed ===
    z_min = coords[:, 2].min()
    mask = coords[:, 2] >= (z_min + 1.0)
    coords_masked = coords[mask]
    error_masked = error[mask]
    true_masked = true[mask]
    pred_masked = pred[mask]
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords_masked[:, 0], coords_masked[:, 1], coords_masked[:, 2], c=error_masked, cmap='coolwarm', s=10)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Absolute Error (°C)', fontsize=14)
    ax.set_xlabel('X (mm)', fontsize=14)
    ax.set_ylabel('Y (mm)', fontsize=14)
    ax.set_zlabel('Z (mm)', fontsize=14)
    autoscale(ax, coords_masked)
    ax.set_title(f'LNN-GKO Predicted Error Map (Substrate Removed)', fontsize=14)
    plt.tight_layout()
    # plt.savefig("I:/LNN_GKO paper/Vanilla GNN Predicted Error Map (substrate Removed)16.png", dpi=300)
    plt.show()


    # RMSE
    rmse = np.sqrt(np.mean((pred - true) ** 2))

    max_error = np.max(error)
    mae = np.mean(np.abs(pred - true))

    print(f"RMSE: {rmse:.4f}")
    print(f"Max Error: {max_error:.4f}")
    print(f"MAE: {mae:.4f}")



    rmse_substrate_Removed = np.sqrt(np.mean((pred_masked - true_masked) ** 2))

    max_error_substrate_Removed = np.max(error_masked)
    mae_substrate_Removed = np.mean(np.abs(pred_masked - true_masked))

    print(f"RMSE_substrate_Removed: {rmse_substrate_Removed:.4f}")
    print(f"Max Error_substrate_Removed: {max_error_substrate_Removed:.4f}")
    print(f"MAE_substrate_Removed: {mae_substrate_Removed:.4f}")

if __name__ == "__main__":
    plot_deeponet_prediction(data, model)


