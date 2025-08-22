from torch_geometric.nn import GraphNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import DataLoader
import os, csv, math, glob
from sklearn.model_selection import train_test_split
from torch_geometric.utils import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
import networkx as nx
import pandas as pd

BATCH_ROOT_PATH = "E:/PIML machine learning/piml/piml_bulging_distortion/dataset/training_dataloader"

def load_all_deeponet_datasets(data_dir=BATCH_ROOT_PATH, pattern="deepONet_training_dataset*.pt", test_ratio=0.2):
    dataset_files = glob.glob(os.path.join(data_dir, pattern))

    if not dataset_files:
        print("not found data", data_dir)
        return None, None

    print(f"found {len(dataset_files)} data：")
    for idx, file in enumerate(dataset_files):
        print(f" {idx + 1:02d}. {os.path.basename(file)}")

    print("loading...")

    all_data = []
    for file in dataset_files:
        data_list = torch.load(file)
        if not isinstance(data_list, list):
            print(f"waring {os.path.basename(file)} not list")
            continue
        all_data.extend(data_list)

    print(f"number: {len(all_data)}")

    train_data, test_data = train_test_split(all_data, test_size=test_ratio, random_state=42)
    print(f"training: {len(train_data)}")
    print(f"testing: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print(f"training batch: {len(train_loader)}")
    print(f"testing batch: {len(test_loader)}")

    return train_loader, test_loader

class EdgeFeatureGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, use_feature_attention=False):
        super().__init__(aggr='add')
        self.use_feature_attention = use_feature_attention
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(3, out_channels)
        self.attn = nn.Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()
        self.weights = nn.Parameter(torch.ones(1))


        self.w_xyz = nn.Sequential(
            nn.Linear(3, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        self.w_type = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_attr):

        if self.use_feature_attention:

            x_xyz = x[:, 0:3].float()  # shape: [N, 3]

            x_xyz_weighted = self.w_xyz(x_xyz) * x_xyz  # shape: [N, 3]

            x_type = x[:, 3:6].float()  # shape: [N, 3]
            x_type_weighted = self.w_type(x_type) * x_type * 0.2
            # shape: [N, 3]
            x_ini_temp = x[:, 6].float()
            x_ini_temp = x_ini_temp * self.weights

            x_rear = x[:, 7:].float()
            # shape: [N, 3]
            x = torch.cat([x_xyz_weighted, x_type_weighted, x_ini_temp, x_rear], dim=1)
            #print("xshape",x.shape)

        else:
            x = x.float()

        x = self.lin_node(x)  # [N, in_channels] -> [N, out_channels]

        edge_attr = torch.as_tensor(edge_attr, device=edge_index.device).float()
        if edge_attr.dim() == 3:
            edge_attr = edge_attr.squeeze(0)

        edge_attr = self.lin_edge(edge_attr)  #  [E, 3] -> [E, out_channels]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr, index, ptr, size_i):

        z = x_j * torch.sigmoid(edge_attr)  # [E, out_channels]
        alpha = (z * self.attn).sum(dim=-1)  #  [E]
        alpha = F.leaky_relu(alpha, 0.1)
        alpha = softmax(alpha, index)
        return z * alpha.view(-1, 1)

class GNNWithInjectedTemp(nn.Module):
    def __init__(self, temp_feat_dim=64, trunk_hidden_dim=64):
        super().__init__()

        self.temp_mlp = nn.Sequential(
            nn.Linear(26, 96),
            nn.LeakyReLU(0.01),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01)
        )

        in_dim = 7 + temp_feat_dim

        self.gnn1 = EdgeFeatureGATLayer(in_dim, trunk_hidden_dim, use_feature_attention=False)
        self.gnn2 = EdgeFeatureGATLayer(trunk_hidden_dim, trunk_hidden_dim, use_feature_attention=False)
        self.gnn3 = EdgeFeatureGATLayer(trunk_hidden_dim, trunk_hidden_dim, use_feature_attention=False)
        self.gnn4 = EdgeFeatureGATLayer(trunk_hidden_dim, trunk_hidden_dim, use_feature_attention=False)
        self.gnn5 = EdgeFeatureGATLayer(trunk_hidden_dim, trunk_hidden_dim, use_feature_attention=False)
        self.gnn6 = EdgeFeatureGATLayer(trunk_hidden_dim, trunk_hidden_dim, use_feature_attention=False)

        self.norm1 = GraphNorm(trunk_hidden_dim)
        self.norm2 = GraphNorm(trunk_hidden_dim)
        self.norm3 = GraphNorm(trunk_hidden_dim)
        self.norm4 = GraphNorm(trunk_hidden_dim)
        self.norm5 = GraphNorm(trunk_hidden_dim)
        self.norm6 = GraphNorm(trunk_hidden_dim)

        self.residual_mlp = nn.Sequential(
            nn.Linear(trunk_hidden_dim, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.activation = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x = data.x.float()  # [x, y, z, z_def, onehot_3] → [N, 7]
        keep = [0, 1, 2, 4, 5, 6]  # [x, y, z, z_def, 1, 1, 1]
        x = x[:, keep]
        x = torch.cat([x, data.ini_GNN_temp], dim=1)

        N = x.size(0)
        device = x.device

        temp_features = torch.zeros(N, 64, device=device)

        temp_seq = data.x_branch[:, 6:]  #  [N, T]

        encoded_temp = self.temp_mlp(temp_seq)  # [M, 192]
        temp_features[data.sample_idx] = encoded_temp

        # [N, 7 + temp_feat_dim]
        x = torch.cat([x, temp_features], dim=1)

        x = self.dropout(self.activation(self.norm1(self.gnn1(x, data.edge_index, data.edge_attr))))
        x = self.dropout(self.activation(self.norm2(self.gnn2(x, data.edge_index, data.edge_attr))))
        x = self.dropout(self.activation(self.norm3(self.gnn3(x, data.edge_index, data.edge_attr))))
        x = self.dropout(self.activation(self.norm4(self.gnn4(x, data.edge_index, data.edge_attr))))
        x = self.dropout(self.activation(self.norm5(self.gnn5(x, data.edge_index, data.edge_attr))))
        # x = self.dropout(self.activation(self.norm6(self.gnn6(x, data.edge_index, data.edge_attr))))

        dz_pred = self.residual_mlp(x)  # [N, 1]

        dz_pred = data.ini_GNN_temp.float() + dz_pred

        return dz_pred

def train_deeponet(model, train_loader, test_loader, epochs=10000, lr=1e-6, save_dir="./training_logs", patience=8, beta=1e-3):
    import os, time, csv, torch
    import torch.nn.functional as F

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log = {
        'epoch': [],
        'train_loss': [],
        'test_loss': [],
        'real_train_rmse_mm': [],
        'real_test_rmse_mm': [],
        'elapsed time': []
    }

    best_test_loss = float('inf')
    best_model_path = os.path.join(save_dir, "best_model_GNN_temperature_prediction.pth")
    checkpoint_path = os.path.join(save_dir, "checkpoint.pth")
    log_path = os.path.join(save_dir, "epoch_loss_summary_temperature_pred.csv")
    no_improve_count = 0
    start_epoch = 1
    elapsed_time = 0.0

    # === Load checkpoint if exists ===
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        elapsed_time = checkpoint['elapsed time']
        print(f"Resuming from epoch {start_epoch}, elapsed time {elapsed_time:.1f}s")
    if os.path.exists(log_path):
        prev_log = pd.read_csv(log_path)
        if len(prev_log) > 0:
            log['epoch'] = prev_log['epoch'].tolist()
            log['train_loss'] = prev_log['train_loss'].tolist()
            log['test_loss'] = prev_log['test_loss'].tolist()
            log['real_train_rmse_mm'] = prev_log['real_train_rmse_mm'].tolist()
            log['real_test_rmse_mm'] = prev_log['real_test_rmse_mm'].tolist()
            log['elapsed time'] = prev_log['elapsed time'].tolist()

    try:
        for epoch in range(start_epoch, epochs + 1):
            temp_start = time.time()
            model.train()
            total_train_loss = 0.0
            real_rmse_sum = 0.0

            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                pred = model(data)
                mse_loss = F.mse_loss(pred, data.temp_GNN_y)
                mse_loss.backward()
                optimizer.step()
                total_train_loss += mse_loss.item()

                # === 反归一化 real RMSE ===
                output_real = 0.5 * (pred.squeeze(-1) + 1) * (2100 - 25) + 25
                y_real = 0.5 * (data.temp_GNN_y.squeeze(-1) + 1) * (2100 - 25) + 25
                real_rmse = F.mse_loss(output_real, y_real).sqrt().item()
                real_rmse_sum += real_rmse

            avg_train_loss = total_train_loss / len(train_loader)
            real_world_train_rmse = real_rmse_sum / len(train_loader)

            # === 评估阶段 ===
            model.eval()
            test_total_loss = 0.0
            test_real_rmse_total = 0.0
            with torch.no_grad():
                for test_data in test_loader:
                    test_data = test_data.to(device)
                    test_pred = model(test_data)
                    test_mse = F.mse_loss(test_pred, test_data.temp_GNN_y)
                    test_total_loss += test_mse.item()

                    # === real RMSE ===
                    test_output_real = 0.5 * (test_pred + 1) * (2100 - 25) + 25
                    test_y_real = 0.5 * (test_data.temp_GNN_y + 1) * (2100 - 25) + 25
                    test_real_rmse = F.mse_loss(test_output_real, test_y_real).sqrt().item()
                    test_real_rmse_total += test_real_rmse

            test_avg_loss = test_total_loss / len(test_loader)
            test_avg_real_rmse = test_real_rmse_total / len(test_loader)

            epoch_time = time.time() - temp_start
            elapsed_time += epoch_time
            log['epoch'].append(epoch)
            log['train_loss'].append(avg_train_loss)
            log['test_loss'].append(test_avg_loss)
            log['real_train_rmse_mm'].append(real_world_train_rmse)
            log['real_test_rmse_mm'].append(test_avg_real_rmse)
            log['elapsed time'].append(elapsed_time)

            print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.6f} | Test Loss: {test_avg_loss:.6f} | "
                  f"Train RMSE: {real_world_train_rmse:.6f} mm | Test RMSE: {test_avg_real_rmse:.6f} mm | "
                  f"Time Elapsed: {elapsed_time:.1f} sec")

            if test_avg_loss < best_test_loss and test_avg_loss > 0:
                best_test_loss = test_avg_loss
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch} with test loss {best_test_loss:.6f}")
                no_improve_count = 0
            else:
                no_improve_count += 1
                print(f"⏸ No improvement for {no_improve_count} epochs")

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'elapsed time': elapsed_time
            }, checkpoint_path)

            # Save CSV
            partial_log_path = os.path.join(save_dir, "epoch_loss_summary_temperature_pred.csv")
            with open(partial_log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'test_loss', 'real_train_rmse_mm', 'real_test_rmse_mm', 'elapsed time'])
                for e, tl, tsl, trmse, t_rmse, tsec in zip(
                    log['epoch'], log['train_loss'], log['test_loss'],
                    log['real_train_rmse_mm'], log['real_test_rmse_mm'],
                    log['elapsed time']
                ):
                    writer.writerow([e, tl, tsl, trmse, t_rmse, tsec])

            if no_improve_count >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    except KeyboardInterrupt:
        print("Detected Ctrl+C - stopping training early")

    finally:
        final_log_path = os.path.join(save_dir, "epoch_loss_summary.csv")
        with open(final_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'test_loss', 'real_train_rmse_mm', 'real_test_rmse_mm', 'elapsed time'])
            for e, tl, tsl, trmse, t_rmse, tsec in zip(
                log['epoch'], log['train_loss'], log['test_loss'],
                log['real_train_rmse_mm'], log['real_test_rmse_mm'],
                log['elapsed time']
            ):
                writer.writerow([e, tl, tsl, trmse, t_rmse, tsec])
        print(f"Final epoch log saved to {final_log_path}")
        print("Training complete. Logs and model saved.")


if __name__ == "__main__":
    train_loader, test_loader = load_all_deeponet_datasets(data_dir=BATCH_ROOT_PATH, pattern="deepONet_training_dataset*.pt", test_ratio=0.2)
    DeepONetModel = GNNWithInjectedTemp()
    print("training......")
    train_deeponet(DeepONetModel, train_loader, test_loader, epochs=50000, lr=1e-5, save_dir="./training_logs", patience=20)

