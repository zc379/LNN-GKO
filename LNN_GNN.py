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

class RNNEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_seq):
        out, _ = self.rnn(input_seq)               # [S, T, hidden_dim]
        h_last = out[:, -1, :]
        return torch.tanh(self.out_proj(h_last))   # [S, latent_dim]

class LiquidNeuronEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.in_proj = nn.Linear(1, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, input_seq):
        S, T, D = input_seq.shape  # input_seq: [S, T, 2] (T, dt)
        h = torch.zeros(S, self.W_h.out_features, device=input_seq.device)
        for t in range(T):
            x_t = self.in_proj(input_seq[:, t, 0:1])  # temp
            delta = input_seq[:, t, 1]  # dt
            dh = (-h + torch.tanh(x_t + self.W_h(h))) / self.tau
            h_new = h + delta.unsqueeze(-1) * dh
            # h = h + h_new
        return torch.tanh(self.out_proj(h_new))
def load_all_deeponet_datasets(data_dir=BATCH_ROOT_PATH, pattern="deepONet_training_dataset*.pt", test_ratio=0.2):
    dataset_files = glob.glob(os.path.join(data_dir, pattern))

    if not dataset_files:
        print("not found data", data_dir)
        return None, None

    print(f"found {len(dataset_files)} data：")
    for idx, file in enumerate(dataset_files):
        print(f" {idx+1:02d}. {os.path.basename(file)}")

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
def compute_geodesic_distances(edge_index, num_nodes, sensor_indices, node_features, base_dmax=8.5):
    import networkx as nx
    import torch

    z_vals = node_features[:, 2]  # [num_nodes], normalized z
    G = nx.Graph()
    G.add_edges_from(edge_index.t().tolist())

    dist = torch.full((num_nodes, len(sensor_indices)), float('inf'))

    for j, src in enumerate(sensor_indices):
        try:
            paths = nx.single_source_shortest_path(G, source=int(src))
            all_nodes = sorted({n for path in paths.values() for n in path})
            all_nodes_tensor = torch.tensor(all_nodes, device=z_vals.device)
            z_all = z_vals.index_select(0, all_nodes_tensor)

            # {node_id: z_val}
            node_z_map = {int(n.item()): float(z.item()) for n, z in zip(all_nodes_tensor, z_all)}

            for tgt, path in paths.items():
                d = len(path) - 1
                num_below = sum(node_z_map[n] < -0.95 for n in path)
                dist[tgt, j] = d + num_below
        except:
            continue

    return dist

class EdgeFeatureGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, use_feature_attention=False):
        super().__init__(aggr='add')
        self.use_feature_attention = use_feature_attention
        self.lin_node = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(3, out_channels)
        self.attn = nn.Parameter(torch.Tensor(1, out_channels))
        self.reset_parameters()


    def reset_parameters(self):

        nn.init.xavier_uniform_(self.lin_node.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, x, edge_index, edge_attr):

        if self.use_feature_attention:


            x_xyz = x[:, 0:3].float()  # shape: [N, 3]

            x_xyz_weighted = self.w_xyz(x_xyz) * x_xyz  # shape: [N, 3]

            x_type = x[:, 3:6].float()  # shape: [N, 3]
            x_type = x_type[:, 3]
            x_type_weighted = self.w_type(x_type) * x_type  # * 0.2
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
        alpha = (z * self.attn).sum(dim=-1)  # [E]
        alpha = F.leaky_relu(alpha, 0.1)
        alpha = softmax(alpha, index)
        return z * alpha.view(-1, 1)

class GKO_GNN_Model(nn.Module):
    def __init__(self, temp_input_dim=26, latent_dim=64, hidden_dim=64, kernel_scale=3.5, d_max=8.5):
        super().__init__()
        self.kernel_scale = nn.Parameter(torch.tensor(kernel_scale))
        self.d_max = d_max
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

        # self.sensor_encoder = RNNEncoder()
        self.sensor_encoder = LiquidNeuronEncoder()

        self.sensor_encoder1 = nn.Sequential(
            nn.Linear(26, 96),
            nn.LeakyReLU(0.01),
            nn.Linear(96, latent_dim),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.01)
        )

        self.gnn1 = EdgeFeatureGATLayer(71, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.gnn2 = EdgeFeatureGATLayer(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)
        self.gnn3 = EdgeFeatureGATLayer(hidden_dim, hidden_dim)
        self.norm3 = GraphNorm(hidden_dim)
        self.gnn4 = EdgeFeatureGATLayer(hidden_dim, hidden_dim)
        self.norm4 = GraphNorm(hidden_dim)
        self.gnn5 = EdgeFeatureGATLayer(hidden_dim, hidden_dim)
        self.norm5 = GraphNorm(hidden_dim)
        self.latent_norm = nn.LayerNorm(latent_dim)


        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        self.activation = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x_full = data.x[:, :3].float()                    # [N, 3]
        temp_seq = data.x_branch[:, 6:]          # [M, T]
        _, T = temp_seq.shape
        delta_t_seq = torch.tensor([2.0] * (T - 1) + [120.0], device=temp_seq.device)
        delta_t_seq = delta_t_seq / delta_t_seq.max()
        delta_t_seq = delta_t_seq.unsqueeze(0).expand(temp_seq.size(0), -1)
        input_seq = torch.stack([temp_seq, delta_t_seq], dim=-1)
        z_sensors = self.sensor_encoder(input_seq)         # [M, latent_dim]
        dist = data.geodesic_dist.to(x_full.device)
        K = torch.exp(-dist / (self.kernel_scale + 1e-6))  # [N, M]
        K = K.masked_fill(dist > self.d_max, 0.0)
        x_latent = K @ z_sensors  # [N, latent_dim]
        x_latent = self.latent_norm(x_latent)
###################################################################################################
        x_xyz = data.x.float()
        keep = [0, 1, 2, 4, 5, 6]
        x_xyz = x_xyz[:, keep]
        x_xyz_weighted = self.w_xyz(x_xyz[:, 0:3]) * x_xyz[:, 0:3]
        x_type = x_xyz[:, 3:6].float()
        x_type_weighted = self.w_type(x_type) * x_type
        x_ini_temp = data.ini_GNN_temp.float()
        x_ini_temp = x_ini_temp * self.weights
        x_all = torch.cat([x_xyz_weighted, x_type_weighted, x_ini_temp, x_latent], dim=1)
###################################################################################################
        x1 = self.dropout(self.activation(self.norm1(self.gnn1(x_all, data.edge_index, data.edge_attr))))

        x2_in = x1
        x2 = self.dropout(self.activation(self.norm2(self.gnn2(x1, data.edge_index, data.edge_attr))))
        x2 = x2 + x2_in

        x3_in = x2
        x3 = self.dropout(self.activation(self.norm3(self.gnn3(x2, data.edge_index, data.edge_attr))))
        x3 = x3 + x3_in

        x4_in = x3
        x4 = self.dropout(self.activation(self.norm4(self.gnn4(x3, data.edge_index, data.edge_attr))))
        x4 = x4 + x4_in

        x5_in = x4
        x5 = self.dropout(self.activation(self.norm5(self.gnn5(x4, data.edge_index, data.edge_attr))))
        x5 = x5 + x5_in
        out = self.out_mlp(x5).squeeze(-1)  # [N]
        return out

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
                mse_loss = F.mse_loss(pred, data.temp_GNN_y.squeeze(-1))
                mse_loss.backward()
                optimizer.step()
                total_train_loss += mse_loss.item()

                # === real RMSE ===
                output_real = 0.5 * (pred + 1) * (2100 - 25) + 25
                y_real = 0.5 * (data.temp_GNN_y.squeeze(-1) + 1) * (2100 - 25) + 25
                real_rmse = F.mse_loss(output_real, y_real).sqrt().item()
                real_rmse_sum += real_rmse

            avg_train_loss = total_train_loss / len(train_loader)
            real_world_train_rmse = real_rmse_sum / len(train_loader)

            model.eval()
            test_total_loss = 0.0
            test_real_rmse_total = 0.0
            with torch.no_grad():
                for test_data in test_loader:
                    test_data = test_data.to(device)
                    test_pred = model(test_data)
                    test_mse = F.mse_loss(test_pred, test_data.temp_GNN_y.squeeze(-1))
                    test_total_loss += test_mse.item()

                    # === real RMSE ===
                    test_output_real = 0.5 * (test_pred + 1) * (2100 - 25) + 25
                    test_y_real = 0.5 * (test_data.temp_GNN_y.squeeze(-1) + 1) * (2100 - 25) + 25
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
        print(" Detected Ctrl+C - stopping training early")

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
    DeepONetModel = GKO_GNN_Model()
    print("training......")
    train_deeponet(DeepONetModel, train_loader, test_loader, epochs=50000, lr=1e-6, save_dir="./training_logs", patience=20)

