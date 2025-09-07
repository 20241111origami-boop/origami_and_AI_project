import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import random # For MockValidator

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import KFold
# --- 既存のインポート文に追加 ---
from validator import validate_fold_file
# --- 1. Configuration (Phase C) ---
class Config:
    DATA_DIR = "data"
    HIDDEN_DIM = 64
    LATENT_DIM = 32
    GAT_HEADS = 4
    EPOCHS = 200 # Note: Each epoch now has two training steps per batch
    BATCH_SIZE = 8
    LEARNING_RATE = 0.005
    LAMBDA_RECON = 0.01
    LAMBDA_CLASS = 1.0
    K_SPLITS = 5
    RANDOM_STATE = 42
    USE_DFT_IN_SUPERVISED = True # Controls DFT in the supervised part
    DFT_GAMMA = 2.0
    
    # --- NEW: Feedback Loop parameters (from Design Doc v1.1) ---
    FB_ALPHA = 1.0  # Base learning strength for feedback
    FB_BETA = 0.5   # How much reward influences strength



class RewardTransformer:
    """
    validator.py (v4) の出力を、GNNへの学習信号（報酬と重み）に変換するクラス。
    設計書v1.1に基づき、validator.pyの実装に完全に適合させたver1.1。
    """
    
    # validator.pyが返すエラータイプと、その深刻度レベルのマッピング
    ERROR_TYPE_TO_LEVEL = {
        # Level 0-G (大域的エラー)
        "VertexOutOfBounds": "0-G",
        "BoundaryEdgeNotOnFrame": "0-G",
        "ImproperEdgeIntersection": "0-G",
        "ImproperCollinearOverlap": "0-G",
        # Level 0-L (局所幾何学エラー)
        "BoundaryCount": "0-L",
        "OverlappingCreases": "0-L",
        # Level 1 (折り紙工学定理エラー)
        "Maekawa": "1",
        "Kawasaki": "1",
        "BigLittleBigCondition": "1",
        "GeneralizedBigLittleBigLemma": "1",
        # Default / File Errors
        "File Error": "0-G",
        "Format Error": "0-G",
        "InvalidCoordinateFormat": "0-G"
    }

    def __init__(self, dft_gamma=2.0, max_errors_to_report=10):
        self.dft_gamma = dft_gamma
        self.max_errors_to_report = max_errors_to_report
        self.level_penalties = {"0-G": 0.5, "0-L": 0.2, "1": 0.1, "default": 0.1}

    def _get_edges_for_vertex(self, vertex_idx, graph_data):
        """指定された頂点に接続する全てのエッジのインデックスを返すヘルパー関数"""
        edge_indices = []
        # graph_data.edge_index は [2, num_total_edge_entries] の形状
        # 0行目がsource, 1行目がtarget
        for i in range(graph_data.edge_index.size(1)):
            if graph_data.edge_index[0, i] == vertex_idx or graph_data.edge_index[1, i] == vertex_idx:
                edge_indices.append(i)
        return list(set(edge_indices)) # 重複を削除

    def transform(self, is_valid, errors, edge_confidence, graph_data):
        """
        検証結果をスカラー報酬と要素ごとの重みに変換する。

        Args:
            is_valid (bool): validator.pyからの全体検証結果。
            errors (list[dict]): validator.pyからのエラーリスト。
            edge_confidence (torch.Tensor): モデルが生成した各エッジに対するp_maxのテンソル。
            graph_data (torch_geometric.data.Data): 生成されたグラフデータ。

        Returns:
            tuple[float, torch.Tensor]: (scalar_reward, element_weights)
        """
        num_edges = graph_data.edge_attr.size(0)
        device = graph_data.x.device

        if is_valid:
            scalar_reward = 1.0
            element_weights = torch.ones(num_edges, dtype=torch.float, device=device)
            return scalar_reward, element_weights

        # --- 失敗時の処理 ---
        # 1. スカラー報酬の計算
        base_reward = -0.5
        
        first_error_type = errors[0].get("type", "Unknown") if errors else "Unknown"
        error_level = self.ERROR_TYPE_TO_LEVEL.get(first_error_type, "default")
        level_penalty = self.level_penalties.get(error_level, self.level_penalties["default"])
        
        count_penalty = 0.1 * (len(errors) / self.max_errors_to_report)
        scalar_reward = base_reward - level_penalty - count_penalty

        # 2. DFTに基づく要素ごとの重み (element_weights) の計算
        element_weights = torch.ones(num_edges, dtype=torch.float, device=device)

        for error in errors:
            responsible_edge_indices = []
            context = error.get("context", {})

            # ケースA: エラーがエッジに直接関連付けられている
            if "edge_indices" in context: # 例: 交差エラー
                responsible_edge_indices.extend(context["edge_indices"])
            elif "edge_index" in context: # 例: 境界線エラー
                responsible_edge_indices.append(context["edge_index"])

            # ケースB: エラーが頂点に関連付けられている
            elif "vertex" in error: # 例: 川崎定理、前川定理エラー
                vertex_idx = error["vertex"]
                # この頂点に接続する全てのエッジを責任対象とする
                responsible_edge_indices.extend(self._get_edges_for_vertex(vertex_idx, graph_data))

            # 責任のあるエッジの重みを更新
            for edge_idx in set(responsible_edge_indices): # 重複を排除
                # .foldのエッジインデックスとPyGの無向グラフエッジインデックスを対応させる
                # PyGではエッジiは (2*i) と (2*i+1) の2つのエントリを持つ
                pyg_indices = [2 * edge_idx, 2 * edge_idx + 1]
                
                for pyg_idx in pyg_indices:
                    if pyg_idx < len(edge_confidence):
                        confidence = edge_confidence[pyg_idx]
                        weight = confidence.detach().pow(self.dft_gamma).clamp(min=0.01, max=1.0)
                        element_weights[pyg_idx] = weight
        
        return scalar_reward, element_weights



# --- MockValidatorの代わりにこのクラスを定義 ---
class RealValidator:
    """
    validator.pyを呼び出すためのラッパークラス。
    PyGグラフを.fold形式の辞書に変換するブリッジ機能を持つ。
    """
    def _pyg_data_to_fold_dict(self, graph_data):
        """
        torch_geometric.data.Dataオブジェクトを、validatorが要求する
        .fold形式のPython辞書に変換する。
        """
        # 頂点座標の変換
        # [[x1, y1], [x2, y2], ...]の形式
        vertices_coords = graph_data.x.tolist()

        # エッジ情報の変換
        # PyGの無向グラフ表現から、.foldの一方向表現に戻す
        edge_set = set()
        edges_vertices = []
        
        # edge_index: [2, num_edges * 2]
        # edge_attr: [num_edges * 2]
        for i in range(graph_data.edge_index.size(1)):
            u, v = graph_data.edge_index[0, i].item(), graph_data.edge_index[1, i].item()
            
            # 重複エッジを避けるため、(min, max)のタプルで管理
            if u < v:
                edge_tuple = (u, v)
                if edge_tuple not in edge_set:
                    edge_set.add(edge_tuple)
                    # .foldは1-based indexを期待する場合があるが、validator.pyは0-basedで
                    # 書かれているため、ここでは0-basedのまま渡す。
                    edges_vertices.append([u, v])
        
        # エッジの割り当て情報を再構築
        # 注意: このデモでは、無向グラフの一方向のエッジ属性のみを取得する
        # より堅牢な実装では、両方向のエッジ属性が一致することを保証すべき
        edge_type_reverse_mapping = {0: "B", 1: "M", 2: "V", 3: "F", 4: "U"}
        edges_assignment = []
        for i in range(0, graph_data.edge_attr.size(0), 2):
             assignment_idx = graph_data.edge_attr[i].item()
             edges_assignment.append(edge_type_reverse_mapping.get(assignment_idx, "U"))

        return {
            "vertices_coords": vertices_coords,
            "edges_vertices": edges_vertices,
            "edges_assignment": edges_assignment
        }

    def validate(self, graph_data):
        """
        MockValidatorと同じインターフェースを持つvalidateメソッド。
        """
        # 1. PyGグラフを.fold辞書形式に変換
        fold_dict = self._pyg_data_to_fold_dict(graph_data)
        
        # 2. 外部のvalidatorモジュールの関数を呼び出す
        # validate_fold_fileは辞書の代わりにファイルパスを要求する可能性があるため、
        # ここでは一時ファイルに書き出すか、validator側が辞書を受け取れるように
        # 修正する必要がある。ここではvalidatorが辞書を直接扱える関数
        # `validate_fold_data`を持っていると仮定する。
        # (もしvalidator.pyがファイルパスしか受け付けない場合は、
        #  json.dumpで一時ファイルに書き出す処理が必要)

        # --- ここでvalidator.pyの関数を呼び出す ---
        # validator.pyの `validate_fold_file` を少し改造して、
        # データ辞書を直接受け取れる `validate_fold_data` があると仮定します。
        # もし `validate_fold_file` しかない場合は、一時ファイルに保存する必要があります。
        # ここでは、簡単のため、validator.pyに以下のような関数があるとします：
        # def validate_fold_data(data): ... (validate_fold_fileの中身と同じ)
        
        # このデモでは、validator.pyの関数を直接呼び出す代わりに、
        # 辞書をファイルに保存して、そのパスを渡す堅牢な方法を示します。
        temp_file_path = "temp_validation.fold"
        with open(temp_file_path, 'w') as f:
            json.dump(fold_dict, f)
        
        result = validate_fold_file(temp_file_path) # validator.pyの関数を呼び出し
        
        os.remove(temp_file_path) # 一時ファイルを削除
        
        return result["valid"], result["errors"]
# --- 4. OrigamiDataset (Fixed loading issue) ---
class OrigamiDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform)
        
        # Fix: Use weights_only=False to handle torch_geometric data loading
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except Exception as e:
            print(f"Warning: Could not load processed file ({e}). Reprocessing...")
            # If loading fails, force reprocessing
            if os.path.exists(self.processed_paths[0]):
                os.remove(self.processed_paths[0])
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    
    @property
    def raw_file_names(self):
        raw_dir = os.path.join(self.root, 'raw')
        if not os.path.exists(raw_dir): 
            return []
        return [f for f in os.listdir(raw_dir) if f.endswith('.fold')]
    
    @property
    def processed_file_names(self): 
        return ['data.pt']
    
    def download(self): 
        pass
    
    def process(self):
        data_list = []
        edge_type_mapping = {"B": 0, "M": 1, "V": 2, "F": 3, "U": 4}
        
        for filename in self.raw_file_names:
            path = os.path.join(self.raw_dir, filename)
            try:
                with open(path, 'r') as f: 
                    fold_data = json.load(f)
                
                vertices_coords = fold_data['vertices_coords']
                edges_vertices = fold_data['edges_vertices']
                edges_assignment = fold_data['edges_assignment']
                
                if vertices_coords and isinstance(vertices_coords[0], list): 
                    x = torch.tensor(vertices_coords, dtype=torch.float)
                else: 
                    x = torch.tensor(vertices_coords, dtype=torch.float).view(len(vertices_coords) // 2, 2)
                
                num_nodes = x.size(0)
                source_nodes, target_nodes, edge_attrs = [], [], []
                
                for i, edge in enumerate(edges_vertices):
                    u, v = edge[0], edge[1]
                    if not (0 <= u < num_nodes and 0 <= v < num_nodes): 
                        continue
                    source_nodes.extend([u, v])
                    target_nodes.extend([v, u])
                    edge_type = edge_type_mapping.get(edges_assignment[i], 4)
                    edge_attrs.extend([edge_type, edge_type])
                
                if not source_nodes: 
                    continue
                
                edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
            
            except Exception as e:
                print(f"Warning: Could not process file {filename}: {e}")
                continue
        
        if not data_list: 
            raise ValueError("No valid graphs found!")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# --- 5. GNNAutoencoder Model (with new generation method) ---
class GNNAutoencoder(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, latent_dim, num_edge_classes, heads=4):
        super().__init__()
        self.conv1 = GATConv(node_feat_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, latent_dim, heads=1)
        self.decode_nodes = torch.nn.Linear(latent_dim, node_feat_dim)
        self.decode_edges = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_edge_classes)
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        recon_x = self.decode_nodes(z)
        edge_z = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return recon_x, self.decode_edges(edge_z)

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        return self.decode(z, data.edge_index)

    @torch.no_grad()
    def generate_sample_with_confidence(self, data):
        """Generates a new graph based on the input graph's structure and returns confidence."""
        self.eval() # Ensure model is in evaluation mode
        
        z = self.encode(data.x, data.edge_index)
        recon_x, recon_edge_logits = self.decode(z, data.edge_index)
        
        # Calculate confidence (p_max) from logits
        edge_probs = F.softmax(recon_edge_logits, dim=-1)
        edge_confidence = edge_probs.max(dim=-1).values
        
        # Create a new graph data object from the model's output
        generated_graph = Data(
            x=recon_x,
            edge_index=data.edge_index,
            edge_attr=recon_edge_logits.argmax(dim=-1) # Use predicted assignments
        )
        return generated_graph, edge_confidence

# --- 6. Training & Evaluation Pipeline (Hybrid Loop) ---
def train_one_epoch(model, loader, optimizer, config, validator, reward_transformer):
    model.train()
    total_sup_loss, total_fb_loss = 0, 0
    
    for data in loader:
        # --- Step A: Supervised Learning ---
        optimizer.zero_grad()
        recon_x, recon_edge_logits = model(data)
        
        loss_recon = F.mse_loss(recon_x, data.x)
        if config.USE_DFT_IN_SUPERVISED:
            # Our original DFT loss based on p_max of the prediction
            probs = F.softmax(recon_edge_logits, dim=-1)
            p_max = probs.max(dim=-1).values
            weights = p_max.detach().pow(config.DFT_GAMMA)
            unweighted_loss = F.cross_entropy(recon_edge_logits, data.edge_attr, reduction='none')
            loss_class = (unweighted_loss * weights).mean()
        else:
            loss_class = F.cross_entropy(recon_edge_logits, data.edge_attr)
        
        supervised_loss = config.LAMBDA_RECON * loss_recon + config.LAMBDA_CLASS * loss_class
        supervised_loss.backward()
        optimizer.step()
        total_sup_loss += supervised_loss.item() * data.num_graphs

        # --- Step B: Feedback Loop Learning ---
        optimizer.zero_grad()
        
        # 1. Generate a sample
        generated_graph, edge_confidence = model.generate_sample_with_confidence(data)
        
        # 2. Validate it
        is_valid, errors = validator.validate(generated_graph)
        
        # 3. Get feedback signal
        scalar_reward, element_weights = reward_transformer.transform(is_valid, errors, edge_confidence, generated_graph)
        
        # 4. Calculate feedback loss
        model.train() # Switch back to train mode for feedback pass
        recon_x_fb, recon_edge_logits_fb = model(generated_graph)
        
        fb_loss_recon = F.mse_loss(recon_x_fb, generated_graph.x)
        unweighted_fb_loss_class = F.cross_entropy(recon_edge_logits_fb, generated_graph.edge_attr, reduction='none')
        # Apply DFT weights from the RewardTransformer
        fb_loss_class = (unweighted_fb_loss_class * element_weights).mean()
        
        # 5. Apply feedback strength
        feedback_strength = config.FB_ALPHA - config.FB_BETA * scalar_reward
        total_feedback_loss = (config.LAMBDA_RECON * fb_loss_recon + config.LAMBDA_CLASS * fb_loss_class) * feedback_strength
        
        total_feedback_loss.backward()
        optimizer.step()
        total_fb_loss += total_feedback_loss.item() * data.num_graphs

    num_samples = len(loader.dataset)
    return total_sup_loss / num_samples, total_fb_loss / num_samples

@torch.no_grad()
def evaluate(model, loader, config):
    # Evaluation remains the same: test supervised performance
    model.eval()
    total_loss, correct_edges, total_edges = 0, 0, 0
    for data in loader:
        recon_x, recon_edge_logits = model(data)
        loss_recon = F.mse_loss(recon_x, data.x)
        loss_class = F.cross_entropy(recon_edge_logits, data.edge_attr)
        loss = config.LAMBDA_RECON * loss_recon + config.LAMBDA_CLASS * loss_class
        total_loss += loss.item() * data.num_graphs
        pred = recon_edge_logits.argmax(dim=-1)
        correct_edges += (pred == data.edge_attr).sum().item()
        total_edges += data.edge_attr.size(0)
    num_samples = len(loader.dataset)
    return total_loss / num_samples, correct_edges / total_edges

# --- 7. Main Execution ---

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    config = Config()
    
    # Check if data directory exists
    if not os.path.exists(config.DATA_DIR):
        print(f"Creating data directory: {config.DATA_DIR}")
        os.makedirs(config.DATA_DIR)
        
    raw_dir = os.path.join(config.DATA_DIR, 'raw')
    if not os.path.exists(raw_dir):
        print(f"Creating raw data directory: {raw_dir}")
        os.makedirs(raw_dir)
        print(f"Please place your .fold files in {raw_dir} and run again.")
        return
    
    try:
        dataset = OrigamiDataset(root=config.DATA_DIR, force_reload=False)
    except ValueError as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Dataset loaded successfully: {len(dataset)} samples")
    if len(dataset) == 0:
        print("No data found. Please check your .fold files.")
        return
    
    # --- CHANGED: Use RealValidator instead of MockValidator ---
    validator = RealValidator()
    reward_transformer = RewardTransformer(dft_gamma=config.DFT_GAMMA)

    kf = KFold(n_splits=config.K_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_results = []
    
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"\n--- Processing Fold {fold+1}/{config.K_SPLITS} ---")
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        num_edge_classes = 5
        model = GNNAutoencoder(
            node_feat_dim=dataset.num_node_features, hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM, num_edge_classes=num_edge_classes, heads=config.GAT_HEADS
        )
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        for epoch in range(1, config.EPOCHS + 1):
            sup_loss, fb_loss = train_one_epoch(model, train_loader, optimizer, config, validator, reward_transformer)
            if epoch % 10 == 0 or epoch == config.EPOCHS:
                val_loss, val_acc = evaluate(model, val_loader, config)
                print(f'Epoch: {epoch:03d}, SupLoss: {sup_loss:.4f}, FbLoss: {fb_loss:.4f}, ValLoss: {val_loss:.4f}, ValAcc: {val_acc:.4f}')

        final_loss, final_acc = evaluate(model, val_loader, config)
        fold_results.append({'loss': final_loss, 'acc': final_acc})

    # Aggregate and print final results
    val_losses = np.array([res['loss'] for res in fold_results])
    val_accuracies = np.array([res['acc'] for res in fold_results])
    print(f"\n--- Final Performance (Mean ± Std across {config.K_SPLITS} folds) ---")
    print(f"Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Val Edge Accuracy: {np.mean(val_accuracies):.4f} ± {np.std(val_accuracies):.4f}")



if __name__ == '__main__':
    main()
