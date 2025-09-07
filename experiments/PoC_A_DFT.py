import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv

# --- 1. Configuration ---
class Config:
    # Data parameters
    DATA_DIR = "data" # Root directory for data
    
    # Model hyperparameters
    HIDDEN_DIM = 64
    LATENT_DIM = 32
    GAT_HEADS = 4
    
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 8 # Adjust based on dataset size and memory
    LEARNING_RATE = 0.005
    
    # Loss weights
    LAMBDA_RECON = 0.01 # Weight for vertex coordinate reconstruction loss
    LAMBDA_CLASS = 1.0 # Weight for edge type classification loss

    # --- NEW: DFT Hyperparameter ---
    DFT_GAMMA = 2.0 # (gamma >= 1), 2.0 is a good starting point


# --- 2. Component 1: OrigamiDatasetLoader ---
class OrigamiDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform)
        # Fix: Use weights_only=False for torch_geometric data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # Expects .fold files to be in data/raw
        raw_dir = os.path.join(self.root, 'raw')
        if not os.path.exists(raw_dir):
            return []
        return [f for f in os.listdir(raw_dir) if f.endswith('.fold')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # No download needed, files are provided locally
        pass

    def process(self):
        data_list = []
        # This mapping should cover all edge types in the dataset
        edge_type_mapping = {"B": 0, "M": 1, "V": 2, "F": 3, "U": 4}
        
        for filename in self.raw_file_names:
            path = os.path.join(self.raw_dir, filename)
            
            with open(path, 'r') as f:
                fold_data = json.load(f)

            # Extract data first before debug prints
            vertices_coords = fold_data['vertices_coords']
            edges_vertices = fold_data['edges_vertices']
            edges_assignment = fold_data['edges_assignment']

            # 1. Node features (x): vertex coordinates
            if isinstance(vertices_coords[0], list):
                # Already in [[x1, y1], [x2, y2], ...] format
                x = torch.tensor(vertices_coords, dtype=torch.float)
            else:
                # Flat list format [x1, y1, x2, y2, ...]
                num_vertices = len(vertices_coords) // 2
                x = torch.tensor(vertices_coords, dtype=torch.float).view(num_vertices, 2)
            
            num_nodes = x.size(0)
            print(f"Processing {filename}: {num_nodes} nodes")
            
            # Debug: Check data structure
            print(f"  - vertices_coords type: {type(vertices_coords)}, length: {len(vertices_coords)}")
            if len(vertices_coords) > 0:
                print(f"  - first vertex: {vertices_coords[0]}")
            print(f"  - edges_vertices length: {len(edges_vertices)}")
            if len(edges_vertices) > 0:
                print(f"  - first edge: {edges_vertices[0]}, type: {type(edges_vertices[0])}")
            print(f"  - edges_assignment length: {len(edges_assignment)}")
            if len(edges_assignment) > 0:
                print(f"  - first assignment: {edges_assignment[0]}")

            # 2. Edge connectivity (edge_index) and edge attributes (edge_attr)
            source_nodes = []
            target_nodes = []
            edge_attrs = []

            for i, edge in enumerate(edges_vertices):
                # .fold is 0-based, don't have to convert to 0-based
                u, v = edge[0] , edge[1] 
                
                # Validate indices
                if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                    print(f"Warning: Invalid edge in {filename}: {edge} -> ({u}, {v}), skipping...")
                    continue
                
                # Add edges for undirected graph
                source_nodes.extend([u, v])
                target_nodes.extend([v, u])

                assignment = edges_assignment[i]
                edge_type = edge_type_mapping.get(assignment, 4) # Default to 'U'
                
                # Add same attribute for both directions
                edge_attrs.extend([edge_type, edge_type])

            if len(source_nodes) == 0:
                print(f"Warning: No valid edges found in {filename}, skipping this file...")
                continue

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            # Edge attributes are the labels for classification, must be Long
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            
            print(f"  - Created graph with {edge_index.size(1)} edges")
            
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(graph_data)

        if len(data_list) == 0:
            raise ValueError("No valid graphs found in the dataset!")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _download(self):
        # Override to handle force_reload
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        super()._download()

    def _process(self):
        # Override to handle force_reload
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        super()._process()

# --- 3. Component 2: GNNAutoencoder Model ---
class GNNAutoencoder(torch.nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, latent_dim, num_edge_classes, heads=4):
        super().__init__()
        
        # Encoder
        self.conv1 = GATConv(node_feat_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, latent_dim, heads=1)

        # Decoder
        self.decode_nodes = torch.nn.Linear(latent_dim, node_feat_dim)
        
        self.decode_edges = torch.nn.Sequential(
            torch.nn.Linear(latent_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_edge_classes)
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z, edge_index):
        recon_x = self.decode_nodes(z)
        
        edge_z = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        recon_edge_logits = self.decode_edges(edge_z)
        
        return recon_x, recon_edge_logits

    def forward(self, data):
        z = self.encode(data.x, data.edge_index)
        recon_x, recon_edge_logits = self.decode(z, data.edge_index)
        return recon_x, recon_edge_logits

# --- 4. Component 3 & 4: Training and Evaluation Pipeline (Updated for DFT) ---

def calculate_dft_loss(recon_x, recon_edge_logits, data, config):
    """Calculates the combined loss using the DFT weighting scheme."""
    
    # --- DFT Weight Calculation ---
    # 1. Get probabilities from logits
    edge_probs = F.softmax(recon_edge_logits, dim=-1)
    
    # 2. Find the max probability for each edge prediction
    p_max, _ = edge_probs.max(dim=-1)
    
    # 3. Calculate DFT weights: w = (p_max)^gamma
    # We detach p_max so that gradients don't flow through the weight calculation
    weights = p_max.detach().pow(config.DFT_GAMMA).clamp(min=0.01, max=1.0)

    # --- Loss Calculation ---
    # 1. Reconstruction Loss (Regression)
    loss_recon = F.mse_loss(recon_x, data.x)

    # 2. Weighted Classification Loss (Classification)
    # Use reduction='none' to get per-element loss, then apply weights and average
    unweighted_class_loss = F.cross_entropy(recon_edge_logits, data.edge_attr, reduction='none')
    loss_class = (unweighted_class_loss * weights).mean()
    
    # Combine losses with lambdas
    # We can also weight the reconstruction loss by the average confidence
    # For simplicity, let's start by only weighting the classification loss,
    # as it's the source of instability.
    total_loss = config.LAMBDA_RECON * loss_recon + config.LAMBDA_CLASS * loss_class
    
    return total_loss, loss_recon, loss_class


def train_one_epoch(model, loader, optimizer, config):
    model.train()
    total_loss, total_recon_loss, total_class_loss = 0, 0, 0
    
    for data in loader:
        optimizer.zero_grad()
        
        recon_x, recon_edge_logits = model(data)
        
        # Use the new DFT loss function
        loss, loss_recon, loss_class = calculate_dft_loss(recon_x, recon_edge_logits, data, config)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_recon_loss += loss_recon.item() * data.num_graphs
        total_class_loss += loss_class.item() * data.num_graphs
        
    num_total_graphs = len(loader.dataset)
    return total_loss / num_total_graphs, total_recon_loss / num_total_graphs, total_class_loss / num_total_graphs

@torch.no_grad()
def evaluate(model, loader, config):
    model.eval()
    total_loss, total_recon_loss, total_class_loss = 0, 0, 0
    correct_edges = 0
    total_edges = 0

    for data in loader:
        recon_x, recon_edge_logits = model(data)
        
        # Use the new DFT loss function here as well for consistent reporting
        loss, loss_recon, loss_class = calculate_dft_loss(recon_x, recon_edge_logits, data, config)
        
        total_loss += loss.item() * data.num_graphs
        total_recon_loss += loss_recon.item() * data.num_graphs
        total_class_loss += loss_class.item() * data.num_graphs
        
        pred = recon_edge_logits.argmax(dim=-1)
        correct_edges += (pred == data.edge_attr).sum().item()
        total_edges += data.edge_attr.size(0)

    num_total_graphs = len(loader.dataset)
    edge_accuracy = correct_edges / total_edges
    return total_loss / num_total_graphs, total_recon_loss / num_total_graphs, total_class_loss / num_total_graphs, edge_accuracy



def main():
    print("--- Starting Baseline GNN Autoencoder Training ---")
    config = Config()
    
    # Setup dataset and dataloader
    # For PoC, use the same dataset for training and validation
    # A proper implementation would split this.
    print(f"Loading dataset from {config.DATA_DIR}...")
    # Ensure fresh processing by removing processed data first
    processed_path = os.path.join(config.DATA_DIR, 'processed', 'data.pt')
    if os.path.exists(processed_path):
        print("Removing existing processed data to ensure fresh processing...")
        os.remove(processed_path)
    
    dataset = OrigamiDataset(root=config.DATA_DIR, force_reload=True)
    
    # Check if dataset is empty
    if len(dataset) == 0:
        print("Error: No data found in dataset. Please check that .fold files exist in data/raw/ directory.")
        return
    
    # Simple 80/20 split for train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Handle case where dataset is too small
    if train_size == 0:
        train_size = 1
        val_size = len(dataset) - 1
    if val_size == 0:
        val_size = 1
        train_size = len(dataset) - 1
        
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"Dataset loaded: {len(dataset)} graphs ({train_size} train, {val_size} val).")

    # Setup model and optimizer
    # Assuming the number of edge classes is the max value in the mapping + 1
    num_edge_classes = 5 
    model = GNNAutoencoder(
        node_feat_dim=dataset.num_node_features,
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM,
        num_edge_classes=num_edge_classes,
        heads=config.GAT_HEADS
    )
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    print(f"Model initialized:\n{model}")
    print(f"Starting training for {config.EPOCHS} epochs...")

    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_recon, train_class = train_one_epoch(model, train_loader, optimizer, config)
        val_loss, val_recon, val_class, val_acc = evaluate(model, val_loader, config)
        
        print(f'Epoch: {epoch:03d}, '
              f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, Class: {train_class:.4f}), '
              f'Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, Class: {val_class:.4f}), '
              f'Val Edge Acc: {val_acc:.4f}')

    print("--- Training Finished ---")

if __name__ == '__main__':
    main()
