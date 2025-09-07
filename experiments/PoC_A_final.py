import os
import json
import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.model_selection import KFold

# --- 1. Configuration ---
class Config:
    # Data parameters
    DATA_DIR = "data"
    
    # Model hyperparameters
    HIDDEN_DIM = 64
    LATENT_DIM = 32
    GAT_HEADS = 4
    
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 8
    LEARNING_RATE = 0.005
    
    # Loss weights
    LAMBDA_RECON = 0.01
    LAMBDA_CLASS = 1.0
    
    # K-Fold Cross-Validation parameters
    K_SPLITS = 5
    RANDOM_STATE = 42

    # DFT (Dynamic Focal Training) parameters
    USE_DFT = True
    DFT_GAMMA = 2.0  # Focusing parameter (>= 1.0)
    DFT_ALPHA = 0.25  # Balancing factor for rare classes

# --- 2. OrigamiDataset ---
class OrigamiDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, force_reload=False):
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform)
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
            
            with open(path, 'r') as f:
                fold_data = json.load(f)

            vertices_coords = fold_data['vertices_coords']
            edges_vertices = fold_data['edges_vertices']
            edges_assignment = fold_data['edges_assignment']

            if isinstance(vertices_coords[0], list):
                x = torch.tensor(vertices_coords, dtype=torch.float)
            else:
                num_vertices = len(vertices_coords) // 2
                x = torch.tensor(vertices_coords, dtype=torch.float).view(num_vertices, 2)
            
            num_nodes = x.size(0)
            
            source_nodes, target_nodes, edge_attrs = [], [], []
            for i, edge in enumerate(edges_vertices):
                u, v = edge[0], edge[1]
                if u < 0 or u >= num_nodes or v < 0 or v >= num_nodes:
                    continue
                source_nodes.extend([u, v])
                target_nodes.extend([v, u])
                assignment = edges_assignment[i]
                edge_type = edge_type_mapping.get(assignment, 4)
                edge_attrs.extend([edge_type, edge_type])

            if len(source_nodes) == 0:
                continue

            edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(graph_data)

        if len(data_list) == 0:
            raise ValueError("No valid graphs found in the dataset!")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
    def _download(self):
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        super()._download()

    def _process(self):
        if self.force_reload and os.path.exists(self.processed_paths[0]):
            os.remove(self.processed_paths[0])
        super()._process()

# --- 3. GNNAutoencoder Model ---
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

# --- 4. DFT Loss Function ---
def compute_dft_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Dynamic Focal Training (DFT) Loss implementation
    
    Args:
        logits: Raw prediction logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balancing factor for class imbalance
    
    Returns:
        DFT loss value
    """
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Get probabilities for true classes
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
    p_t = (probs * targets_one_hot).sum(dim=-1)  # Probability of true class
    
    # Compute focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma
    
    # Compute alpha weight (optional, for class imbalance)
    alpha_weight = alpha * targets_one_hot + (1 - alpha) * (1 - targets_one_hot)
    alpha_t = (alpha_weight).sum(dim=-1)
    
    # Compute cross entropy
    log_probs = F.log_softmax(logits, dim=-1)
    ce_loss = -(targets_one_hot * log_probs).sum(dim=-1)
    
    # Final DFT loss
    dft_loss = alpha_t * focal_weight * ce_loss
    
    return dft_loss.mean()

# --- 5. Training Pipeline ---
def train_one_epoch(model, loader, optimizer, config):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_class_loss = 0
    
    for data in loader:
        optimizer.zero_grad()
        recon_x, recon_edge_logits = model(data)
        
        # Reconstruction loss
        loss_recon = F.mse_loss(recon_x, data.x)
        
        # Classification loss
        if config.USE_DFT:
            # Use DFT loss for better handling of hard examples
            loss_class = compute_dft_loss(
                recon_edge_logits, 
                data.edge_attr, 
                gamma=config.DFT_GAMMA, 
                alpha=config.DFT_ALPHA
            )
        else:
            # Standard cross-entropy loss
            loss_class = F.cross_entropy(recon_edge_logits, data.edge_attr)
        
        # Combined loss
        loss = config.LAMBDA_RECON * loss_recon + config.LAMBDA_CLASS * loss_class
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_recon_loss += loss_recon.item() * data.num_graphs
        total_class_loss += loss_class.item() * data.num_graphs
    
    num_samples = len(loader.dataset)
    return (total_loss / num_samples, 
            total_recon_loss / num_samples, 
            total_class_loss / num_samples)

@torch.no_grad()
def evaluate(model, loader, config):
    model.eval()
    total_loss, total_recon_loss, total_class_loss = 0, 0, 0
    correct_edges, total_edges = 0, 0
    
    for data in loader:
        recon_x, recon_edge_logits = model(data)
        
        # Reconstruction loss
        loss_recon = F.mse_loss(recon_x, data.x)
        
        # Classification loss (use same method as training)
        if config.USE_DFT:
            loss_class = compute_dft_loss(
                recon_edge_logits, 
                data.edge_attr, 
                gamma=config.DFT_GAMMA, 
                alpha=config.DFT_ALPHA
            )
        else:
            loss_class = F.cross_entropy(recon_edge_logits, data.edge_attr)
        
        # Combined loss
        loss = config.LAMBDA_RECON * loss_recon + config.LAMBDA_CLASS * loss_class
        
        total_loss += loss.item() * data.num_graphs
        total_recon_loss += loss_recon.item() * data.num_graphs
        total_class_loss += loss_class.item() * data.num_graphs
        
        # Accuracy calculation
        pred = recon_edge_logits.argmax(dim=-1)
        correct_edges += (pred == data.edge_attr).sum().item()
        total_edges += data.edge_attr.size(0)
    
    num_samples = len(loader.dataset)
    edge_accuracy = correct_edges / total_edges if total_edges > 0 else 0
    
    return (total_loss / num_samples, 
            total_recon_loss / num_samples, 
            total_class_loss / num_samples, 
            edge_accuracy)

# --- 6. Main Execution ---
def main():
    print("--- Starting GNN Autoencoder Training with DFT and K-Fold Cross-Validation ---")
    config = Config()
    
    # Load dataset
    print(f"Loading dataset from {config.DATA_DIR}...")
    dataset = OrigamiDataset(root=config.DATA_DIR, force_reload=False)
    
    if len(dataset) == 0:
        print("Error: No data found. Ensure .fold files are in data/raw/")
        return
    
    print(f"Dataset loaded successfully with {len(dataset)} graphs.")
    print(f"DFT Loss: {'Enabled' if config.USE_DFT else 'Disabled'}")
    if config.USE_DFT:
        print(f"  - Gamma (focusing): {config.DFT_GAMMA}")
        print(f"  - Alpha (balancing): {config.DFT_ALPHA}")

    # K-Fold Cross-Validation setup
    kf = KFold(n_splits=config.K_SPLITS, shuffle=True, random_state=config.RANDOM_STATE)
    fold_results = []
    
    print(f"\n--- Starting {config.K_SPLITS}-Fold Cross-Validation ---")
    
    # K-Fold loop
    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f"\n--- Processing Fold {fold+1}/{config.K_SPLITS} ---")
        
        # Create data loaders for this fold
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        print(f"Train size: {len(train_subset)}, Validation size: {len(val_subset)}")
        
        # Initialize fresh model and optimizer for each fold
        num_edge_classes = 5
        model = GNNAutoencoder(
            node_feat_dim=dataset.num_node_features,
            hidden_dim=config.HIDDEN_DIM,
            latent_dim=config.LATENT_DIM,
            num_edge_classes=num_edge_classes,
            heads=config.GAT_HEADS
        )
        optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        # Training loop for this fold
        best_val_acc = 0
        patience_counter = 0
        patience_limit = 20
        
        for epoch in tqdm(range(1, config.EPOCHS + 1), desc=f"Fold {fold+1} Training"):
            # Train for one epoch
            train_metrics = train_one_epoch(model, train_loader, optimizer, config)
            
            # Evaluate every 10 epochs or at the end
            if epoch % 10 == 0 or epoch == config.EPOCHS:
                val_metrics = evaluate(model, val_loader, config)
                val_loss, val_recon_loss, val_class_loss, val_acc = val_metrics
                
                # Simple early stopping based on accuracy
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Optional: early stopping
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Final evaluation for this fold
        final_metrics = evaluate(model, val_loader, config)
        final_val_loss, final_recon_loss, final_class_loss, final_val_acc = final_metrics
        
        print(f"Fold {fold+1} Final Results:")
        print(f"  Val Loss: {final_val_loss:.4f}")
        print(f"  Val Recon Loss: {final_recon_loss:.4f}")
        print(f"  Val Class Loss: {final_class_loss:.4f}")
        print(f"  Val Edge Accuracy: {final_val_acc:.4f}")
        
        fold_results.append(final_metrics)

    # Aggregate results across all folds
    print(f"\n--- Cross-Validation Results Summary ---")
    
    val_losses = np.array([res[0] for res in fold_results])
    val_recon_losses = np.array([res[1] for res in fold_results])
    val_class_losses = np.array([res[2] for res in fold_results])
    val_accuracies = np.array([res[3] for res in fold_results])

    print(f"\nFinal Performance Report (Mean Â± Std across {config.K_SPLITS} folds):")
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10}")
    print("-" * 40)
    print(f"{'Val Loss':<20} {np.mean(val_losses):<10.4f} {np.std(val_losses):<10.4f}")
    print(f"{'Val Recon Loss':<20} {np.mean(val_recon_losses):<10.4f} {np.std(val_recon_losses):<10.4f}")
    print(f"{'Val Class Loss':<20} {np.mean(val_class_losses):<10.4f} {np.std(val_class_losses):<10.4f}")
    print(f"{'Val Edge Accuracy':<20} {np.mean(val_accuracies):<10.4f} {np.std(val_accuracies):<10.4f}")
    print("-" * 40)
    
    # Additional statistics
    print(f"\nDetailed Statistics:")
    print(f"Best single fold accuracy: {np.max(val_accuracies):.4f}")
    print(f"Worst single fold accuracy: {np.min(val_accuracies):.4f}")
    print(f"Coefficient of variation (accuracy): {np.std(val_accuracies)/np.mean(val_accuracies)*100:.2f}%")
    
    print("\n--- Experiment Complete ---")
    print("This provides a statistically reliable baseline for future model improvements.")

if __name__ == '__main__':
    main()
