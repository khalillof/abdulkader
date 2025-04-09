import pandas as pd
import numpy as np
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start timing the script execution
start_time = time.time()

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Load a small sample of the dataset instead of the entire file
def load_data(filepath, sample_size=1000):
    if os.path.exists(filepath):
        return pd.read_csv(filepath, nrows=sample_size)  # Load only first 1000 rows for efficiency
    else:
        logging.warning(f"File {filepath} not found! Returning empty DataFrame.")
        return pd.DataFrame()

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm.notebook import tqdm
import networkx as nx
import json
from IPython.display import display, HTML

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    
# PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Load the preprocessed data
# In a real scenario, you would load the saved data from previous notebooks
# Here we'll simulate loading the data

def load_preprocessed_data(data_path='./data'):
    """
    Load preprocessed data for the Kaggle hierarchical text classification dataset.
    
    Args:
        data_path: Path to the preprocessed data
        
    Returns:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        class_hierarchy: Dictionary representing the class hierarchy
    """
    # In a real scenario, you would load actual saved data
    # For this notebook, we'll create synthetic data that matches the structure
    
    # Simulate class hierarchy with 6 level-1 classes, ~11 level-2 classes per level-1, and ~8 level-3 classes per level-2
    level1_classes = ['Technology', 'Science', 'Business', 'Entertainment', 'Health', 'Politics']
    
    class_hierarchy = {}
    for l1 in level1_classes:
        class_hierarchy[l1] = {}
        for i in range(1, 12):  # ~11 level-2 classes per level-1
            l2 = f"{l1}_{i}"
            class_hierarchy[l1][l2] = []
            for j in range(1, 9):  # ~8 level-3 classes per level-2
                l3 = f"{l2}_{j}"
                class_hierarchy[l1][l2].append(l3)
    
    # Create mappings from class names to indices
    level1_to_idx = {cls: i for i, cls in enumerate(level1_classes)}
    
    level2_classes = []
    for l1 in level1_classes:
        level2_classes.extend(list(class_hierarchy[l1].keys()))
    level2_to_idx = {cls: i for i, cls in enumerate(level2_classes)}
    
    level3_classes = []
    for l1 in level1_classes:
        for l2 in class_hierarchy[l1]:
            level3_classes.extend(class_hierarchy[l1][l2])
    level3_to_idx = {cls: i for i, cls in enumerate(level3_classes)}
    
    # Create synthetic graph data
    def create_synthetic_graph_data(num_samples, level1_to_idx, level2_to_idx, level3_to_idx, class_hierarchy):
        graph_data_list = []
        
        for _ in range(num_samples):
            # Randomly select classes from each level
            l1 = np.random.choice(level1_classes)
            l2 = np.random.choice(list(class_hierarchy[l1].keys()))
            l3 = np.random.choice(class_hierarchy[l1][l2])
            
            # Create node features (simulate text embeddings)
            num_nodes = np.random.randint(10, 30)  # Random number of nodes (words)
            node_features = torch.randn(num_nodes, 300)  # 300-dim word embeddings
            
            # Create edges (connections between words)
            edge_index = []
            for i in range(num_nodes):
                # Connect each node to a few random nodes
                connections = np.random.choice(num_nodes, size=min(5, num_nodes), replace=False)
                for j in connections:
                    if i != j:  # Avoid self-loops
                        edge_index.append([i, j])
            
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            # Create labels
            y_level1 = torch.tensor([level1_to_idx[l1]], dtype=torch.long)
            y_level2 = torch.tensor([level2_to_idx[l2]], dtype=torch.long)
            y_level3 = torch.tensor([level3_to_idx[l3]], dtype=torch.long)
            
            # Create PyTorch Geometric Data object
            data = Data(x=node_features, edge_index=edge_index, 
                        y_level1=y_level1, y_level2=y_level2, y_level3=y_level3,
                        num_nodes=num_nodes)
            
            graph_data_list.append(data)
        
        return graph_data_list
    
    # Create train, validation, and test datasets
    train_data = create_synthetic_graph_data(800, level1_to_idx, level2_to_idx, level3_to_idx, class_hierarchy)
    val_data = create_synthetic_graph_data(100, level1_to_idx, level2_to_idx, level3_to_idx, class_hierarchy)
    test_data = create_synthetic_graph_data(200, level1_to_idx, level2_to_idx, level3_to_idx, class_hierarchy)
    
    class_info = {
        'level1_to_idx': level1_to_idx,
        'level2_to_idx': level2_to_idx,
        'level3_to_idx': level3_to_idx,
        'idx_to_level1': {v: k for k, v in level1_to_idx.items()},
        'idx_to_level2': {v: k for k, v in level2_to_idx.items()},
        'idx_to_level3': {v: k for k, v in level3_to_idx.items()},
        'num_level1_classes': len(level1_to_idx),
        'num_level2_classes': len(level2_to_idx),
        'num_level3_classes': len(level3_to_idx)
    }
    
    return train_data, val_data, test_data, class_hierarchy, class_info

# Load the data
train_data, val_data, test_data, class_hierarchy, class_info = load_preprocessed_data()

print(f"Number of training samples: {len(train_data)}")
print(f"Number of validation samples: {len(val_data)}")
print(f"Number of test samples: {len(test_data)}")
print(f"Number of level-1 classes: {class_info['num_level1_classes']}")
print(f"Number of level-2 classes: {class_info['num_level2_classes']}")
print(f"Number of level-3 classes: {class_info['num_level3_classes']}")

# Import the GNN models from the previous notebook
# In a real scenario, you would import the actual model classes
# Here we'll redefine the models for completeness

from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

class HierarchicalGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_level1_classes, num_level2_classes, num_level3_classes):
        super(HierarchicalGCN, self).__init__()
        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Hierarchical classification layers
        self.level1_classifier = nn.Linear(hidden_dim, num_level1_classes)
        self.level2_classifier = nn.Linear(hidden_dim + num_level1_classes, num_level2_classes)
        self.level3_classifier = nn.Linear(hidden_dim + num_level2_classes, num_level3_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch):
        # Graph convolution layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level representations
        x = global_mean_pool(x, batch)
        
        # Level 1 classification
        level1_logits = self.level1_classifier(x)
        level1_probs = F.softmax(level1_logits, dim=1)
        
        # Level 2 classification (using level 1 predictions)
        level2_input = torch.cat([x, level1_probs], dim=1)
        level2_logits = self.level2_classifier(level2_input)
        level2_probs = F.softmax(level2_logits, dim=1)
        
        # Level 3 classification (using level 2 predictions)
        level3_input = torch.cat([x, level2_probs], dim=1)
        level3_logits = self.level3_classifier(level3_input)
        
        return level1_logits, level2_logits, level3_logits

class HierarchicalGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_level1_classes, num_level2_classes, num_level3_classes, heads=4):
        super(HierarchicalGAT, self).__init__()
        
        # Graph attention layers
        self.conv1 = GATConv(input_dim, hidden_dim // heads, heads=heads)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1)
        
        # Hierarchical classification layers
        self.level1_classifier = nn.Linear(hidden_dim, num_level1_classes)
        self.level2_classifier = nn.Linear(hidden_dim + num_level1_classes, num_level2_classes)
        self.level3_classifier = nn.Linear(hidden_dim + num_level2_classes, num_level3_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, edge_index, batch):
        # Graph attention layers
        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        
        # Global pooling to get graph-level representations
        x = global_max_pool(x, batch)
        
        # Level 1 classification
        level1_logits = self.level1_classifier(x)
        level1_probs = F.softmax(level1_logits, dim=1)
        
        # Level 2 classification (using level 1 predictions)
        level2_input = torch.cat([x, level1_probs], dim=1)
        level2_logits = self.level2_classifier(level2_input)
        level2_probs = F.softmax(level2_logits, dim=1)
        
        # Level 3 classification (using level 2 predictions)
        level3_input = torch.cat([x, level2_probs], dim=1)
        level3_logits = self.level3_classifier(level3_input)
        
        return level1_logits, level2_logits, level3_logits

# Initialize the models
input_dim = 300  # Dimension of word embeddings
hidden_dim = 256

gcn_model = HierarchicalGCN(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_level1_classes=class_info['num_level1_classes'],
    num_level2_classes=class_info['num_level2_classes'],
    num_level3_classes=class_info['num_level3_classes']
).to(device)

gat_model = HierarchicalGAT(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_level1_classes=class_info['num_level1_classes'],
    num_level2_classes=class_info['num_level2_classes'],
    num_level3_classes=class_info['num_level3_classes'],
    heads=4
).to(device)

print(f"GCN model parameters: {sum(p.numel() for p in gcn_model.parameters() if p.requires_grad):,}")
print(f"GAT model parameters: {sum(p.numel() for p in gat_model.parameters() if p.requires_grad):,}")

def train_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The GNN model
        loader: DataLoader for the training data
        optimizer: Optimizer for training
        device: Device to use for training
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        level1_logits, level2_logits, level3_logits = model(batch.x, batch.edge_index, batch.batch)
        
        # Calculate loss for each level
        loss_level1 = F.cross_entropy(level1_logits, batch.y_level1)
        loss_level2 = F.cross_entropy(level2_logits, batch.y_level2)
        loss_level3 = F.cross_entropy(level3_logits, batch.y_level3)
        
        # Combine losses with weights
        loss = 0.2 * loss_level1 + 0.3 * loss_level2 + 0.5 * loss_level3
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, device):
    """
    Evaluate the model on the given data.
    
    Args:
        model: The GNN model
        loader: DataLoader for the evaluation data
        device: Device to use for evaluation
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    y_true_level1 = []
    y_pred_level1 = []
    y_true_level2 = []
    y_pred_level2 = []
    y_true_level3 = []
    y_pred_level3 = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            level1_logits, level2_logits, level3_logits = model(batch.x, batch.edge_index, batch.batch)
            
            # Get predictions
            _, level1_preds = torch.max(level1_logits, dim=1)
            _, level2_preds = torch.max(level2_logits, dim=1)
            _, level3_preds = torch.max(level3_logits, dim=1)
            
            # Collect true labels and predictions
            y_true_level1.extend(batch.y_level1.cpu().numpy())
            y_pred_level1.extend(level1_preds.cpu().numpy())
            y_true_level2.extend(batch.y_level2.cpu().numpy())
            y_pred_level2.extend(level2_preds.cpu().numpy())
            y_true_level3.extend(batch.y_level3.cpu().numpy())
            y_pred_level3.extend(level3_preds.cpu().numpy())
    
    # Calculate metrics for each level
    metrics = {}
    
    # Level 1 metrics
    metrics['level1_accuracy'] = accuracy_score(y_true_level1, y_pred_level1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_level1, y_pred_level1, average='macro')
    metrics['level1_precision'] = precision
    metrics['level1_recall'] = recall
    metrics['level1_f1'] = f1
    
    # Level 2 metrics
    metrics['level2_accuracy'] = accuracy_score(y_true_level2, y_pred_level2)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_level2, y_pred_level2, average='macro')
    metrics['level2_precision'] = precision
    metrics['level2_recall'] = recall
    metrics['level2_f1'] = f1
    
    # Level 3 metrics
    metrics['level3_accuracy'] = accuracy_score(y_true_level3, y_pred_level3)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_level3, y_pred_level3, average='macro')
    metrics['level3_precision'] = precision
    metrics['level3_recall'] = recall
    metrics['level3_f1'] = f1
    
    # Calculate hierarchical accuracy (all levels correct)
    correct_all_levels = sum(1 for i in range(len(y_true_level1)) 
                            if y_true_level1[i] == y_pred_level1[i] 
                            and y_true_level2[i] == y_pred_level2[i] 
                            and y_true_level3[i] == y_pred_level3[i])
    metrics['hierarchical_accuracy'] = correct_all_levels / len(y_true_level1)
    
    return metrics, (y_true_level1, y_pred_level1, y_true_level2, y_pred_level2, y_true_level3, y_pred_level3)

# Create data loaders
train_loader = PyGDataLoader(train_data, batch_size=32, shuffle=True)
val_loader = PyGDataLoader(val_data, batch_size=32, shuffle=False)
test_loader = PyGDataLoader(test_data, batch_size=32, shuffle=False)

# Training parameters
num_epochs = 30
lr = 0.001
weight_decay = 5e-4

# Initialize optimizers
gcn_optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr, weight_decay=weight_decay)
gat_optimizer = torch.optim.Adam(gat_model.parameters(), lr=lr, weight_decay=weight_decay)

# Learning rate schedulers
gcn_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gcn_optimizer, mode='min', factor=0.5, patience=5, verbose=True)
gat_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(gat_optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Training history
gcn_history = {'train_loss': [], 'val_metrics': []}
gat_history = {'train_loss': [], 'val_metrics': []}

# Train GCN model
print("Training GCN model...")
best_val_f1 = 0
best_gcn_state = None

for epoch in range(num_epochs):
    # Train for one epoch
    train_loss = train_epoch(gcn_model, train_loader, gcn_optimizer, device)
    gcn_history['train_loss'].append(train_loss)
    
    # Evaluate on validation set
    val_metrics, _ = evaluate(gcn_model, val_loader, device)
    gcn_history['val_metrics'].append(val_metrics)
    
    # Update learning rate
    gcn_scheduler.step(train_loss)
    
    # Save best model
    if val_metrics['level3_f1'] > best_val_f1:
        best_val_f1 = val_metrics['level3_f1']
        best_gcn_state = gcn_model.state_dict()
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
          f"Val L1 Acc: {val_metrics['level1_accuracy']:.4f}, "
          f"Val L2 Acc: {val_metrics['level2_accuracy']:.4f}, "
          f"Val L3 Acc: {val_metrics['level3_accuracy']:.4f}, "
          f"Val Hier Acc: {val_metrics['hierarchical_accuracy']:.4f}")

# Load best model
gcn_model.load_state_dict(best_gcn_state)

# Train GAT model
print("\nTraining GAT model...")
best_val_f1 = 0
best_gat_state = None

for epoch in range(num_epochs):
    # Train for one epoch
    train_loss = train_epoch(gat_model, train_loader, gat_optimizer, device)
    gat_history['train_loss'].append(train_loss)
    
    # Evaluate on validation set
    val_metrics, _ = evaluate(gat_model, val_loader, device)
    gat_history['val_metrics'].append(val_metrics)
    
    # Update learning rate
    gat_scheduler.step(train_loss)
    
    # Save best model
    if val_metrics['level3_f1'] > best_val_f1:
        best_val_f1 = val_metrics['level3_f1']
        best_gat_state = gat_model.state_dict()
    
    # Print progress
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
          f"Val L1 Acc: {val_metrics['level1_accuracy']:.4f}, "
          f"Val L2 Acc: {val_metrics['level2_accuracy']:.4f}, "
          f"Val L3 Acc: {val_metrics['level3_accuracy']:.4f}, "
          f"Val Hier Acc: {val_metrics['hierarchical_accuracy']:.4f}")

# Load best model
gat_model.load_state_dict(best_gat_state)

# Evaluate GCN model on test set
print("Evaluating GCN model on test set...")
gcn_test_metrics, gcn_test_preds = evaluate(gcn_model, test_loader, device)

# Evaluate GAT model on test set
print("Evaluating GAT model on test set...")
gat_test_metrics, gat_test_preds = evaluate(gat_model, test_loader, device)

# Print test metrics
print("\nGCN Test Metrics:")
print(f"Level 1 Accuracy: {gcn_test_metrics['level1_accuracy']:.4f}")
print(f"Level 2 Accuracy: {gcn_test_metrics['level2_accuracy']:.4f}")
print(f"Level 3 Accuracy: {gcn_test_metrics['level3_accuracy']:.4f}")
print(f"Hierarchical Accuracy: {gcn_test_metrics['hierarchical_accuracy']:.4f}")
print(f"Level 1 F1 Score: {gcn_test_metrics['level1_f1']:.4f}")
print(f"Level 2 F1 Score: {gcn_test_metrics['level2_f1']:.4f}")
print(f"Level 3 F1 Score: {gcn_test_metrics['level3_f1']:.4f}")

print("\nGAT Test Metrics:")
print(f"Level 1 Accuracy: {gat_test_metrics['level1_accuracy']:.4f}")
print(f"Level 2 Accuracy: {gat_test_metrics['level2_accuracy']:.4f}")
print(f"Level 3 Accuracy: {gat_test_metrics['level3_accuracy']:.4f}")
print(f"Hierarchical Accuracy: {gat_test_metrics['hierarchical_accuracy']:.4f}")
print(f"Level 1 F1 Score: {gat_test_metrics['level1_f1']:.4f}")
print(f"Level 2 F1 Score: {gat_test_metrics['level2_f1']:.4f}")
print(f"Level 3 F1 Score: {gat_test_metrics['level3_f1']:.4f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(gcn_history['train_loss'], label='GCN')
plt.plot(gat_history['train_loss'], label='GAT')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot validation metrics
metrics_to_plot = ['level1_accuracy', 'level2_accuracy', 'level3_accuracy', 'hierarchical_accuracy']
titles = ['Level 1 Accuracy', 'Level 2 Accuracy', 'Level 3 Accuracy', 'Hierarchical Accuracy']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    gcn_values = [metrics[metric] for metrics in gcn_history['val_metrics']]
    gat_values = [metrics[metric] for metrics in gat_history['val_metrics']]
    
    axes[i].plot(gcn_values, label='GCN')
    axes[i].plot(gat_values, label='GAT')
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Accuracy')
    axes[i].set_title(title)
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Extract predictions for level 1
y_true_level1_gcn, y_pred_level1_gcn, _, _, _, _ = gcn_test_preds
y_true_level1_gat, y_pred_level1_gat, _, _, _, _ = gat_test_preds

# Create confusion matrices
cm_gcn = confusion_matrix(y_true_level1_gcn, y_pred_level1_gcn)
cm_gat = confusion_matrix(y_true_level1_gat, y_pred_level1_gat)

# Get class names
class_names = list(class_info['idx_to_level1'].values())

# Plot confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.heatmap(cm_gcn, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('True')
axes[0].set_title('GCN Confusion Matrix (Level 1)')

sns.heatmap(cm_gat, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')
axes[1].set_title('GAT Confusion Matrix (Level 1)')

plt.tight_layout()
plt.show()

# Define a simple MLP baseline
class HierarchicalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_level1_classes, num_level2_classes, num_level3_classes):
        super(HierarchicalMLP, self).__init__()
        
        # MLP layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Hierarchical classification layers
        self.level1_classifier = nn.Linear(hidden_dim, num_level1_classes)
        self.level2_classifier = nn.Linear(hidden_dim + num_level1_classes, num_level2_classes)
        self.level3_classifier = nn.Linear(hidden_dim + num_level2_classes, num_level3_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # MLP layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Level 1 classification
        level1_logits = self.level1_classifier(x)
        level1_probs = F.softmax(level1_logits, dim=1)
        
        # Level 2 classification (using level 1 predictions)
        level2_input = torch.cat([x, level1_probs], dim=1)
        level2_logits = self.level2_classifier(level2_input)
        level2_probs = F.softmax(level2_logits, dim=1)
        
        # Level 3 classification (using level 2 predictions)
        level3_input = torch.cat([x, level2_probs], dim=1)
        level3_logits = self.level3_classifier(level3_input)
        
        return level1_logits, level2_logits, level3_logits

# Function to prepare data for MLP
def prepare_mlp_data(graph_data):
    """
    Convert graph data to flat features for MLP.
    
    Args:
        graph_data: List of PyTorch Geometric Data objects
        
    Returns:
        X: Feature matrix
        y_level1, y_level2, y_level3: Labels for each level
    """
    X = []
    y_level1 = []
    y_level2 = []
    y_level3 = []
    
    for data in graph_data:
        # Average node features to get graph-level representation
        graph_feat = torch.mean(data.x, dim=0).numpy()
        X.append(graph_feat)
        
        # Extract labels
        y_level1.append(data.y_level1.item())
        y_level2.append(data.y_level2.item())
        y_level3.append(data.y_level3.item())
    
    return np.array(X), np.array(y_level1), np.array(y_level2), np.array(y_level3)

# Prepare data for MLP
X_train, y_train_level1, y_train_level2, y_train_level3 = prepare_mlp_data(train_data)
X_val, y_val_level1, y_val_level2, y_val_level3 = prepare_mlp_data(val_data)
X_test, y_test_level1, y_test_level2, y_test_level3 = prepare_mlp_data(test_data)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_level1_tensor = torch.LongTensor(y_train_level1)
y_train_level2_tensor = torch.LongTensor(y_train_level2)
y_train_level3_tensor = torch.LongTensor(y_train_level3)

X_val_tensor = torch.FloatTensor(X_val)
y_val_level1_tensor = torch.LongTensor(y_val_level1)
y_val_level2_tensor = torch.LongTensor(y_val_level2)
y_val_level3_tensor = torch.LongTensor(y_val_level3)

X_test_tensor = torch.FloatTensor(X_test)
y_test_level1_tensor = torch.LongTensor(y_test_level1)
y_test_level2_tensor = torch.LongTensor(y_test_level2)
y_test_level3_tensor = torch.LongTensor(y_test_level3)

# Create datasets and data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_level1_tensor, y_train_level2_tensor, y_train_level3_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_level1_tensor, y_val_level2_tensor, y_val_level3_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_level1_tensor, y_test_level2_tensor, y_test_level3_tensor)

train_loader_mlp = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader_mlp = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader_mlp = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize MLP model
mlp_model = HierarchicalMLP(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    num_level1_classes=class_info['num_level1_classes'],
    num_level2_classes=class_info['num_level2_classes'],
    num_level3_classes=class_info['num_level3_classes']
).to(device)

# Define optimizer
mlp_optimizer = torch.optim.Adam(mlp_model.parameters(), lr=lr, weight_decay=weight_decay)

# Training function for MLP
def train_epoch_mlp(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for X, y_level1, y_level2, y_level3 in loader:
        X, y_level1, y_level2, y_level3 = X.to(device), y_level1.to(device), y_level2.to(device), y_level3.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        level1_logits, level2_logits, level3_logits = model(X)
        
        # Calculate loss for each level
        loss_level1 = F.cross_entropy(level1_logits, y_level1)
        loss_level2 = F.cross_entropy(level2_logits, y_level2)
        loss_level3 = F.cross_entropy(level3_logits, y_level3)
        
        # Combine losses with weights
        loss = 0.2 * loss_level1 + 0.3 * loss_level2 + 0.5 * loss_level3
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    epoch_loss = total_loss / len(loader.dataset)
    return epoch_loss

# Evaluation function for MLP
def evaluate_mlp(model, loader, device):
    model.eval()
    
    y_true_level1 = []
    y_pred_level1 = []
    y_true_level2 = []
    y_pred_level2 = []
    y_true_level3 = []
    y_pred_level3 = []
    
    with torch.no_grad():
        for X, y_level1, y_level2, y_level3 in loader:
            X, y_level1, y_level2, y_level3 = X.to(device), y_level1.to(device), y_level2.to(device), y_level3.to(device)
            
            # Forward pass
            level1_logits, level2_logits, level3_logits = model(X)
            
            # Get predictions
            _, level1_preds = torch.max(level1_logits, dim=1)
            _, level2_preds = torch.max(level2_logits, dim=1)
            _, level3_preds = torch.max(level3_logits, dim=1)
            
            # Collect true labels and predictions
            y_true_level1.extend(y_level1.cpu().numpy())
            y_pred_level1.extend(level1_preds.cpu().numpy())
            y_true_level2.extend(y_level2.cpu().numpy())
            y_pred_level2.extend(level2_preds.cpu().numpy())
            y_true_level3.extend(y_level3.cpu().numpy())
            y_pred_level3.extend(level3_preds.cpu().numpy())
    
    # Calculate metrics for each level
    metrics = {}
    
    # Level 1 metrics
    metrics['level1_accuracy'] = accuracy_score(y_true_level1, y_pred_level1)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_level1, y_pred_level1, average='macro')
    metrics['level1_precision'] = precision
    metrics['level1_recall'] = recall
    metrics['level1_f1'] = f1
    
    # Level 2 metrics
    metrics['level2_accuracy'] = accuracy_score(y_true_level2, y_pred_level2)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_level2, y_pred_level2, average='macro')
    metrics['level2_precision'] = precision
    metrics['level2_recall'] = recall
    metrics['level2_f1'] = f1
    
    # Level 3 metrics
    metrics['level3_accuracy'] = accuracy_score(y_true_level3, y_pred_level3)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_level3, y_pred_level3, average='macro')
    metrics['level3_precision'] = precision
    metrics['level3_recall'] = recall
    metrics['level3_f1'] = f1
    
    # Calculate hierarchical accuracy (all levels correct)
    correct_all_levels = sum(1 for i in range(len(y_true_level1)) 
                            if y_true_level1[i] == y_pred_level1[i] 
                            and y_true_level2[i] == y_pred_level2[i] 
                            and y_true_level3[i] == y_pred_level3[i])
    metrics['hierarchical_accuracy'] = correct_all_levels / len(y_true_level1)
    
    return metrics

# Train MLP model
print("Training MLP model...")
mlp_history = {'train_loss': [], 'val_metrics': []}

for epoch in range(num_epochs):
    # Train for one epoch
    train_loss = train_epoch_mlp(mlp_model, train_loader_mlp, mlp_optimizer, device)
    mlp_history['train_loss'].append(train_loss)
    
    # Evaluate on validation set
    val_metrics = evaluate_mlp(mlp_model, val_loader_mlp, device)
    mlp_history['val_metrics'].append(val_metrics)
    
    # Print progress
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Val L1 Acc: {val_metrics['level1_accuracy']:.4f}, "
              f"Val L2 Acc: {val_metrics['level2_accuracy']:.4f}, "
              f"Val L3 Acc: {val_metrics['level3_accuracy']:.4f}, "
              f"Val Hier Acc: {val_metrics['hierarchical_accuracy']:.4f}")

# Evaluate MLP model on test set
mlp_test_metrics = evaluate_mlp(mlp_model, test_loader_mlp, device)

# Print test metrics
print("\nMLP Test Metrics:")
print(f"Level 1 Accuracy: {mlp_test_metrics['level1_accuracy']:.4f}")
print(f"Level 2 Accuracy: {mlp_test_metrics['level2_accuracy']:.4f}")
print(f"Level 3 Accuracy: {mlp_test_metrics['level3_accuracy']:.4f}")
print(f"Hierarchical Accuracy: {mlp_test_metrics['hierarchical_accuracy']:.4f}")
print(f"Level 1 F1 Score: {mlp_test_metrics['level1_f1']:.4f}")
print(f"Level 2 F1 Score: {mlp_test_metrics['level2_f1']:.4f}")
print(f"Level 3 F1 Score: {mlp_test_metrics['level3_f1']:.4f}")

# Collect test metrics for all models
models = ['MLP', 'GCN', 'GAT']
metrics = ['level1_accuracy', 'level2_accuracy', 'level3_accuracy', 'hierarchical_accuracy',
           'level1_f1', 'level2_f1', 'level3_f1']
metric_names = ['Level 1 Accuracy', 'Level 2 Accuracy', 'Level 3 Accuracy', 'Hierarchical Accuracy',
                'Level 1 F1', 'Level 2 F1', 'Level 3 F1']

# Create a DataFrame to store the results
results = pd.DataFrame(index=metric_names, columns=models)

# Fill in the results
for i, metric in enumerate(metrics):
    results.loc[metric_names[i], 'MLP'] = mlp_test_metrics[metric]
    results.loc[metric_names[i], 'GCN'] = gcn_test_metrics[metric]
    results.loc[metric_names[i], 'GAT'] = gat_test_metrics[metric]

# Display the results
display(results.style.format("{:.4f}").background_gradient(cmap='Blues', axis=1))

# Plot the results
plt.figure(figsize=(12, 8))
ax = results.plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Metric')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Model')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# End timing
end_time = time.time()
logging.info(f"Script execution completed in {end_time - start_time:.2f} seconds.")
