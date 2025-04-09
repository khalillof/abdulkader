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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.manifold import TSNE
import networkx as nx
import json
from IPython.display import display, HTML
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Simulate loading results
def load_results():
    """
    Load or simulate results from model training and evaluation.
    
    Returns:
        Dictionary containing results for different models
    """
    # In a real scenario, you would load actual saved results
    # For this notebook, we'll create synthetic results
    
    # Define models
    models = ['MLP', 'GCN', 'GAT']
    
    # Define metrics
    metrics = ['level1_accuracy', 'level2_accuracy', 'level3_accuracy', 'hierarchical_accuracy',
               'level1_f1', 'level2_f1', 'level3_f1']
    
    # Create a dictionary to store results
    results = {}
    
    # Generate synthetic results for each model
    # We'll make GAT perform best, followed by GCN, then MLP
    base_values = {
        'MLP': {
            'level1_accuracy': 0.78, 'level2_accuracy': 0.65, 'level3_accuracy': 0.52, 'hierarchical_accuracy': 0.45,
            'level1_f1': 0.77, 'level2_f1': 0.64, 'level3_f1': 0.51
        },
        'GCN': {
            'level1_accuracy': 0.85, 'level2_accuracy': 0.72, 'level3_accuracy': 0.61, 'hierarchical_accuracy': 0.53,
            'level1_f1': 0.84, 'level2_f1': 0.71, 'level3_f1': 0.60
        },
        'GAT': {
            'level1_accuracy': 0.89, 'level2_accuracy': 0.78, 'level3_accuracy': 0.67, 'hierarchical_accuracy': 0.58,
            'level1_f1': 0.88, 'level2_f1': 0.77, 'level3_f1': 0.66
        }
    }
    
    # Add some random noise to make it more realistic
    for model in models:
        results[model] = {}
        for metric in metrics:
            # Add small random noise
            noise = np.random.uniform(-0.02, 0.02)
            results[model][metric] = base_values[model][metric] + noise
            # Ensure values are between 0 and 1
            results[model][metric] = max(0, min(1, results[model][metric]))
    
    # Simulate training history
    history = {}
    for model in models:
        history[model] = {
            'train_loss': [],
            'val_metrics': []
        }
        
        # Generate synthetic training loss
        initial_loss = 2.5 if model == 'MLP' else (2.0 if model == 'GCN' else 1.8)
        final_loss = 0.8 if model == 'MLP' else (0.6 if model == 'GCN' else 0.5)
        
        for epoch in range(30):  # 30 epochs
            # Exponential decay with noise
            progress = epoch / 29  # 0 to 1
            loss = initial_loss * np.exp(-3 * progress) + final_loss + np.random.uniform(-0.1, 0.1)
            history[model]['train_loss'].append(loss)
            
            # Generate synthetic validation metrics
            val_metrics = {}
            for metric in metrics:
                # Start from lower value and increase over epochs
                start_val = base_values[model][metric] * 0.7
                end_val = base_values[model][metric]
                val = start_val + (end_val - start_val) * (1 - np.exp(-5 * progress))
                # Add noise
                val += np.random.uniform(-0.03, 0.03)
                # Ensure values are between 0 and 1
                val_metrics[metric] = max(0, min(1, val))
            
            history[model]['val_metrics'].append(val_metrics)
    
    # Simulate confusion matrices
    confusion_matrices = {}
    
    # Define class names
    level1_classes = ['Technology', 'Science', 'Business', 'Entertainment', 'Health', 'Politics']
    
    for model in models:
        confusion_matrices[model] = {}
        
        # Level 1 confusion matrix
        cm_level1 = np.zeros((6, 6), dtype=int)
        
        # Diagonal elements (correct predictions) should be higher
        for i in range(6):
            # Higher values for better models
            diag_factor = 0.7 if model == 'MLP' else (0.8 if model == 'GCN' else 0.85)
            cm_level1[i, i] = int(100 * diag_factor + np.random.randint(-10, 10))
            
            # Off-diagonal elements (incorrect predictions)
            remaining = 100 - cm_level1[i, i]
            for j in range(6):
                if i != j:
                    # Distribute remaining percentage among other classes
                    cm_level1[i, j] = int(remaining / 5 + np.random.randint(-5, 5))
            
            # Ensure row sums to 100
            row_sum = np.sum(cm_level1[i, :])
            if row_sum != 100:
                cm_level1[i, i] += (100 - row_sum)
        
        confusion_matrices[model]['level1'] = cm_level1
    
    # Simulate predictions for error analysis
    predictions = {}
    for model in models:
        # Create 100 synthetic examples
        n_samples = 100
        
        # True labels
        y_true_level1 = np.random.randint(0, 6, n_samples)
        y_true_level2 = np.random.randint(0, 66, n_samples)
        y_true_level3 = np.random.randint(0, 528, n_samples)
        
        # Predicted labels (with different accuracy based on model)
        accuracy_factor = 0.7 if model == 'MLP' else (0.8 if model == 'GCN' else 0.85)
        
        y_pred_level1 = np.copy(y_true_level1)
        y_pred_level2 = np.copy(y_true_level2)
        y_pred_level3 = np.copy(y_true_level3)
        
        # Introduce errors
        for i in range(n_samples):
            if np.random.random() > accuracy_factor:
                # Level 1 error
                y_pred_level1[i] = np.random.randint(0, 6)
            
            if np.random.random() > accuracy_factor:
                # Level 2 error
                y_pred_level2[i] = np.random.randint(0, 66)
            
            if np.random.random() > accuracy_factor:
                # Level 3 error
                y_pred_level3[i] = np.random.randint(0, 528)
        
        predictions[model] = {
            'y_true_level1': y_true_level1,
            'y_pred_level1': y_pred_level1,
            'y_true_level2': y_true_level2,
            'y_pred_level2': y_pred_level2,
            'y_true_level3': y_true_level3,
            'y_pred_level3': y_pred_level3
        }
    
    # Simulate embeddings for visualization
    embeddings = {}
    for model in models:
        # Create synthetic embeddings
        n_samples = 500
        n_features = 128
        
        # Generate random embeddings
        X = np.random.randn(n_samples, n_features)
        
        # Generate labels
        y_level1 = np.random.randint(0, 6, n_samples)
        
        # Make embeddings more clustered by class
        for cls in range(6):
            # Get indices for this class
            idx = np.where(y_level1 == cls)[0]
            
            # Add class-specific offset
            offset = np.random.randn(n_features) * 5
            X[idx] += offset
        
        embeddings[model] = {
            'X': X,
            'y_level1': y_level1
        }
    
    # Create class hierarchy
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
    
    # Create mappings from indices to class names
    idx_to_level1 = {i: cls for i, cls in enumerate(level1_classes)}
    
    level2_classes = []
    for l1 in level1_classes:
        level2_classes.extend(list(class_hierarchy[l1].keys()))
    idx_to_level2 = {i: cls for i, cls in enumerate(level2_classes)}
    
    level3_classes = []
    for l1 in level1_classes:
        for l2 in class_hierarchy[l1]:
            level3_classes.extend(class_hierarchy[l1][l2])
    idx_to_level3 = {i: cls for i, cls in enumerate(level3_classes)}
    
    class_info = {
        'idx_to_level1': idx_to_level1,
        'idx_to_level2': idx_to_level2,
        'idx_to_level3': idx_to_level3,
        'class_hierarchy': class_hierarchy
    }
    
    return results, history, confusion_matrices, predictions, embeddings, class_info

# Load results
results, history, confusion_matrices, predictions, embeddings, class_info = load_results()

# Display basic results
models = ['MLP', 'GCN', 'GAT']
metrics = ['level1_accuracy', 'level2_accuracy', 'level3_accuracy', 'hierarchical_accuracy',
           'level1_f1', 'level2_f1', 'level3_f1']
metric_names = ['Level 1 Accuracy', 'Level 2 Accuracy', 'Level 3 Accuracy', 'Hierarchical Accuracy',
                'Level 1 F1', 'Level 2 F1', 'Level 3 F1']

# Create a DataFrame to store the results
results_df = pd.DataFrame(index=metric_names, columns=models)

# Fill in the results
for i, metric in enumerate(metrics):
    for model in models:
        results_df.loc[metric_names[i], model] = results[model][metric]

# Display the results
display(results_df.style.format("{:.4f}").background_gradient(cmap='Blues', axis=1))

# Plot the results
plt.figure(figsize=(14, 8))
ax = results_df.plot(kind='bar', figsize=(14, 8))
plt.title('Model Performance Comparison', fontsize=16)
plt.ylabel('Score', fontsize=14)
plt.xlabel('Metric', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Model', fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=10)

plt.tight_layout()
plt.show()

# Create a radar chart for a different visualization
fig = go.Figure()

# Add traces for each model
for model in models:
    fig.add_trace(go.Scatterpolar(
        r=[results[model][metric] for metric in metrics],
        theta=metric_names,
        fill='toself',
        name=model
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title="Model Performance Radar Chart",
    width=800,
    height=600
)

fig.show()

# Plot training loss
plt.figure(figsize=(12, 6))
for model in models:
    plt.plot(history[model]['train_loss'], label=model)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Training Loss', fontsize=14)
plt.title('Training Loss Over Time', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Plot validation metrics
metrics_to_plot = ['level1_accuracy', 'level2_accuracy', 'level3_accuracy', 'hierarchical_accuracy']
titles = ['Level 1 Accuracy', 'Level 2 Accuracy', 'Level 3 Accuracy', 'Hierarchical Accuracy']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    for model in models:
        values = [metrics[metric] for metrics in history[model]['val_metrics']]
        axes[i].plot(values, label=model)
    
    axes[i].set_xlabel('Epoch', fontsize=12)
    axes[i].set_ylabel('Accuracy', fontsize=12)
    axes[i].set_title(title, fontsize=14)
    axes[i].legend(fontsize=10)
    axes[i].grid(True)

plt.tight_layout()
plt.show()

# Get class names
level1_classes = list(class_info['idx_to_level1'].values())

# Plot confusion matrices for level 1
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, model in enumerate(models):
    cm = confusion_matrices[model]['level1']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=level1_classes, yticklabels=level1_classes, ax=axes[i])
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('True', fontsize=12)
    axes[i].set_title(f'{model} Confusion Matrix (Level 1)', fontsize=14)
    
    # Rotate tick labels
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Create normalized confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for i, model in enumerate(models):
    cm = confusion_matrices[model]['level1']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=level1_classes, yticklabels=level1_classes, ax=axes[i])
    axes[i].set_xlabel('Predicted', fontsize=12)
    axes[i].set_ylabel('True', fontsize=12)
    axes[i].set_title(f'{model} Normalized Confusion Matrix (Level 1)', fontsize=14)
    
    # Rotate tick labels
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()

# Function to analyze errors
def analyze_errors(model_name, preds, class_info):
    """
    Analyze errors made by the model.
    
    Args:
        model_name: Name of the model
        preds: Dictionary containing predictions
        class_info: Dictionary containing class information
    """
    # Extract predictions
    y_true_level1 = preds['y_true_level1']
    y_pred_level1 = preds['y_pred_level1']
    y_true_level2 = preds['y_true_level2']
    y_pred_level2 = preds['y_pred_level2']
    y_true_level3 = preds['y_true_level3']
    y_pred_level3 = preds['y_pred_level3']
    
    # Calculate error rates by class (level 1)
    level1_errors = {}
    for cls in range(6):  # 6 level-1 classes
        # Get indices for this class
        idx = np.where(y_true_level1 == cls)[0]
        if len(idx) > 0:
            # Calculate error rate
            errors = np.sum(y_pred_level1[idx] != cls)
            error_rate = errors / len(idx)
            level1_errors[class_info['idx_to_level1'][cls]] = error_rate
    
    # Find common error patterns (level 1)
    error_patterns = {}
    for true_cls in range(6):
        for pred_cls in range(6):
            if true_cls != pred_cls:
                # Get indices where true class is true_cls and predicted class is pred_cls
                idx = np.where((y_true_level1 == true_cls) & (y_pred_level1 == pred_cls))[0]
                if len(idx) > 0:
                    true_name = class_info['idx_to_level1'][true_cls]
                    pred_name = class_info['idx_to_level1'][pred_cls]
                    error_patterns[f"{true_name} → {pred_name}"] = len(idx)
    
    # Sort error patterns by frequency
    error_patterns = {k: v for k, v in sorted(error_patterns.items(), key=lambda item: item[1], reverse=True)}
    
    # Calculate hierarchical error rates
    hierarchical_errors = {
        'Level 1 only': 0,
        'Level 2 only': 0,
        'Level 3 only': 0,
        'Levels 1 & 2': 0,
        'Levels 1 & 3': 0,
        'Levels 2 & 3': 0,
        'All levels': 0,
        'No errors': 0
    }
    
    for i in range(len(y_true_level1)):
        level1_error = y_true_level1[i] != y_pred_level1[i]
        level2_error = y_true_level2[i] != y_pred_level2[i]
        level3_error = y_true_level3[i] != y_pred_level3[i]
        
        if level1_error and not level2_error and not level3_error:
            hierarchical_errors['Level 1 only'] += 1
        elif not level1_error and level2_error and not level3_error:
            hierarchical_errors['Level 2 only'] += 1
        elif not level1_error and not level2_error and level3_error:
            hierarchical_errors['Level 3 only'] += 1
        elif level1_error and level2_error and not level3_error:
            hierarchical_errors['Levels 1 & 2'] += 1
        elif level1_error and not level2_error and level3_error:
            hierarchical_errors['Levels 1 & 3'] += 1
        elif not level1_error and level2_error and level3_error:
            hierarchical_errors['Levels 2 & 3'] += 1
        elif level1_error and level2_error and level3_error:
            hierarchical_errors['All levels'] += 1
        else:
            hierarchical_errors['No errors'] += 1
    
    # Convert to percentages
    total = len(y_true_level1)
    hierarchical_errors = {k: v / total * 100 for k, v in hierarchical_errors.items()}
    
    # Print results
    print(f"Error Analysis for {model_name}:\n")
    
    print("Error Rates by Class (Level 1):")
    for cls, error_rate in level1_errors.items():
        print(f"  {cls}: {error_rate:.2f}")
    
    print("\nTop 5 Error Patterns (Level 1):")
    for i, (pattern, count) in enumerate(list(error_patterns.items())[:5]):
        print(f"  {pattern}: {count} instances")
    
    print("\nHierarchical Error Distribution:")
    for error_type, percentage in hierarchical_errors.items():
        print(f"  {error_type}: {percentage:.2f}%")
    
    # Create visualizations
    # Error rates by class
    plt.figure(figsize=(10, 6))
    plt.bar(level1_errors.keys(), level1_errors.values())
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Error Rate', fontsize=12)
    plt.title(f'{model_name}: Error Rates by Class (Level 1)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Top error patterns
    top_patterns = dict(list(error_patterns.items())[:5])
    plt.figure(figsize=(12, 6))
    plt.bar(top_patterns.keys(), top_patterns.values())
    plt.xlabel('Error Pattern', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{model_name}: Top 5 Error Patterns (Level 1)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # Hierarchical error distribution
    plt.figure(figsize=(12, 6))
    plt.pie(hierarchical_errors.values(), labels=hierarchical_errors.keys(), autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title(f'{model_name}: Hierarchical Error Distribution', fontsize=14)
    plt.tight_layout()
    plt.show()

# Analyze errors for each model
for model in models:
    analyze_errors(model, predictions[model], class_info)

# Function to visualize embeddings
def visualize_embeddings(model_name, embedding_data, class_info):
    """
    Visualize embeddings using t-SNE.
    
    Args:
        model_name: Name of the model
        embedding_data: Dictionary containing embeddings
        class_info: Dictionary containing class information
    """
    # Extract embeddings and labels
    X = embedding_data['X']
    y = embedding_data['y_level1']
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'class': [class_info['idx_to_level1'][label] for label in y]
    })
    
    # Plot using matplotlib
    plt.figure(figsize=(12, 10))
    for cls in df['class'].unique():
        plt.scatter(df[df['class'] == cls]['x'], df[df['class'] == cls]['y'], label=cls, alpha=0.7)
    plt.xlabel('t-SNE dimension 1', fontsize=12)
    plt.ylabel('t-SNE dimension 2', fontsize=12)
    plt.title(f'{model_name}: t-SNE Visualization of Embeddings', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot using plotly for interactive visualization
    fig = px.scatter(df, x='x', y='y', color='class', title=f'{model_name}: t-SNE Visualization of Embeddings',
                    labels={'x': 't-SNE dimension 1', 'y': 't-SNE dimension 2', 'class': 'Class'},
                    width=900, height=700)
    fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode='markers'))
    fig.show()

# Visualize embeddings for each model
for model in models:
    visualize_embeddings(model, embeddings[model], class_info)

# Function to visualize class hierarchy
def visualize_class_hierarchy(class_hierarchy):
    """
    Visualize the class hierarchy as a network.
    
    Args:
        class_hierarchy: Dictionary representing the class hierarchy
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    # Add root node
    G.add_node('ROOT', level=0)
    
    # Add level 1 nodes and connect to root
    for l1 in class_hierarchy.keys():
        G.add_node(l1, level=1)
        G.add_edge('ROOT', l1)
        
        # Add level 2 nodes and connect to level 1
        for l2 in class_hierarchy[l1].keys():
            G.add_node(l2, level=2)
            G.add_edge(l1, l2)
            
            # Add level 3 nodes and connect to level 2
            for l3 in class_hierarchy[l1][l2]:
                G.add_node(l3, level=3)
                G.add_edge(l2, l3)
    
    # Get node positions using a hierarchical layout
    pos = nx.multipartite_layout(G, subset_key='level')
    
    # Define node colors based on level
    node_colors = []
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level == 0:
            node_colors.append('gold')
        elif level == 1:
            node_colors.append('lightblue')
        elif level == 2:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightcoral')
    
    # Define node sizes based on level
    node_sizes = []
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level == 0:
            node_sizes.append(1000)
        elif level == 1:
            node_sizes.append(500)
        elif level == 2:
            node_sizes.append(200)
        else:
            node_sizes.append(50)
    
    # Create a subset of the graph for visualization (it's too large to show everything)
    # Let's show the root, all level 1 nodes, and a sample of level 2 and 3 nodes
    nodes_to_keep = ['ROOT']
    
    # Add all level 1 nodes
    nodes_to_keep.extend(list(class_hierarchy.keys()))
    
    # Add a sample of level 2 nodes (first 2 for each level 1)
    for l1 in class_hierarchy.keys():
        l2_nodes = list(class_hierarchy[l1].keys())[:2]
        nodes_to_keep.extend(l2_nodes)
        
        # Add a sample of level 3 nodes (first 2 for each selected level 2)
        for l2 in l2_nodes:
            l3_nodes = class_hierarchy[l1][l2][:2]
            nodes_to_keep.extend(l3_nodes)
    
    # Create a subgraph
    H = G.subgraph(nodes_to_keep)
    
    # Get positions, colors, and sizes for the subgraph
    sub_pos = {node: pos[node] for node in H.nodes()}
    sub_colors = []
    sub_sizes = []
    for node in H.nodes():
        level = H.nodes[node]['level']
        if level == 0:
            sub_colors.append('gold')
            sub_sizes.append(1000)
        elif level == 1:
            sub_colors.append('lightblue')
            sub_sizes.append(500)
        elif level == 2:
            sub_colors.append('lightgreen')
            sub_sizes.append(200)
        else:
            sub_colors.append('lightcoral')
            sub_sizes.append(100)
    
    # Plot the subgraph
    plt.figure(figsize=(16, 10))
    nx.draw_networkx(
        H, pos=sub_pos,
        node_color=sub_colors,
        node_size=sub_sizes,
        font_size=8,
        arrows=True,
        with_labels=True,
        edge_color='gray',
        alpha=0.8
    )
    plt.title('Hierarchical Text Classification - Class Hierarchy (Sample)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print statistics about the hierarchy
    print("Class Hierarchy Statistics:")
    print(f"Number of level 1 classes: {len(class_hierarchy)}")
    
    level2_count = sum(len(class_hierarchy[l1]) for l1 in class_hierarchy)
    print(f"Number of level 2 classes: {level2_count}")
    
    level3_count = sum(len(class_hierarchy[l1][l2]) for l1 in class_hierarchy for l2 in class_hierarchy[l1])
    print(f"Number of level 3 classes: {level3_count}")
    
    print(f"Average number of level 2 classes per level 1: {level2_count / len(class_hierarchy):.2f}")
    print(f"Average number of level 3 classes per level 2: {level3_count / level2_count:.2f}")

# Visualize class hierarchy
visualize_class_hierarchy(class_info['class_hierarchy'])

# Create a DataFrame for level-wise performance
level_metrics = ['level1_accuracy', 'level2_accuracy', 'level3_accuracy']
level_names = ['Level 1', 'Level 2', 'Level 3']

level_df = pd.DataFrame(index=models, columns=level_names)

# Fill in the results
for model in models:
    for i, metric in enumerate(level_metrics):
        level_df.loc[model, level_names[i]] = results[model][metric]

# Display the results
display(level_df.style.format("{:.4f}").background_gradient(cmap='Blues', axis=0))

# Plot the results
plt.figure(figsize=(10, 6))
level_df.plot(kind='bar', figsize=(10, 6))
plt.title('Model Performance by Hierarchy Level', fontsize=16)
plt.ylabel('Accuracy', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Hierarchy Level', fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Calculate performance drop across levels
drop_df = pd.DataFrame(index=models, columns=['Level 1 → 2', 'Level 2 → 3', 'Level 1 → 3'])

for model in models:
    drop_df.loc[model, 'Level 1 → 2'] = level_df.loc[model, 'Level 1'] - level_df.loc[model, 'Level 2']
    drop_df.loc[model, 'Level 2 → 3'] = level_df.loc[model, 'Level 2'] - level_df.loc[model, 'Level 3']
    drop_df.loc[model, 'Level 1 → 3'] = level_df.loc[model, 'Level 1'] - level_df.loc[model, 'Level 3']

# Display the results
display(drop_df.style.format("{:.4f}").background_gradient(cmap='Reds', axis=0))

# Plot the results
plt.figure(figsize=(10, 6))
drop_df.plot(kind='bar', figsize=(10, 6))
plt.title('Performance Drop Across Hierarchy Levels', fontsize=16)
plt.ylabel('Accuracy Drop', fontsize=14)
plt.xlabel('Model', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Level Transition', fontsize=12, title_fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# End timing
end_time = time.time()
logging.info(f"Script execution completed in {end_time - start_time:.2f} seconds.")
