"""
Facebook Page-Page Dataset available here on Rangpur:
/home/groups/comp3710
"""
import torch
from torch_geometric.datasets import FacebookPagePage


# Download and load the Facebook Page-Page dataset
dataset = FacebookPagePage(root="data/FacebookPagePage")

# The dataset contains one graph
data = dataset[0]


# Inspect the graph dimensions
print("=== Graph Dimensions ===")
print("Number of graphs:", len(dataset))
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of node features:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)


# Inspect the node features
print("\n=== Node Features ===")
print("Feature matrix shape:", data.x.shape)
print("Feature data type:", data.x.dtype)
print("First node's features:", data.x[0])


# Inspect the node labels
print("\n=== Node Labels ===")
print("Label tensor shape:", data.y.shape)
print("Label data type:", data.y.dtype)
print("Unique labels:", torch.unique(data.y))
print("First 10 labels:", data.y[:10])


# Inspect the class distribution
print("\n=== Class Distribution ===")

classes, counts = torch.unique(data.y, return_counts=True)

for class_id, count in zip(classes, counts):
    percentage = count.item() / data.num_nodes * 100

    print(
        f"Class {class_id.item()}: "
        f"{count.item()} nodes ({percentage:.2f}%)"
    )


