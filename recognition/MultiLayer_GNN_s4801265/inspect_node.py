from dataset import load_dataset

data = load_dataset(seed=42)

# Choose a node from the test set
node_id = data.test_mask.nonzero(as_tuple=True)[0][0].item()

# Extract the first 10 features
features = data.x[node_id, :10].tolist()

# Find neighbouring nodes (edge_index[0] contains source nodes)
edge_index = data.edge_index
neighbours = edge_index[1, edge_index[0] == node_id].unique().tolist()

print("Node ID:", node_id)
print("Actual Class:", data.y[node_id].item())
print("First 10 Features:", [round(v, 4) for v in features])
print("Number of Neighbours:", len(neighbours))
print("Example Neighbours:", neighbours[:10])