import torch
import plotly.graph_objects as go
import numpy as np
from torch_geometric.nn import knn_graph
from dataset import get_dataloaders
from models import GeoGraph3D
import yaml


def load_model(config_path, model_path, device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Initialize model structure
    model = GeoGraph3D(
        k=config["model"]["k"],
        output_channels=40,  # ModelNet40 has 40 classes
        emb_dims=config["model"]["emb_dims"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config


def visualize_dynamic_graph(data, k=20, title="Dynamic Graph Visualization"):
    """
    Creates a 3D interactive plot of the point cloud and its graph connections.
    """
    pos = data.pos.cpu().numpy()

    # 1. Re-compute the graph edges (k-NN) just like the model does
    # This shows "How the AI sees the object"
    edge_index = knn_graph(data.pos, k=k, batch=data.batch, loop=False)
    edge_index = edge_index.cpu().numpy()

    # 2. Setup 3D Plot
    trace_nodes = go.Scatter3d(
        x=pos[:, 0],
        y=pos[:, 1],
        z=pos[:, 2],
        mode="markers",
        marker=dict(size=4, color=pos[:, 2], colorscale="Viridis", opacity=0.8),
        name="Points",
    )

    # 3. Draw Edges (Lines between neighbors)
    # We only draw a subset to keep the visual clean
    edge_x = []
    edge_y = []
    edge_z = []

    # Limit edges for visualization speed (first 1000 edges)
    for i in range(min(edge_index.shape[1], 2000)):
        start_idx = edge_index[0, i]
        end_idx = edge_index[1, i]
        edge_x.extend([pos[start_idx, 0], pos[end_idx, 0], None])
        edge_y.extend([pos[start_idx, 1], pos[end_idx, 1], None])
        edge_z.extend([pos[start_idx, 2], pos[end_idx, 2], None])

    trace_edges = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color="lightgrey", width=1),
        opacity=0.3,
        name="Graph Connections",
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig = go.Figure(data=[trace_nodes, trace_edges], layout=layout)
    fig.write_html("dynamic_graph_viz.html")
    print(">> Saved visualization to 'dynamic_graph_viz.html'")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and a sample
    config_path = "configs/config.yaml"
    # NOTE: Run train.py first to generate 'best_model.pth'
    try:
        model, config = load_model(config_path, "best_model.pth", device)
        _, test_loader, _ = get_dataloaders(config)

        # Get one sample batch
        data = next(iter(test_loader))
        data = data.to(device)

        # Take the first object in the batch
        # We need to slice the batch to get just one object
        single_obj_mask = data.batch == 0

        # Create a mini data object for visualization
        from torch_geometric.data import Data

        viz_data = Data(
            pos=data.pos[single_obj_mask],
            batch=torch.zeros(single_obj_mask.sum(), dtype=torch.long).to(device),
        )

        visualize_dynamic_graph(viz_data, k=20)

    except FileNotFoundError:
        print("ERROR: 'best_model.pth' not found. Please run 'train.py' first!")
