import os
import torch
import matplotlib.pyplot as plt
# only gpu 0 is available
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_adj(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adjacency file not found: {path}")
    adj = torch.load(path)
    print(f"Loaded adjacency from {path} with shape {adj.shape}")
    return adj

def get_first_frame(adj):
    # adj shape = [Samples, SL, N, N]
    # pick sample 5, frame 10
    return adj[5, 10].cpu().numpy()

def plot_comparison(baseline, improved, save_path="adjacency_comparison.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(baseline, cmap="viridis")
    plt.colorbar()
    plt.title("Baseline Adjacency (Frame 0)")

    plt.subplot(1, 2, 2)
    plt.imshow(improved, cmap="viridis")
    plt.colorbar()
    plt.title("Improved Adjacency (Frame 0)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved comparison figure to {save_path}")
    plt.show()

if __name__ == "__main__":
    # ---- update these paths if needed ----
    baseline_path  = os.path.expanduser("~/BestPickled/Test/Adj_Mat_Scene.pt")
    improved_path  = "./Pickled/Test/Adj_Mat_Scene.pt"

    print("Loading baseline adjacency...")
    adj_base = load_adj(baseline_path)

    print("Loading improved adjacency...")
    adj_improved = load_adj(improved_path)

    print("Extracting first frames...")
    base_frame = get_first_frame(adj_base)
    imp_frame  = get_first_frame(adj_improved)

    print("Plotting comparison...")
    plot_comparison(base_frame, imp_frame)