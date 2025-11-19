import cv2
import torch
import networkx as nx
import matplotlib.pyplot as plt
from graphs_creation import extract_landmarks, build_graph
import numpy as np

def build_inter_graph(coords_src, coords_ref, edges, shift=1.2):
    G = nx.Graph()

    src = coords_src.cpu().numpy()
    ref = coords_ref.cpu().numpy()

    ref_shifted = ref.copy()
    ref_shifted[:,0] += shift    

    N_src = len(src)
    N_ref = len(ref)

    for i in range(N_src):
        G.add_node(i, pos=(src[i,0], src[i,1]), color="red")

    for i in range(N_ref):
        G.add_node(N_src + i, pos=(ref_shifted[i,0], ref_shifted[i,1]), color="blue")

    # Add edges
    edges_np = edges.t().cpu().numpy()

    for u, v in edges_np:
        if (u < N_src and v < N_src) or (u >= N_src and v >= N_src):
         
            G.add_edge(u, v, color="gray")
        else:
           
            G.add_edge(u, v, color="green")

    return G

def draw_inter_face_graph(G, save_path=None):
    pos = nx.get_node_attributes(G, "pos")
    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    edge_colors = [G.edges[e]["color"] for e in G.edges()]

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        node_size=40,
        node_color=node_colors,
        edge_color=edge_colors,
        width=0.8,
        alpha=0.9
    )
    plt.gca().invert_yaxis()
    plt.title("Inter-Face Graph with Cross Edges")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def visualize_inter_face_graph(src_path, ref_path, k=6, device="cpu", save_path=None):
    img_src = cv2.imread(src_path)
    img_ref = cv2.imread(ref_path)

    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2RGB)
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)

    coords_src, labels_src = extract_landmarks(img_src, device=device)
    coords_ref, labels_ref = extract_landmarks(img_ref, device=device)

    edges = build_graph(coords_src, labels_src, coords_ref, labels_ref, k=k)

    G = build_inter_graph(coords_src, coords_ref, edges)
    draw_inter_face_graph(G, save_path)


visualize_inter_face_graph(
    "/home/DSE411/Documents/mlg/project/makeup_dataset/all/images/makeup/makeup_0301.png",
    "/home/DSE411/Documents/mlg/project/makeup_dataset/all/images/non-makeup/non_makeup_0031.png",
    k=6,
    device="cpu",
    save_path="inter_face_graph.png"
)
