import torch
import networkx as nx
import numpy as np
from torch_geometric.data import Data
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def initialize_feature_anchoring_dist(data: Data, anchoring_mode: str):
    """
    Initializes the anchoring distribution based on the specified mode.

    Args:
        data (Data): The graph data.
        anchoring_mode (str): The anchoring mode ('Gaussian').

    Returns:
        torch.distributions.Normal: The normal distribution for sampling anchors.
    """
    if anchoring_mode == 'Gaussian':
        feature_mean = torch.mean(data.x, dim=0)
        feature_std = torch.std(data.x, dim=0) + 1e-6
        anchoring_dist = torch.distributions.Normal(feature_mean, feature_std)
    else:
        raise ValueError('Unsupported anchoring mode.')

    return anchoring_dist

def initialize_class_feature_anchoring_dist(data: Data, anchoring_mode: str, mask):
    """
    Initializes separate anchoring distributions for each class.

    Args:
        data (Data): The graph data.
        anchoring_mode (str): The anchoring mode ('Gaussian').
        mask (Tensor): Boolean mask to select specific nodes.

    Returns:
        dict: Mapping of class labels to their normal distributions.
    """
    if anchoring_mode != 'Gaussian':
        raise ValueError('Unsupported anchoring mode.')

    x = data.x[mask]
    labels = data.y[mask]
    unique_classes = torch.unique(labels)
    class_anchoring_dists = {}

    for class_label in unique_classes:
        class_mask = labels == class_label
        class_features = x[class_mask]
        feature_mean = torch.mean(class_features, dim=0)
        feature_std = torch.std(class_features, dim=0) + 1e-6
        class_anchoring_dists[class_label.item()] = torch.distributions.Normal(feature_mean, feature_std)

    return class_anchoring_dists

def compute_graph_laplacian(G):
    """
    Computes the normalized Laplacian matrix of a graph.

    Args:
        G (Graph): A NetworkX graph.

    Returns:
        ndarray: The normalized Laplacian matrix.
    """
    return nx.normalized_laplacian_matrix(G).todense()

def spectral_clustering(L, num_clusters):
    """
    Performs spectral clustering on a Laplacian matrix.

    Args:
        L (ndarray): The Laplacian matrix.
        num_clusters (int): The number of clusters.

    Returns:
        tuple: Cluster labels and eigenvectors.
    """
    eigvals, eigvecs = eigsh(L, k=num_clusters, which='SM')
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(eigvecs)
    return kmeans.labels_, eigvecs

def calculate_normal_distribution(node_features, cluster_labels, num_clusters):
    """
    Calculates normal distributions for each cluster.

    Args:
        node_features (ndarray): The node features.
        cluster_labels (ndarray): The cluster labels for nodes.
        num_clusters (int): The number of clusters.

    Returns:
        list: List of normal distributions for each cluster.
    """
    anchors_dist = []
    for cluster_id in range(num_clusters):
        cluster_nodes = np.where(cluster_labels == cluster_id)[0]
        cluster_features = node_features[cluster_nodes]
        feature_mean = torch.mean(torch.tensor(cluster_features), dim=0)
        feature_std = torch.std(torch.tensor(cluster_features), dim=0) + 1e-6
        anchoring_dist = torch.distributions.Normal(feature_mean, feature_std)
        anchors_dist.append(anchoring_dist)
    return anchors_dist

def sample_anchors_from_normal(anchors, cluster_labels):
    """
    Samples anchors from normal distributions for each cluster.

    Args:
        anchors (list): List of normal distributions.
        cluster_labels (ndarray): The cluster labels for nodes.

    Returns:
        ndarray: Array of sampled anchor points.
    """
    sampled_anchors = [anchors[cluster_id].sample().numpy() for cluster_id in cluster_labels]
    return np.array(sampled_anchors)

def apply_anchoring(node_features, sampled_anchors):
    """
    Applies anchoring to node features.

    Args:
        node_features (ndarray): The original node features.
        sampled_anchors (ndarray): The sampled anchor points.

    Returns:
        ndarray: The anchored node features.
    """
    anchored_node_features = [
        np.concatenate((node_features[i] - sampled_anchors[i], node_features[i]))
        for i in range(len(node_features))
    ]
    return np.array(anchored_node_features)

def anchor_from_cluster(data, num_clusters=10):
    """
    Performs clustering-based anchoring for a graph.

    Args:
        data (Data): The graph data.
        num_clusters (int, optional): The number of clusters. Defaults to 10.

    Returns:
        list: Normal distributions for each cluster.
    """
    L = compute_graph_laplacian(data)
    cluster_labels, _ = spectral_clustering(L, num_clusters)
    cluster_distributions = calculate_normal_distribution(data.x, cluster_labels, num_clusters)
    return cluster_distributions
