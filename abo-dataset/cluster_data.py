import polars as pl
import numpy as np
from pathlib import Path
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import argparse

def load_embeddings(file_path):
    """
    Load pre-generated embeddings from pickle file
    
    Args:
        file_path: Path to the embeddings pickle file
        
    Returns:
        numpy.ndarray: Combined embeddings for clustering
    """
    print(f"Loading embeddings from {file_path}")
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings with shape: {embeddings.shape}")
    return embeddings

def apply_hierarchical_clustering(embeddings, n_clusters_l1=8, n_clusters_l2=15, n_clusters_l3=30):
    """
    Apply hierarchical clustering at three levels
    
    Args:
        embeddings: Feature embeddings for clustering
        n_clusters_l1: Number of clusters at level 1 (default: 8)
        n_clusters_l2: Number of clusters at level 2 (default: 15)
        n_clusters_l3: Number of clusters at level 3 (default: 30)
        
    Returns:
        tuple: (l1_clusters, l2_clusters, l3_clusters) cluster assignments
    """
    print(f"Applying hierarchical clustering with {n_clusters_l1}/{n_clusters_l2}/{n_clusters_l3} clusters...")
    
    # Level 1: Broad categories
    l1_clustering = AgglomerativeClustering(
        n_clusters=n_clusters_l1,
        metric='euclidean',
        linkage='ward'
    )
    l1_clusters = l1_clustering.fit_predict(embeddings)
    
    # Level 2: Subcategories
    l2_clusters = np.zeros(len(embeddings), dtype=int)
    cluster_offset = 0
    
    for l1_id in range(n_clusters_l1):
        # Get samples for this Level 1 cluster
        mask = l1_clusters == l1_id
        subset_embeddings = embeddings[mask]
        
        if len(subset_embeddings) > 1:
            # Calculate proportional number of clusters for this subset
            n_clusters_for_subset = max(2, int(n_clusters_l2 * len(subset_embeddings) / len(embeddings)))
            
            # Apply clustering to this subset
            subset_clustering = AgglomerativeClustering(
                n_clusters=n_clusters_for_subset,
                metric='euclidean',
                linkage='ward'
            )
            subset_labels = subset_clustering.fit_predict(subset_embeddings)
            
            # Assign cluster labels (with offset to ensure unique labels across L1 clusters)
            l2_clusters[mask] = subset_labels + cluster_offset
            cluster_offset += n_clusters_for_subset
        else:
            # Handle single-element clusters
            l2_clusters[mask] = cluster_offset
            cluster_offset += 1
    
    # Level 3: Detailed groupings
    l3_clusters = np.zeros(len(embeddings), dtype=int)
    cluster_offset = 0
    
    # Get unique L2 cluster ids
    unique_l2_clusters = np.unique(l2_clusters)
    
    for l2_id in unique_l2_clusters:
        # Get samples for this Level 2 cluster
        mask = l2_clusters == l2_id
        subset_embeddings = embeddings[mask]
        
        if len(subset_embeddings) > 1:
            # Calculate proportional number of clusters for this subset
            n_clusters_for_subset = max(2, int(n_clusters_l3 * len(subset_embeddings) / len(embeddings)))
            
            # Apply clustering to this subset
            subset_clustering = AgglomerativeClustering(
                n_clusters=n_clusters_for_subset,
                metric='euclidean',
                linkage='ward'
            )
            subset_labels = subset_clustering.fit_predict(subset_embeddings)
            
            # Assign cluster labels (with offset to ensure unique labels across L2 clusters)
            l3_clusters[mask] = subset_labels + cluster_offset
            cluster_offset += n_clusters_for_subset
        else:
            # Handle single-element clusters
            l3_clusters[mask] = cluster_offset
            cluster_offset += 1
    
    print(f"Clustering complete: {len(np.unique(l1_clusters))} L1 clusters, "
          f"{len(np.unique(l2_clusters))} L2 clusters, {len(np.unique(l3_clusters))} L3 clusters")
    
    return l1_clusters, l2_clusters, l3_clusters

def generate_hierarchy_codes(df, l1_clusters, l2_clusters, l3_clusters):
    """
    Generate hierarchy codes based on product type and cluster assignments
    
    Args:
        df: DataFrame with product data
        l1_clusters, l2_clusters, l3_clusters: Cluster assignments at each level
        
    Returns:
        pl.Series: Hierarchy codes for each product
    """
    # Mapping from product type to code prefix
    type_mapping = {
        "furniture": "FUR",
        "sofa": "SOF",
        "chair": "CHR",
        "table": "TBL",
        "desk": "DSK",
        "bed": "BED",
        "shelf": "SHF",
        "cabinet": "CAB",
        "storage": "STR",
        "lamp": "LMP",
        "lighting": "LGT",
        "rug": "RUG",
        "curtain": "CRT",
        "pillow": "PLW",
        "blanket": "BLK",
        "kitchenware": "KIT",
        "appliance": "APL"
    }
    
    # Default code for unknown product types
    default_code = "GEN"
    
    # Create Polars series for cluster assignments
    l1_series = pl.Series(l1_clusters)
    l2_series = pl.Series(l2_clusters)
    l3_series = pl.Series(l3_clusters)
    
    # Add cluster assignments to DataFrame
    df_with_clusters = df.with_columns([
        pl.lit(list(range(df.height))).alias("index"),
        l1_series.alias("l1_cluster"),
        l2_series.alias("l2_cluster"),
        l3_series.alias("l3_cluster")
    ])
    
    # Generate hierarchy codes using Polars expressions
    hierarchy_codes = df_with_clusters.select([
        pl.col("product_type").map_elements(
            lambda product_type: next(
                (code for key, code in type_mapping.items() 
                 if key in str(product_type).lower()),
                default_code
            )
        ).alias("type_code"),
        pl.col("l1_cluster").cast(pl.Int64),
        pl.col("l2_cluster").cast(pl.Int64),
        pl.col("l3_cluster").cast(pl.Int64)
    ]).with_columns([
        (
            pl.col("type_code") + "-" + 
            pl.col("l1_cluster").map_elements(lambda x: f"{int(x):02d}") +
            pl.col("l2_cluster").map_elements(lambda x: f"{int(x):02d}") + "-" +
            pl.col("l3_cluster").map_elements(lambda x: f"{int(x):02d}")
        ).alias("hierarchy_code")
    ]).select("hierarchy_code")
    
    return hierarchy_codes["hierarchy_code"]

def visualize_clusters(embeddings, l1_clusters, output_dir):
    """
    Visualize clusters using PCA
    
    Args:
        embeddings: Feature embeddings
        l1_clusters: Level 1 cluster assignments
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply PCA to reduce dimensions for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Plot each cluster with a different color
    for cluster_id in np.unique(l1_clusters):
        mask = l1_clusters == cluster_id
        plt.scatter(
            reduced_embeddings[mask, 0],
            reduced_embeddings[mask, 1],
            label=f'Cluster {cluster_id}',
            alpha=0.7
        )
    
    plt.title('Level 1 Clusters (PCA Visualization)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'l1_clusters_visualization.png'))
    plt.close()
    
    print(f"Cluster visualization saved to {output_dir}")

def analyze_clusters(df, l1_clusters):
    """
    Analyze the contents of each cluster to understand what they represent
    
    Args:
        df: DataFrame with product data
        l1_clusters: Level 1 cluster assignments
    """
    print("\nCluster Analysis:")
    
    for cluster_id in range(len(np.unique(l1_clusters))):
        # Get items in this cluster
        cluster_mask = l1_clusters == cluster_id
        cluster_indices = [i for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
        
        # Use cluster indices to filter the dataframe
        cluster_items = df.filter(pl.col("index").is_in(cluster_indices))
        
        # Get the most common product types
        if 'product_type' in df.columns:
            product_counts = (
                cluster_items
                .group_by('product_type')
                .agg(pl.count().alias('count'))
                .sort('count', descending=True)
                .limit(5)
            )
            
            print(f"\nCluster {cluster_id} ({cluster_items.height} items):")
            print("  Top product types:")
            for row in product_counts.iter_rows(named=True):
                print(f"    - {row['product_type']}: {row['count']} items")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cluster data with specified split')
    parser.add_argument('--split', type=str, default='train', 
                        choices=['train', 'val', 'test', 'full'],
                        help='Data split to use (default: train)')
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path(__file__).parent
    
    # Use paths based on selected split
    split = args.split
    input_file = base_dir / f"clustering_dataset_{split}.parquet"
    embeddings_file = base_dir / "embeddings" / f"embeddings_{split}.pkl"
    output_file = base_dir / f"hierarchy_codes_{split}.parquet"
    vis_dir = base_dir / "visualizations" / split
    
    # Print selected split
    print(f"Using {split} split for clustering")
    
    # Load data
    print(f"Loading clustering dataset from {input_file}")
    df = pl.read_parquet(input_file)
    # Add index column for cluster analysis
    df = df.with_column(pl.lit(list(range(df.height))).alias("index"))
    print(f"Loaded dataset with {df.height} rows and {df.width} columns")
    
    # Load pre-generated embeddings
    embeddings = load_embeddings(embeddings_file)
    
    # Apply hierarchical clustering
    l1_clusters, l2_clusters, l3_clusters = apply_hierarchical_clustering(
        embeddings,
        n_clusters_l1=8,
        n_clusters_l2=15,
        n_clusters_l3=30
    )
    
    # Generate hierarchy codes
    hierarchy_codes = generate_hierarchy_codes(df, l1_clusters, l2_clusters, l3_clusters)
    
    # Add clusters and hierarchy codes to DataFrame
    df_with_clusters = df.with_columns([
        pl.Series("l1_cluster", l1_clusters),
        pl.Series("l2_cluster", l2_clusters),
        pl.Series("l3_cluster", l3_clusters),
        pl.Series("hierarchy_code", hierarchy_codes)
    ])
    
    # Visualize clusters
    visualize_clusters(embeddings, l1_clusters, str(vis_dir))
    
    # Analyze clusters
    analyze_clusters(df, l1_clusters)
    
    # Save results
    df_with_clusters.write_parquet(str(output_file))
    print(f"Saved hierarchy codes to {output_file}")
    
    # Print sample hierarchy codes
    print("\nSample hierarchy codes:")
    sample_indices = np.random.choice(len(df), min(10, len(df)), replace=False)
    sample_data = df_with_clusters.filter(pl.col("index").is_in(sample_indices.tolist()))
    
    for row in sample_data.select(['product_type', 'hierarchy_code']).iter_rows(named=True):
        print(f"Product Type: {row['product_type']} -> Hierarchy Code: {row['hierarchy_code']}")
