import polars as pl
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import json
import os

def analyze_hierarchy_distribution(df):
    """
    Analyze the distribution of hierarchy codes in the DataFrame
    
    Args:
        df: Polars DataFrame with hierarchy codes
    """
    print("\nAnalyzing hierarchy code distribution...")
    
    # Group by hierarchy_code and count
    distribution = (
        df.group_by("hierarchy_code")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    
    # Print top 20 most common codes
    print("\nTop 20 most common hierarchy codes:")
    print(distribution.head(20))
    
    # Print summary statistics
    total_codes = distribution["count"].sum()
    unique_codes = len(distribution)
    print(f"\nTotal items: {total_codes}")
    print(f"Unique hierarchy codes: {unique_codes}")
    
    # Calculate percentage of GEN codes
    gen_codes = distribution.filter(pl.col("hierarchy_code").str.starts_with("GEN"))
    gen_count = gen_codes["count"].sum()
    gen_percentage = (gen_count / total_codes) * 100
    print(f"\nGEN codes: {gen_count} ({gen_percentage:.2f}%)")
    
    return distribution

def analyze_cluster_distribution(df):
    """
    Analyze the distribution of clusters at each level
    
    Args:
        df: Polars DataFrame with cluster assignments
    """
    print("\nAnalyzing cluster distribution...")
    
    # Analyze each level
    for level in ['l1_cluster', 'l2_cluster', 'l3_cluster']:
        print(f"\n{level.upper()} Distribution:")
        distribution = (
            df.group_by(level)
            .agg(pl.len().alias("count"))
            .sort(level)
        )
        print(distribution)
        
        # Calculate statistics
        total = distribution["count"].sum()
        unique = len(distribution)
        print(f"Total items: {total}")
        print(f"Unique clusters: {unique}")
        print(f"Average items per cluster: {total/unique:.2f}")

def analyze_product_types_by_cluster(df):
    """
    Analyze the distribution of product types within each top-level cluster
    
    Args:
        df: Polars DataFrame with cluster assignments and product types
    """
    print("\nAnalyzing product types by cluster...")
    
    # Get top 5 clusters by size
    top_clusters = (
        df.group_by("l1_cluster")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(5)
    )
    
    print("\nTop 5 clusters by size:")
    for row in top_clusters.iter_rows(named=True):
        cluster_id = row["l1_cluster"]
        count = row["count"]
        print(f"\nCluster {cluster_id} ({count} items):")
        
        # Get product type distribution for this cluster
        product_types = (
            df.filter(pl.col("l1_cluster") == cluster_id)
            .group_by("product_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .head(10)
        )
        
        for pt_row in product_types.iter_rows(named=True):
            print(f"  - {pt_row['product_type']}: {pt_row['count']} items")

def plot_hierarchy_distribution(distribution, output_dir):
    """
    Create a bar plot of the top N hierarchy codes
    
    Args:
        distribution: Polars DataFrame with hierarchy code counts
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(15, 8))
    top_n = 20
    
    # Get top N codes
    top_codes = distribution.head(top_n)
    
    # Create bar plot
    sns.barplot(data=top_codes.to_pandas(), x="hierarchy_code", y="count")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Top {top_n} Hierarchy Codes")
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir) / "hierarchy_distribution.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved distribution plot to {output_path}")

def visualize_clusters(df, output_dir):
    """
    Create visualizations of clusters using PCA
    
    Args:
        df: Polars DataFrame with embeddings and cluster assignments
        output_dir: Directory to save visualizations
    """
    print("\nGenerating cluster visualizations...")
    
    # Convert embeddings from string to numpy array
    embeddings = np.array([json.loads(e) for e in df['embedding']])
    
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    # Create visualization for each cluster level
    for level in ['l1_cluster', 'l2_cluster', 'l3_cluster']:
        plt.figure(figsize=(15, 10))
        
        # Get unique clusters and their sizes
        cluster_sizes = (
            df.group_by(level)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        
        # Plot each cluster with a different color
        for cluster_id in np.unique(df[level]):
            mask = df[level] == cluster_id
            size = cluster_sizes.filter(pl.col(level) == cluster_id)["count"][0]
            plt.scatter(
                reduced_embeddings[mask, 0],
                reduced_embeddings[mask, 1],
                label=f'Cluster {cluster_id} (n={size})',
                alpha=0.7
            )
        
        plt.title(f'{level.upper()} Clusters (PCA Visualization)')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(output_dir) / f'{level}_clusters_visualization.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {level} visualization to {output_path}")
    
    # Create cluster size distribution plots
    for level in ['l1_cluster', 'l2_cluster', 'l3_cluster']:
        plt.figure(figsize=(15, 6))
        
        # Get cluster sizes
        cluster_sizes = (
            df.group_by(level)
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        
        # Create bar plot
        sns.barplot(data=cluster_sizes.to_pandas(), x=level, y="count")
        plt.title(f'{level.upper()} Cluster Sizes')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Items')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        output_path = Path(output_dir) / f'{level}_size_distribution.png'
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved {level} size distribution to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze cluster and hierarchy code distributions')
    parser.add_argument('--input_file', type=str, 
                      default='abo-dataset/with_hierarchy_codes.parquet',
                      help='Path to the parquet file with hierarchy codes (default: abo-dataset/with_hierarchy_codes.parquet)')
    parser.add_argument('--output_dir', type=str, default='abo-dataset/analysis',
                      help='Directory to save analysis plots (default: abo-dataset/analysis)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the parquet file
    print(f"Reading data from {args.input_file}")
    df = pl.read_parquet(args.input_file)
    print(f"Loaded {df.height} rows")
    
    # Run analyses
    distribution = analyze_hierarchy_distribution(df)
    analyze_cluster_distribution(df)
    analyze_product_types_by_cluster(df)
    
    # Create visualizations
    plot_hierarchy_distribution(distribution, args.output_dir)
    visualize_clusters(df, args.output_dir)

if __name__ == "__main__":
    main()
