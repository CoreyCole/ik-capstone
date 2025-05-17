import polars as pl
import glob
import os
from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from typing import Optional

def load_jsonl_files(directory: str, sample_size: int = None) -> pl.DataFrame:
    """
    Load all JSONL files from the specified directory into a Polars DataFrame.
    
    Args:
        directory (str): Path to the directory containing JSONL files
        sample_size (int, optional): Maximum number of records to load (for testing)
        
    Returns:
        pl.DataFrame: Combined DataFrame containing all data
    """
    # Get all .json files (excluding .gz files)
    json_files = glob.glob(os.path.join(directory, "listings_*.json"))
    
    # Since there are schema inconsistencies, we'll load and process the raw data first
    all_data = []
    schema_details = {}
    
    for file_path in json_files:
        print(f"Processing {file_path}...")
        
        # Read raw data first
        with open(file_path, 'r') as f:
            file_data = [json.loads(line) for line in f if line.strip()]
            
        # Record schema information for debugging
        file_name = os.path.basename(file_path)
        if file_data:
            schema_details[file_name] = {
                "columns": list(file_data[0].keys()),
                "count": len(file_data)
            }
            all_data.extend(file_data)
            
            # If we've collected enough samples, stop
            if sample_size and len(all_data) >= sample_size:
                all_data = all_data[:sample_size]
                print(f"Stopped after collecting {len(all_data)} records (sample_size={sample_size})")
                break
    
    # Print schema analysis
    print("\nSchema analysis:")
    for file_name, info in schema_details.items():
        print(f"{file_name}: {len(info['columns'])} columns, {info['count']} records")
    
    # Find all unique column names
    all_columns = sorted(list(set().union(*[set(item.keys()) for item in all_data])))
    print(f"\nTotal unique columns across all files: {len(all_columns)}")
    print("Columns:", all_columns)
    
    # Normalize all records to have the same schema
    normalized_data = []
    for item in all_data:
        normalized_item = {}
        
        # Ensure all columns are present and convert complex objects to strings
        for col in all_columns:
            if col in item:
                # Handle different types by converting lists or dicts to strings
                if isinstance(item[col], (list, dict)):
                    normalized_item[col] = json.dumps(item[col])
                else:
                    normalized_item[col] = item[col]
            else:
                normalized_item[col] = None
                
        normalized_data.append(normalized_item)
    
    # Create DataFrame from normalized data
    combined_df = pl.DataFrame(normalized_data)
    print(f"Total rows in combined DataFrame: {combined_df.height}")
    
    return combined_df

def select_columns_for_clustering(df: pl.DataFrame) -> pl.DataFrame:
    """
    Select and preprocess columns that are relevant for hierarchical clustering.
    
    Args:
        df (pl.DataFrame): The original DataFrame with all columns
        
    Returns:
        pl.DataFrame: DataFrame with selected and preprocessed columns
    """
    # - product_type (Insurance policy type)
    # - brand (Business organization)
    # - model_name (Policy template)
    # - item_dimensions (Coverage limits)
    # - material (Coverage specifics)
    # - price/weight as proxy (Premium)
    # - product_description (Policy description)
    # - color, pattern, style (Additional attributes that could influence categorization)
    
    # Define columns to keep
    columns_to_keep = [
        'item_id',           # Primary identifier
        'product_type',      # Primary category
        'brand',             # Manufacturer/organization
        'model_name',        # Product model
        'model_number',      # Specific model variant
        'material',          # Material composition
        'item_dimensions',   # Size attributes
        'item_weight',       # Weight (proxy for price/premium)
        'product_description', # Detailed description
        'color',             # Color attributes
        'pattern',           # Pattern information
        'style',             # Style characteristics
        'bullet_point',      # Key features (often contains important categorization info)
        'item_keywords',     # Keywords (useful for classification)
        'item_name'          # Product name (contains useful categorization info)
    ]
    
    # Filter to keep only columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_keep if col in df.columns]
    
    # Select the relevant columns
    cluster_df = df.select(existing_columns)
    
    print(f"\nSelected {len(existing_columns)} columns for clustering:")
    print(existing_columns)
    
    return cluster_df

def preprocess_text_fields(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extract and preprocess text fields from JSON strings using only Polars
    
    Args:
        df: DataFrame with text fields encoded as JSON strings
        
    Returns:
        DataFrame with extracted text fields
    """
    text_columns = ['product_type', 'brand', 'item_name', 'product_description', 
                   'color', 'pattern', 'style', 'material']
    
    # Define an expression to extract text from JSON strings
    def extract_text_expr(col: str) -> pl.Expr:
        return (
            pl.when(pl.col(col).is_null())
            .then(pl.lit(""))
            .otherwise(
                pl.col(col).map_elements(
                    lambda json_str: extract_json_value(json_str),
                    return_dtype=pl.Utf8
                )
            )
        )
    
    # Helper function to extract values from JSON
    def extract_json_value(json_str: Optional[str]) -> str:
        if json_str is None:
            return ""
        try:
            data = json.loads(json_str)
            # Handle list of objects with 'value' key
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                if 'value' in data[0]:
                    return data[0]['value']
            # Handle single object with 'value' key
            elif isinstance(data, dict) and 'value' in data:
                return data['value']
            return str(data)
        except (json.JSONDecodeError, TypeError):
            return str(json_str)
    
    # Apply the extraction to each text column that exists in the DataFrame
    for column in text_columns:
        if column in df.columns:
            print(f"Processing {column}...")
            df = df.with_columns(
                extract_text_expr(column).alias(column)
            )
    
    return df

def generate_embeddings(df: pl.DataFrame) -> np.ndarray:
    """
    Generate embeddings from text and numerical features
    
    Args:
        df: Preprocessed DataFrame
        
    Returns:
        numpy.ndarray: Combined embeddings for clustering
    """
    print("Generating embeddings...")
    
    # Convert to pandas for the embedding model
    pdf = df.to_pandas()
    
    # Initialize sentence transformer for text embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # 1. Text embeddings
    text_columns = ['product_type', 'brand', 'item_name', 'product_description', 
                    'color', 'pattern', 'style', 'material']
    available_text_cols = [col for col in text_columns if col in pdf.columns]
    
    # Concatenate available text columns for embedding
    text_data = pdf[available_text_cols].fillna("").agg(' '.join, axis=1)
    print(f"Generating text embeddings for {len(text_data)} items...")
    text_embeddings = model.encode(text_data.tolist(), show_progress_bar=True)
    print(f"Text embedding shape: {text_embeddings.shape}")
    
    # 2. Categorical features encoding
    categorical_features = []
    categorical_columns = ['product_type', 'brand', 'color', 'pattern', 'style', 'material']
    available_cat_cols = [col for col in categorical_columns if col in pdf.columns]
    
    for col in available_cat_cols:
        print(f"Encoding categorical column: {col}")
        le = LabelEncoder()
        encoded = le.fit_transform(pdf[col].fillna('unknown'))
        categorical_features.append(encoded.reshape(-1, 1))
    
    if categorical_features:
        categorical_embeddings = np.hstack(categorical_features)
        print(f"Categorical embedding shape: {categorical_embeddings.shape}")
    else:
        categorical_embeddings = np.zeros((len(pdf), 1))
    
    # 3. Numerical features
    numerical_columns = ['item_weight_numeric', 'height', 'width', 'length']
    available_num_cols = [col for col in numerical_columns if col in pdf.columns]
    
    if available_num_cols and len(available_num_cols) > 0:
        # Extract and normalize numerical features
        numerical_data = pdf[available_num_cols].copy()
        
        # Fill NaN values with mean for each column
        for col in available_num_cols:
            col_mean = numerical_data[col].mean()
            numerical_data[col] = numerical_data[col].fillna(col_mean)
        
        # Standardize numerical features
        scaler = StandardScaler()
        numerical_embeddings = scaler.fit_transform(numerical_data)
        print(f"Numerical embedding shape: {numerical_embeddings.shape}")
    else:
        numerical_embeddings = np.zeros((len(pdf), 1))
    
    # 4. Combine embeddings with appropriate weighting
    # Normalize each embedding type
    text_embeddings_norm = text_embeddings / (np.linalg.norm(text_embeddings, axis=1, keepdims=True) + 1e-10)
    categorical_embeddings_norm = categorical_embeddings / (np.linalg.norm(categorical_embeddings, axis=1, keepdims=True) + 1e-10)
    numerical_embeddings_norm = numerical_embeddings / (np.linalg.norm(numerical_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Weighted combination: text (60%), categorical (30%), numerical (10%)
    combined_embeddings = np.hstack([
        text_embeddings_norm * 0.6,
        categorical_embeddings_norm * 0.3,
        numerical_embeddings_norm * 0.1
    ])
    
    print(f"Combined embedding shape: {combined_embeddings.shape}")
    return combined_embeddings

if __name__ == "__main__":
    # Path to the metadata directory
    metadata_dir = Path(__file__).parent / "metadata"
    
    # Sample size for testing (set to None to process all data)
    sample_size = 5000
    
    # Load and combine JSON files with sample size limit
    df = load_jsonl_files(str(metadata_dir), sample_size=sample_size)
    
    # Display basic information about the DataFrame
    print("\nDataFrame Info:")
    print(df.shape)
    print(df.columns)
    
    # Select columns for clustering
    cluster_df = select_columns_for_clustering(df)
    
    # Preprocess the data
    preprocessed_df = preprocess_text_fields(cluster_df)
    
    # Generate embeddings
    embeddings = generate_embeddings(preprocessed_df)
    
    # Create train/validation/test splits (60/20/20)
    indices = np.arange(len(preprocessed_df))
    train_indices, test_indices = train_test_split(indices, test_size=0.4, random_state=42)
    val_indices, test_indices = train_test_split(test_indices, test_size=0.5, random_state=42)
    
    # Add an index column to the DataFrame for filtering
    preprocessed_df = preprocessed_df.with_column(pl.Series("temp_index", indices))
    
    # Filter using Polars expressions
    train_df = preprocessed_df.filter(pl.col("temp_index").is_in(train_indices))
    val_df = preprocessed_df.filter(pl.col("temp_index").is_in(val_indices))
    test_df = preprocessed_df.filter(pl.col("temp_index").is_in(test_indices))
    
    # Remove the temporary index column
    train_df = train_df.drop("temp_index")
    val_df = val_df.drop("temp_index")
    test_df = test_df.drop("temp_index")
    
    # Split the embeddings
    train_mask = np.isin(indices, train_indices)
    val_mask = np.isin(indices, val_indices)
    test_mask = np.isin(indices, test_indices)
    
    train_embeddings = embeddings[train_mask]
    val_embeddings = embeddings[val_mask]
    test_embeddings = embeddings[test_mask]
    
    print(f"\nSplit data into:")
    print(f"  Train: {train_df.height} rows ({train_df.height/preprocessed_df.height:.1%})")
    print(f"  Validation: {val_df.height} rows ({val_df.height/preprocessed_df.height:.1%})")
    print(f"  Test: {test_df.height} rows ({test_df.height/preprocessed_df.height:.1%})")
    
    # Save the full combined DataFrame to a parquet file for efficient storage
    output_path = metadata_dir.parent / "combined_listings.parquet"
    df.write_parquet(str(output_path))
    print(f"\nSaved combined DataFrame to: {output_path}")
    
    # Save the clustering dataset separately for each split
    base_cluster_path = metadata_dir.parent / "clustering_dataset"
    
    train_output_path = f"{base_cluster_path}_train.parquet"
    val_output_path = f"{base_cluster_path}_val.parquet"
    test_output_path = f"{base_cluster_path}_test.parquet"
    full_output_path = f"{base_cluster_path}.parquet"
    
    train_df.write_parquet(train_output_path)
    val_df.write_parquet(val_output_path)
    test_df.write_parquet(test_output_path)
    preprocessed_df.write_parquet(full_output_path)
    
    print(f"Saved clustering datasets to:")
    print(f"  Full: {full_output_path}")
    print(f"  Train: {train_output_path}")
    print(f"  Validation: {val_output_path}")
    print(f"  Test: {test_output_path}")
    
    # Save embeddings to pickle files
    embeddings_dir = metadata_dir.parent / "embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    with open(embeddings_dir / "embeddings_full.pkl", 'wb') as f:
        pickle.dump(embeddings, f)
    with open(embeddings_dir / "embeddings_train.pkl", 'wb') as f:
        pickle.dump(train_embeddings, f)
    with open(embeddings_dir / "embeddings_val.pkl", 'wb') as f:
        pickle.dump(val_embeddings, f)
    with open(embeddings_dir / "embeddings_test.pkl", 'wb') as f:
        pickle.dump(test_embeddings, f)
    
    print(f"Saved embeddings to {embeddings_dir}/")
    print(f"  Full: {len(embeddings)} embeddings")
    print(f"  Train: {len(train_embeddings)} embeddings")
    print(f"  Validation: {len(val_embeddings)} embeddings")
    print(f"  Test: {len(test_embeddings)} embeddings") 