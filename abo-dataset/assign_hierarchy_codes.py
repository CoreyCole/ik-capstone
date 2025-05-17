import polars as pl
from collections import defaultdict
import csv
import argparse
from pathlib import Path

def normalize_type(pt):
    if pt is None:
        return ""
    return str(pt).strip().lower().replace(' ', '_')

def generate_hierarchy_codes(df, unmatched_counter=None):
    """
    Generate hierarchy codes based on product type and cluster assignments
    Args:
        df: DataFrame with product data and cluster assignments
        unmatched_counter: defaultdict to count unmatched product types
    Returns:
        pl.Series: Hierarchy codes for each product
    """
    print("\n=== DEBUG: Starting generate_hierarchy_codes ===")
    
    # Get unique product types and create mapping
    unique_types = df.select("product_type").unique().to_series()
    print(f"Found {len(unique_types)} unique product types")
    print(f"First 5 product types: {unique_types.head(5).to_list()}")
    
    type_mapping = {}
    used_codes = set()
    
    for pt in unique_types:
        if pt is None:
            continue
        pt_norm = normalize_type(pt)
        if not pt_norm:
            continue
            
        # Get first three characters, uppercase
        base_code = pt_norm[:3].upper()
        
        # Always add numeric suffix, starting with 000
        counter = 0
        while f"{base_code}{counter:03d}" in used_codes:
            counter += 1
        code = f"{base_code}{counter:03d}"
            
        type_mapping[pt_norm] = code
        used_codes.add(code)
    
    print(f"Created {len(type_mapping)} type mappings")
    if len(type_mapping) > 0:
        print(f"First 5 mappings: {list(type_mapping.items())[:5]}")
    else:
        print("WARNING: No type mappings created!")
    
    normalized_type_mapping = {normalize_type(k): v for k, v in type_mapping.items()}
    
    def get_product_code(product_type):
        pt_norm = normalize_type(product_type)
        if pt_norm in normalized_type_mapping:
            return normalized_type_mapping[pt_norm]
        if unmatched_counter is not None:
            unmatched_counter[pt_norm] += 1
        return "GEN"
    
    # Debug sample of product_type values
    sample_types = df.select("product_type").sample(5).to_series().to_list()
    print(f"Sample product types: {sample_types}")
    print(f"Corresponding codes: {[get_product_code(pt) for pt in sample_types]}")
    
    # Create with type_code column for debugging
    print("\n=== DEBUG: Creating hierarchy_codes DataFrame ===")
    debug_df = df.select([
        pl.col("product_type"),
        pl.col("product_type").map_elements(
            get_product_code,
            return_dtype=pl.String
        ).alias("type_code"),
        pl.col("l1_cluster").cast(pl.Int64),
        pl.col("l2_cluster").cast(pl.Int64),
        pl.col("l3_cluster").cast(pl.Int64)
    ])
    
    # Show a small sample
    print("Sample of debug DataFrame:")
    print(debug_df.head(5))
    
    # Get the hierarchy_codes DataFrame
    hierarchy_codes = debug_df.select([
        pl.col("type_code"),
        pl.col("l1_cluster").cast(pl.Int64),
        pl.col("l2_cluster").cast(pl.Int64),
        pl.col("l3_cluster").cast(pl.Int64)
    ])
    
    # Debug type_code values
    type_code_sample = hierarchy_codes.select("type_code").sample(5).to_series().to_list()
    print(f"Sample type_codes: {type_code_sample}")
    
    print("\n=== DEBUG: Creating final hierarchy codes ===")
    # Print the formula for hierarchy code
    print("Hierarchy code formula: type_code + '-' + formatted_l1_cluster + formatted_l2_cluster + '-' + formatted_l3_cluster")
    
    # Test the formula with a few examples
    test_rows = hierarchy_codes.head(3)
    for i, row in enumerate(test_rows.iter_rows(named=True)):
        type_code = row["type_code"]
        l1 = f"{int(row['l1_cluster']):02d}"
        l2 = f"{int(row['l2_cluster']):02d}"
        l3 = f"{int(row['l3_cluster']):02d}"
        expected = f"{type_code}-{l1}{l2}-{l3}"
        print(f"Row {i}: {type_code} -> {l1} -> {l2} -> {l3} = {expected}")
    
    # Now create the hierarchy codes
    hierarchy_codes = hierarchy_codes.with_columns([
        (
            pl.col("type_code") + "-" + 
            pl.col("l1_cluster").map_elements(lambda x: f"{int(x):02d}", return_dtype=pl.String) +
            pl.col("l2_cluster").map_elements(lambda x: f"{int(x):02d}", return_dtype=pl.String) + "-" +
            pl.col("l3_cluster").map_elements(lambda x: f"{int(x):02d}", return_dtype=pl.String)
        ).alias("hierarchy_code")
    ]).select("hierarchy_code")
    
    # Debug final hierarchy codes
    hier_code_sample = hierarchy_codes.sample(5).to_series().to_list()
    print(f"Sample hierarchy codes: {hier_code_sample}")
    
    return hierarchy_codes["hierarchy_code"]

def main():
    parser = argparse.ArgumentParser(description='Assign hierarchy codes to clustered data')
    parser.add_argument('--input_file', type=str, default='abo-dataset/with_clusters.parquet',
                        help='Path to the parquet file with clusters (default: abo-dataset/with_clusters.parquet)')
    parser.add_argument('--output_file', type=str, default='abo-dataset/with_hierarchy_codes.parquet',
                        help='Path to save the output parquet file (default: abo-dataset/with_hierarchy_codes.parquet)')
    parser.add_argument('--unmatched_csv', type=str, default='abo-dataset/unmatched_product_types.csv',
                        help='Path to save unmatched product types CSV (default: abo-dataset/unmatched_product_types.csv)')
    args = parser.parse_args()

    print(f"Reading clustered data from {args.input_file}")
    df = pl.read_parquet(args.input_file)
    print(f"Loaded {df.height} rows")
    
    # Debug check if product_type column exists
    print(f"DataFrame schema: {df.schema}")
    if "product_type" in df.columns:
        null_count = df.null_count()['product_type'] if 'product_type' in df.columns else 'N/A'
        print(f"product_type column exists. Null count: {null_count}")
        
        # Sample some product types
        print("Sample product types from DataFrame:")
        print(df.select("product_type").head(5))
    else:
        print("WARNING: product_type column does not exist!")

    unmatched_counter = defaultdict(int)
    print("Assigning hierarchy codes...")
    hierarchy_codes = generate_hierarchy_codes(df, unmatched_counter=unmatched_counter)

    print("Saving results with hierarchy codes...")
    df_with_codes = df.with_columns([
        pl.Series("hierarchy_code", hierarchy_codes)
    ])
    df_with_codes.write_parquet(args.output_file)
    print(f"Saved to {args.output_file}")

    print("Writing unmatched product types and their counts to CSV...")
    with open(args.unmatched_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["product_type", "count"])
        for pt, count in sorted(unmatched_counter.items(), key=lambda x: -x[1]):
            writer.writerow([pt, count])
    print(f"Wrote unmatched product types and their counts to {args.unmatched_csv}")

if __name__ == "__main__":
    main()
