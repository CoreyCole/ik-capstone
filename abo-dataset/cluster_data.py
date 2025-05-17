import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import argparse
import pyarrow.parquet as pq
from collections import defaultdict
import csv

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

def apply_hierarchical_clustering_paginated(input_file, batch_size=10000, n_clusters_l1=8, n_clusters_l2=15, n_clusters_l3=30):
    """
    Apply hierarchical clustering at three levels using MiniBatchKMeans and batch processing.
    Args:
        input_file: Path to the parquet file
        batch_size: Number of rows per batch
        n_clusters_l1, n_clusters_l2, n_clusters_l3: Number of clusters at each level
    Returns:
        tuple: (l1_clusters, l2_clusters, l3_clusters) cluster assignments for all rows
    """
    import json
    import numpy as np
    from pathlib import Path

    # Open parquet file for reading in chunks
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows
    print(f"Total rows in file: {total_rows}")

    # First pass: Fit MiniBatchKMeans for L1
    l1_kmeans = MiniBatchKMeans(n_clusters=n_clusters_l1, random_state=42, batch_size=batch_size)
    print(f"Fitting L1 MiniBatchKMeans in batches of {batch_size}...")
    
    # Process batches for L1 fitting
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        embeddings = np.array([json.loads(e) for e in df_batch['embedding']])
        l1_kmeans.partial_fit(embeddings)

    # Second pass: Assign L1 clusters and collect embeddings for L2
    l1_clusters = np.zeros(total_rows, dtype=int)
    all_embeddings = []
    row_idx = 0
    print("Assigning L1 clusters and collecting embeddings...")
    
    # Process batches for L1 assignment
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df_batch = batch.to_pandas()
        embeddings = np.array([json.loads(e) for e in df_batch['embedding']])
        batch_l1 = l1_kmeans.predict(embeddings)
        l1_clusters[row_idx:row_idx+len(batch_l1)] = batch_l1
        all_embeddings.append(embeddings)
        row_idx += len(batch_l1)
        print(f"Processed {row_idx} rows for L1 assignment...")
    
    all_embeddings = np.vstack(all_embeddings)

    # L2 clustering: For each L1 cluster, fit MiniBatchKMeans on its subset
    l2_clusters = np.zeros(total_rows, dtype=int)
    cluster_offset = 0
    print("Fitting and assigning L2 clusters...")
    for l1_id in range(n_clusters_l1):
        mask = l1_clusters == l1_id
        subset_embeddings = all_embeddings[mask]
        if len(subset_embeddings) > 1:
            n_clusters_for_subset = max(2, int(n_clusters_l2 * len(subset_embeddings) / total_rows))
            l2_kmeans = MiniBatchKMeans(n_clusters=n_clusters_for_subset, random_state=42, batch_size=batch_size)
            l2_kmeans.fit(subset_embeddings)
            subset_labels = np.array(l2_kmeans.predict(subset_embeddings))
            l2_clusters[mask] = subset_labels.astype(int) + int(cluster_offset)
            cluster_offset += int(n_clusters_for_subset)
        else:
            l2_clusters[mask] = int(cluster_offset)
            cluster_offset += 1

    # L3 clustering: For each L2 cluster, fit MiniBatchKMeans on its subset
    l3_clusters = np.zeros(total_rows, dtype=int)
    cluster_offset = 0
    unique_l2_clusters = np.unique(l2_clusters)
    print("Fitting and assigning L3 clusters...")
    for l2_id in unique_l2_clusters:
        mask = l2_clusters == l2_id
        subset_embeddings = all_embeddings[mask]
        if len(subset_embeddings) > 1:
            n_clusters_for_subset = max(2, int(n_clusters_l3 * len(subset_embeddings) / total_rows))
            l3_kmeans = MiniBatchKMeans(n_clusters=n_clusters_for_subset, random_state=42, batch_size=batch_size)
            l3_kmeans.fit(subset_embeddings)
            subset_labels = np.array(l3_kmeans.predict(subset_embeddings))
            l3_clusters[mask] = subset_labels.astype(int) + int(cluster_offset)
            cluster_offset += int(n_clusters_for_subset)
        else:
            l3_clusters[mask] = int(cluster_offset)
            cluster_offset += 1

    print(f"Clustering complete: {len(np.unique(l1_clusters))} L1 clusters, "
          f"{len(np.unique(l2_clusters))} L2 clusters, {len(np.unique(l3_clusters))} L3 clusters")
    return l1_clusters, l2_clusters, l3_clusters

def normalize_type(pt):
    if pt is None:
        return ""
    return str(pt).strip().lower().replace(' ', '_')

def generate_hierarchy_codes(df, l1_clusters, l2_clusters, l3_clusters, unmatched_counter=None):
    """
    Generate hierarchy codes based on product type and cluster assignments
    
    Args:
        df: DataFrame with product data
        l1_clusters, l2_clusters, l3_clusters: Cluster assignments at each level
        unmatched_counter: defaultdict to count unmatched product types
    Returns:
        pl.Series: Hierarchy codes for each product
    """
    # Mapping from product type to code prefix with variations
    type_mapping = {
        "cellular_phone_case": "",
        "shoes": "",
        "grocery": "",
        "home": "",
        "boot": "",
        "sandal": "",
        "finering": "",
        "health_personal_care": "",
        "finenecklacebraceletanklet": "",
        "accessory": "",
        "office_products": "",
        "fineearring": "",
        "pet_supplies": "",
        "sporting_goods": "",
        "hardware_handle": "",
        "handbag": "",
        "hat": "",
        "earring": "",
        "outdoor_living": "",
        "wall_art": "",
        "janitorial_supply": "",
        "flat_sheet": "",
        "necklace": "",
        "beauty": "",
        "suitcase": "",
        "safety_supply": "",
        "ottoman": "",
        "biss": "",
        "drinking_cup": "",
        "food_service_supply": "",
        "ring": "",
        "herb": "",
        "auto_accessory": "",
        "headboard": "",
        "coffee": "",
        "nutritional_supplement": "",
        "planter": "",
        "baby_product": "",
        "hardware": "",
        "backpack": "",
        "skin_cleaning_agent": "",
        "luggage": "",
        "wireless_accessory": "",
        "vitamin": "",
        "cleaning_agent": "",
        "battery": "",
        "computer_add_on": "",
        "pantry": "",
        "wallet": "",
        "saute_fry_pan": "",
        "screen_protector": "",
        "skin_moisturizer": "",
        "legume": "",
        "waste_bag": "",
        "tea": "",
        "accessory_or_part_or_supply": "",
        "headphones": "",
        "bread": "",
        "instrument_parts_and_accessories": "",
        "jar": "",
        "charging_adapter": "",
        "plumbing_fixture": "",
        "flat_screen_display_mount": "",
        "bracelet": "",
        "abis_lawn_and_garden": "",
        "recreation_ball": "",
        "clock": "",
        "file_folder": "",
        "writing_instrument": "",
        "medication": "",
        "clothes_hanger": "",
        "dairy_based_drink": "",
        "label": "",
        "computer_component": "",
        "home_mirror": "",
        "thermoplastic_filament": "",
        "sauce": "",
        "bench": "",
        "towel_holder": "",
        "umbrella": "",
        "speakers": "",
        "fineother": "",
        "fruit": "",
        "dresser": "",
        "flatware": "",
        "exercise_mat": "",
        "basket": "",
        "laundry_detergent": "",
        "teaching_equipment": "",
        "dairy_based_cheese": "",
        "mechanical_components": "",
        "lock": "",
        "swatch": "",
        "protein_supplement_powder": "",
        "electric_fan": "",
        "wine": "",
        "noodle": "",
        "paper_product": "",
        "skin_cleaning_wipe": "",
        "toilet_paper_holder": "",
        "snack_mix": "",
        "wrench": "",
        "clothes_rack": "",
        "trash_can": "",
        "cargo_strap": "",
        "eyewear": "",
        "professional_healthcare": "",
        "dishware_plate": "",
        "dishware_place_setting": "",
        "envelope": "",
        "herbal_supplement": "",
        "dutch_ovens": "",
        "power_supplies_or_protection": "",
        "auto_part": "",
        "window_shade": "",
        "cookie": "",
        "breakfast_cereal": "",
        "cosmetic_case": "",
        "faucet": "",
        "laundry_hamper": "",
        "print_copy_paper": "",
        "writing_board": "",
        "keyboards": "",
        "body_positioner": "",
        "outdoor_recreation_product": "",
        "snack_chip_and_crisp": "",
        "juice_and_juice_drink": "",
        "muscle_roller": "",
        "multiport_hub": "",
        "drying_rack": "",
        "vase": "",
        "bottle": "",
        "area_deodorizer": "",
        "shampoo": "",
        "phone_accessory": "",
        "mineral_supplement": "",
        "earmuff": "",
        "lip_color": "",
        "abis_beauty": "",
        "water": "",
        "input_mouse": "",
        "sunscreen": "",
        "tote_bag": "",
        "camera_other_accessories": "",
        "fashionring": "",
        "safe": "",
        "candle": "",
        "dishware_bowl": "",
        "exercise_band": "",
        "cake": "",
        "mouthwash": "",
        "facial_tissue": "",
        "fashionnecklacebraceletanklet": "",
        "leotard": "",
        "drink_flavored": "",
        "technical_sport_shoe": "",
        "food_blender": "",
        "toy_figure": "",
        "packaged_soup_and_stew": "",
        "video_game_accessories": "",
        "cracker": "",
        "sticker_decal": "",
        "air_conditioner": "",
        "cosmetic_brush": "",
        "fashionearring": "",
        "camera_tripod": "",
        "conditioner": "",
        "ce_accessory": "",
        "freestanding_shelter": "",
        "baking_pan": "",
        "sugar": "",
        "orthopedic_brace": "",
        "cleaning_brush": "",
        "marking_pen": "",
        "auto_oil": "",
        "placemat": "",
        "vacuum_cleaner": "",
        "camera_bags_and_cases": "",
        "nut_butter": "",
        "screwdriver": "",
        "dishwasher_detergent": "",
        "salad_dressing": "",
        "bucket": "",
        "power_strip": "",
        "carrying_case_or_bag": "",
        "safety_glasses": "",
        "snack_food_bar": "",
        "bathwater_additive": "",
        "earplug": "",
        "drill_bits": "",
        "microphone": "",
        "watch": "",
        "stapler": "",
        "humidifier": "",
        "fashionother": "",
        "flash_memory": "",
        "antenna": "",
        "honey": "",
        "salwar_suit_set": "",
        "jewelry_set": "",
        "camera_lens_filters": "",
        "water_purification_unit": "",
        "electric_water_boiler": "",
        "dairy_based_ice_cream": "",
        "shoe_insert": "",
        "flour": "",
        "poultry": "",
        "personal_computer": "",
        "fruit_snack": "",
        "lab_supply": "",
        "meal_holder": "",
        "sound_and_recording_equipment": "",
        "nuts": "",
        "ink_or_toner": "",
        "pump_dispenser": "",
        "dairy_based_cream": "",
        "coffee_maker": "",
        "chocolate_candy": "",
        "wound_dressing": "",
        "scissors": "",
        "vehicle_interior_shade": "",
        "steering_wheel_cover": "",
        "blank_media": "",
        "doorstop": "",
        "pliers": "",
        "sunglasses": "",
        "milk_substitute": "",
        "showerhead": "",
        "sweatband": "",
        "eyelid_color": "",
        "ce_carrying_case_or_bag": "",
        "vinegar": "",
        "seals": "",
        "toothbrush": "",
        "inkjet_printer_ink": "",
        "broom": "",
        "shipping_box": "",
        "incense": "",
        "bottle_rack": "",
        "phone": "",
        "toilet_seat": "",
        "wheel": "",
        "popcorn": "",
        "childrens_costume": "",
        "hardware_hinge": "",
        "mouse_pad": "",
        "water_pump": "",
        "food_slicer": "",
        "wildlife_feeder": "",
        "meat": "",
        "paper_towel_holder": "",
        "toys_and_games": "",
        "video_projector": "",
        "sheet_pan": "",
        "vehicle_seat_cover": "",
        "can_opener": "",
        "candy": "",
        "computer_input_device": "",
        "pitcher": "",
        "electromechanical_gauge": "",
        "toaster": "",
        "bag": "",
        "fish": "",
        "weigh_scale": "",
        "computer": "",
        "ladder": "",
        "stationary_bicycle": "",
        "pastry": "",
        "tent": "",
        "card_stock": "",
        "microscopes": "",
        "razor_blade_cartridge": "",
        "candle_holder": "",
        "abis_home_improvement": "",
        "toy_slime": "",
        "baking_mix": "",
        "input_pen": "",
        "pretzel": "",
        "remote_control": "",
        "first_aid_kit": "",
        "casseroles": "",
        "charm": "",
        "art_and_craft_supply": "",
        "pot_holder": "",
        "utility_knife": "",
        "radio": "",
        "hair_iron": "",
        "landline_phone": "",
        "hair_styling_agent": "",
        "body_deodorant": "",
        "essential_oil": "",
        "hair_comb": "",
        "spirits": "",
        "building_material": "",
        "notebook_computer": "",
        "changing_pad_cover": "",
        "sleeping_bag": "",
        "water_flotation_device": "",
        "rice_cookers": "",
        "pressure_cooker": "",
        "washer_dryer_combination": "",
        "educational_supplies": "",
        "pet_apparel": "",
        "thermos": "",
        "toothbrush_holder": "",
        "juicer": "",
        "stringed_instruments": "",
        "hardware_tubing": "",
        "barbecue_grill": "",
        "personal_pill_dispenser": "",
        "hair_removal_agent": "",
        "food_processor": "",
        "multitool": "",
        "pet_pest_control": "",
        "food_mixer": "",
        "animal_litter": "",
        "vest": "",
        "skin_exfoliant": "",
        "thermometer": "",
        "astringent_substance": "",
        "boxing_glove": "",
        "hair_cleaner_conditioner": "",
        "gps_or_navigation_system": "",
        "barbell": "",
        "picture_frame": "",
        "camcorder": "",
        "nail_polish": "",
        "food_preparation_mold": "",
        "rice_mix": "",
        "puzzles": "",
        "microwave_oven": "",
        "massager": "",
        "networking_device": "",
        "self_stick_note": "",
        "manual_shaving_razor": "",
        "bookend": "",
        "camera_support": "",
        "fitness_bench": "",
        "license_plate_attachment": "",
        "fountain": "",
        "dairy_based_butter": "",
        "ice_chest": "",
        "valve": "",
        "system_power_device": "",
        "pacifier": "",
        "binocular": "",
        "tooth_cleaning_agent": "",
        "fishing_equipment": "",
        "face_shaping_makeup": "",
        "gps_or_navigation_accessory": "",
        "golf_club_bag": "",
        "auto_chemical": "",
        "television": "",
        "non_dairy_cream": "",
        "dehumidifier": "",
        "dietary_supplements": "",
        "jerky": "",
        "baking_cup": "",
        "toy_building_block": "",
        "measuring_cup": "",
        "lip_balm": "",
        "abis_pet_products": "",
        "skateboard": "",
        "mascara": "",
        "transport_rack": "",
        "countertop_oven": "",
        "flash_drive": "",
        "power_converter": "",
        "caddy": "",
        "vehicle_mirror": "",
        "refrigerator": "",
        "eyebrow_color": "",
        "vacuum_sealer_machine": "",
        "baking_paper": "",
        "drinking_straw": "",
        "hair_coloring_agent": "",
        "sugar_candy": "",
        "rowing_machine": "",
        "badge_holder": "",
        "baby_bottle": "",
        "receiver_or_amplifier": "",
        "abis_book": "",
        "litter_box": "",
        "drink_coaster": "",
        "sugar_substitute": "",
        "leavening_agent": "",
        "body_lubricant": "",
        "bread_making_machine": "",
        "roasting_pan": "",
        "flavored_drink_concentrate": "",
        "figurine": "",
        "overalls": "",
        "saw_blade": "",
        "paint_brush": "",
        "fastener_drive_bit": "",
        "percussion_instruments": "",
        "massage_stick": "",
        "camera_flash": "",
        "network_interface_controller_adapter": "",
        "bottle_opener": "",
        "printer": "",
        "shellfish": "",
        "hair_brush": "",
        "artificial_tree": "",
        "pet_toy": "",
        "fuel_pump": "",
        "two_way_radio": "",
        "punching_bag": "",
        "hairband": "",
        "countertop_burner": "",
        "calculator": "",
        "deep_fryer": "",
        "wallpaper": "",
        "utility_cart_wagon": "",
        "computer_input_device_accessory": "",
        "power_bank": "",
        "wheel_cutter": "",
        "ice_cube_tray": "",
        "computer_speaker": "",
        "dishwasher": "",
        "cookie_cutter": "",
        "telescope": "",
        "necktie": "",
        "walking_stick": "",
        "knife_block_set": "",
        "air_fryer": "",
        "security_camera": "",
        "air_compressor": "",
        "track_suit": "",
        "air_pump": "",
        "infant_toddler_car_seat": "",
        "protein_drink": "",
        "slow_cooker": "",
        "non_riding_toy_vehicle": "",
        "writing_paper": "",
        "audio_or_video": "",
        "jewelry": "",
        "abis_video_games": "",
        "surveilance_systems": "",
        "pinboard": "",
        "guitars": "",
        "animal_collar": "",
        "vine": "",
        "room_divider": "",
        "sport_racket": "",
        "scanner": "",
        "suspender": "",
        "monitor": "",
        "car_audio_or_theater": "",
        "artificial_plant": "",
        "video_device": "",
        "air_purifier": "",
        "treadmill": "",
        "memory_reader": "",
        "dvd_player_or_recorder": "",
        "digital_device_3": "",
        "vehicle_safety_camera": "",
        "blood_pressure_monitor": "",
        "vehicle_scan_tool": "",
        "golf_club": "",
        "sculpture": "",
        "cellular_phone": "",
        "sleep_mask": "",
        "cooking_oven": "",
        "agricultural_supplies": "",
        "amazon_book_reader_accessory": "",
        "dinnerware": "",
        "cycling_equipment": "",
        "skin_treatment_mask": "",
        "cutting_board": "",
        "stroller": "",
        "thickening_agent": "",
        "waist_cincher": "",
        "kick_scooter": "",
        "lehenga_choli_set": "",
        "av_receiver": "",
        "cosmetic_powder": "",
        "shovel_spade": "",
        "computer_cooling_device": "",
        "bakeware": "",
        "skin_foundation_concealer": "",
        "swing": "",
        "condiment": "",
        "robotic_vacuum_cleaner": "",
        "timer": "",
        "game_dice": "",
        "otc_medication": "",
        "sous_vide_machine": "",
        "food_dehydrator": "",
        "networking_router": "",
        "powersports_vehicle_part": "",
        "outbuilding": "",
        "terminal_block": "",
        "vivarium": "",
        "drill": "",
        "fireplace": "",
        "garlic_press": "",
        "wireless_locked_phone": "",
    }
    
    # Default code for unknown product types
    default_code = "GEN"
    
    # Normalize mapping keys
    normalized_type_mapping = {normalize_type(k): v for k, v in type_mapping.items()}
    
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
    
    def get_product_code(product_type):
        pt_norm = normalize_type(product_type)
        # Try exact match
        if pt_norm in normalized_type_mapping:
            return normalized_type_mapping[pt_norm]
        # Try partial match
        for key, code in normalized_type_mapping.items():
            if key in pt_norm or pt_norm in key:
                return code
        # Try word-based matching
        for word in pt_norm.split('_'):
            if word in normalized_type_mapping:
                return normalized_type_mapping[word]
        if unmatched_counter is not None:
            unmatched_counter[pt_norm] += 1
        return default_code
    
    # Generate hierarchy codes using Polars expressions
    hierarchy_codes = df_with_clusters.select([
        pl.col("product_type").map_elements(
            get_product_code,
            return_dtype=pl.String
        ).alias("type_code"),
        pl.col("l1_cluster").cast(pl.Int64),
        pl.col("l2_cluster").cast(pl.Int64),
        pl.col("l3_cluster").cast(pl.Int64)
    ]).with_columns([
        (
            pl.col("type_code") + "-" + 
            pl.col("l1_cluster").map_elements(lambda x: f"{int(x):02d}", return_dtype=pl.String) +
            pl.col("l2_cluster").map_elements(lambda x: f"{int(x):02d}", return_dtype=pl.String) + "-" +
            pl.col("l3_cluster").map_elements(lambda x: f"{int(x):02d}", return_dtype=pl.String)
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
                .agg(pl.len().alias('count'))
                .sort('count', descending=True)
                .limit(5)
            )
            
            print(f"\nCluster {cluster_id} ({cluster_items.height} items):")
            print("  Top product types:")
            for row in product_counts.iter_rows(named=True):
                print(f"    - {row['product_type']}: {row['count']} items")

def analyze_hierarchy_distribution(parquet_file):
    """
    Analyze the distribution of hierarchy codes in the parquet file
    
    Args:
        parquet_file: Path to the parquet file with hierarchy codes
    """
    print("\nAnalyzing hierarchy code distribution...")
    
    # Read the parquet file
    df = pl.read_parquet(parquet_file)
    
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cluster data with flexible parameters')
    parser.add_argument('--input_file', type=str, default='combined_listings_with_embeddings.parquet',
                        help='Path to the combined dataset with embeddings (default: combined_listings_with_embeddings.parquet)')
    parser.add_argument('--n_clusters_l1', type=int, default=10, help='Number of clusters at level 1 (default: 10)')
    parser.add_argument('--n_clusters_l2', type=int, default=20, help='Number of clusters at level 2 (default: 20)')
    parser.add_argument('--n_clusters_l3', type=int, default=40, help='Number of clusters at level 3 (default: 40)')
    parser.add_argument('--batch_size', type=int, default=20000, help='Batch size for processing (default: 5000)')
    args = parser.parse_args()
    
    # Set up paths
    base_dir = Path(__file__).parent
    input_file = base_dir / args.input_file
    output_dir = base_dir
    vis_dir = base_dir / "visualizations"

    # Print selected input file
    print(f"Loading combined dataset from {input_file}")
    
    # Apply hierarchical clustering with batch processing
    print("Starting hierarchical clustering...")
    l1_clusters, l2_clusters, l3_clusters = apply_hierarchical_clustering_paginated(
        input_file,
        batch_size=args.batch_size,
        n_clusters_l1=args.n_clusters_l1,
        n_clusters_l2=args.n_clusters_l2,
        n_clusters_l3=args.n_clusters_l3
    )

    # Extract and print unique product types before generating hierarchy codes
    print("Extracting unique product types for mapping...")
    df_types = pl.read_parquet(input_file, columns=["product_type"])
    unique_types = df_types["product_type"].unique().to_list()
    print("Unique product types (sample):")
    print(f"Total unique product types: {len(unique_types)}")
    # Optionally, save to a file for easier mapping
    with open("abo-dataset/unique_product_types.txt", "w", encoding="utf-8") as f:
        for pt in unique_types:
            f.write(f"{pt}\n")

    # Process data in batches to generate hierarchy codes
    print("Generating hierarchy codes...")
    hierarchy_codes = []
    unmatched_counter = defaultdict(int)
    parquet_file = pq.ParquetFile(input_file)
    row_idx = 0
    
    for batch in parquet_file.iter_batches(batch_size=args.batch_size):
        df_batch = batch.to_pandas()
        df_batch_pl = pl.from_pandas(df_batch)
        
        # Generate hierarchy codes for this batch
        batch_codes = generate_hierarchy_codes(
            df_batch_pl,
            l1_clusters[row_idx:row_idx + len(df_batch)],
            l2_clusters[row_idx:row_idx + len(df_batch)],
            l3_clusters[row_idx:row_idx + len(df_batch)],
            unmatched_counter=unmatched_counter
        )
        hierarchy_codes.extend(batch_codes)
        
        row_idx += len(df_batch)
        print(f"Processed {row_idx} rows for hierarchy codes...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Save the full DataFrame with hierarchy codes in batches
    print("Saving results...")
    row_idx = 0
    output_file = output_dir / "with_clusters.parquet"
    
    # Collect all batches into a list
    all_batches = []
    for batch in parquet_file.iter_batches(batch_size=args.batch_size):
        df_batch = batch.to_pandas()
        df_batch_pl = pl.from_pandas(df_batch)
        
        # Add clusters and hierarchy codes
        df_batch_with_clusters = df_batch_pl.with_columns([
            pl.Series("l1_cluster", l1_clusters[row_idx:row_idx + len(df_batch)]),
            pl.Series("l2_cluster", l2_clusters[row_idx:row_idx + len(df_batch)]),
            pl.Series("l3_cluster", l3_clusters[row_idx:row_idx + len(df_batch)]),
            pl.Series("hierarchy_code", hierarchy_codes[row_idx:row_idx + len(df_batch)])
        ])
        
        all_batches.append(df_batch_with_clusters)
        row_idx += len(df_batch)
        print(f"Processed {row_idx} rows...")
    
    # Concatenate all batches and write to a single file
    final_df = pl.concat(all_batches)
    final_df.write_parquet(str(output_file))
    print(f"Processing complete! Results saved to {output_file}")
    
    # Analyze the distribution of hierarchy codes
    analyze_hierarchy_distribution(output_file)
