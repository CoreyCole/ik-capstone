# Technical Specification: Hierarchy Code Ranking System

## 1. Introduction

This document outlines the technical specifications for implementing a hierarchy code ranking system using XGBoost's learning-to-rank capabilities with the Amazon Berkeley Objects (ABO) dataset as an analogous data source. The system will recommend payment relationships (hierarchy codes) for insurance policies by finding patterns in similar historical assignments.

## 2. Data Source

### 2.1 Amazon Berkeley Objects (ABO) Dataset

For prototype development, we'll use the ABO dataset which contains detailed product information that can be analogous to insurance policy data:

- **Size**: 147,702 product listings with 398,212 unique catalog images
- **Metadata**: Includes multilingual title, brand, model, year, product type, color, description, dimensions, weight, material, pattern, and style
- **Structure**: JSON-formatted product listings with nested attributes
- **Availability**: Publicly available at https://amazon-berkeley-objects.s3.amazonaws.com/index.html

### 2.2 Data Mapping

One could imagine the amazon products mapping to insurance products attributes. We won't actually change the ABO dataset attributes, but it should be fairly analogous to insurance policies.

| ABO Attribute   | Insurance Policy Analog |
| --------------- | ----------------------- |
| product_type    | Insurance policy type   |
| brand           | Business organization   |
| model_name      | Policy template         |
| dimensions      | Coverage limits         |
| material        | Coverage specifics      |
| price (derived) | Premium                 |
| item_id         | Policy ID/Policy Number |
| description     | Policy description      |

## 3. System Architecture

### 3.1 High-Level System Design

The system consists of the following components:

1. **Data Processing Pipeline**: Loads and preprocesses ABO data
2. **Embedding Generator**: Creates vector representations of products/policies
3. **Hierarchical Clustering Engine**: Groups similar products into a hierarchy
4. **XGBoost Ranking Model**: Ranks candidate hierarchy codes for new policies
5. **Evaluation Framework**: Measures model performance

### 3.2 Data Flow

1. Load raw ABO data → Preprocess → Create embeddings
2. Apply hierarchical clustering → Generate synthetic hierarchy codes
3. Split data into train/val/test → Create ranking examples
4. Train XGBoost model → Evaluate → Fine-tune
5. Deploy model for inference → Receive new policy → Recommend hierarchy codes

## 4. Implementation Details

### 4.1 Data Preprocessing

**Special Considerations**:

- Handle multilingual text fields using common language (English)
- Deal with nested JSON structures by flattening key attributes
- Normalize numerical values with varied units (cm, inches, etc.)

### 4.2 Embedding Generation

We'll use a combination of techniques to create embeddings:

1. **Text Embeddings**:

   - Use SentenceTransformer ('paraphrase-MiniLM-L6-v2') for product descriptions
   - Dimensionality: 384

2. **Categorical Features**:

   - LabelEncoder for each categorical field
   - One-hot encoding for low-cardinality fields

3. **Numerical Features**:

   - StandardScaler to normalize numerical attributes

4. **Combined Embedding**:
   - Weighted concatenation: text (60%), categorical (30%), numerical (10%)
   - Text embeddings: 384 dimensions (fixed)
   - Categorical features:
     - product_type: ~10-20 unique values
     - brand: ~1000+ unique values
     - material: ~50-100 unique values
     - pattern: ~20-30 unique values
     - style: ~20-30 unique values
     - color: ~20-30 unique values
   - Final dimensionality: ~1500-1600

### 4.3 Hierarchical Clustering

We'll implement a 3-level hierarchical clustering approach:

1. **Level 1**: Broad categories (~8 clusters)

   - Use AgglomerativeClustering with 'ward' linkage
   - Parameters: n_clusters=8, affinity='euclidean'

2. **Level 2**: Subcategories (~15 clusters)

   - Parameters: n_clusters=15, affinity='euclidean'

3. **Level 3**: Detailed groupings (~30 clusters)
   - Parameters: n_clusters=30, affinity='euclidean'

**Code Generation Logic**:

```python
def generate_hierarchy_code(product_type, l1_cluster, l2_cluster, l3_cluster):
    type_code = TYPE_MAPPING.get(product_type, 'GEN')
    return f"{type_code}-{l1_cluster:02d}{l2_cluster:02d}-{l3_cluster:02d}"
```

#### 4.3.2 Split data into train, test and validate

- Use clustering results to separate data into train, validation, and test sets using stratified sampling based on the cluster assignments
- All splits contain examples from every cluster
- The distribution of hierarchy codes is preserved across splits
- We can properly evaluate the model's ability to generalize

```python
train_df, val_df, test_df = split_stratified_by_hierarchy_code(
    df,
    hierarchy_codes,
    test_size=0.2,
    val_size=0.1
)
```

### 4.4 XGBoost Ranking Implementation

#### 4.4.1 Data Preparation for Ranking

```python
def create_ranking_data(embeddings, hierarchy_codes):
    X_rank = []
    y_rank = []
    qid = []

    # For each query (policy)
    for query_idx, query_embedding in enumerate(embeddings):
        # Find similar policies using embeddings
        # For each candidate hierarchy code, create a ranking example
        # Assign relevance scores based on similarity

    return X_rank, y_rank, qid
```

#### 4.4.2 XGBoost XRank Configuration

```python
ranker = xgb.XGBRanker(
    objective='rank:ndcg',        # NDCG optimization objective
    tree_method='hist',           # Fast histogram-based algorithm
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    lambdarank_pair_method='mean', # Pair construction strategy
    lambdarank_num_pair_per_sample=8 # Number of pairs per sample
)
```

#### 4.4.3 Handling Group Information

Group information (qid) is critical for LambdaMART algorithms in XGBoost:

```python
ranker.fit(
    X_train, y_train,
    qid=qid_train,            # Query group IDs
    eval_set=[(X_val, y_val)],
    eval_qid=[qid_val],       # Validation query group IDs
    verbose=True
)
```

### 4.5 Inference Process

For a new policy:

1. Create embedding representation
2. Find similar historical policies using vector similarity
3. Extract candidate hierarchy codes from similar policies
4. Prepare features for each policy-code pair
5. Rank candidates using trained XGBoost model
6. Calculate confidence scores based on similarity and ranking
7. Return top-k recommendations with confidence scores

```python
def recommend_hierarchy_codes(new_policy, model, top_k=5):
    # Convert policy to embedding
    # Find similar historical policies
    # Extract candidate hierarchy codes
    # Prepare features for ranking
    # Get predictions from model
    # Calculate confidence scores
    # Return top-k recommendations
```

## 5. Evaluation Framework

The system will be evaluated using the following metrics:

1. **Accuracy@k**: Percentage of times the correct code is in the top-k predictions

   ```python
   def accuracy_at_k(predictions, ground_truth, k=1):
       correct = 0
       for pred, truth in zip(predictions, ground_truth):
           if truth in pred[:k]:
               correct += 1
       return correct / len(predictions)
   ```

2. **Mean Reciprocal Rank (MRR)**: Average position of the correct code

   ```python
   def mrr(predictions, ground_truth):
       reciprocal_ranks = []
       for pred, truth in zip(predictions, ground_truth):
           if truth in pred:
               rank = pred.index(truth) + 1
               reciprocal_ranks.append(1.0 / rank)
           else:
               reciprocal_ranks.append(0.0)
       return sum(reciprocal_ranks) / len(reciprocal_ranks)
   ```

3. **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality
   ```python
   # Using sklearn's implementation
   from sklearn.metrics import ndcg_score
   ```

## 6. References

1. Amazon Berkeley Objects Dataset: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
2. XGBoost Learning to Rank Documentation: https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html
3. Hierarchical Clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
4. SentenceTransformers: https://www.sbert.net/

## 7. Appendix

### 7.1 Sample Data Structure

Sample ABO product listing:

```json
{
  "item_id": "B075X4QMX3",
  "domain_name": "amazon.com",
  "item_name": [
    {
      "language_tag": "en_US",
      "value": "Stone & Beam Westport Modern Nailhead Upholstered Sofa, 87\"W, Linen"
    }
  ],
  "product_type": [{ "value": "sofa" }],
  "brand": [{ "language_tag": "en_US", "value": "Stone & Beam" }],
  "material": [{ "language_tag": "en_US", "value": "linen" }],
  "item_dimensions": {
    "height": {
      "normalized_value": { "unit": "cm", "value": 86.36 },
      "value": 34,
      "unit": "inches"
    },
    "length": {
      "normalized_value": { "unit": "cm", "value": 220.98 },
      "value": 87,
      "unit": "inches"
    },
    "width": {
      "normalized_value": { "unit": "cm", "value": 99.06 },
      "value": 39,
      "unit": "inches"
    }
  }
}
```

### 7.2 Example Hierarchy Code

```
SEA-0102-05
```

Where:

- `SEA`: Seating category (from product_type mapping)
- `01`: Level 1 cluster ID
- `02`: Level 2 cluster ID
- `05`: Level 3 cluster ID
