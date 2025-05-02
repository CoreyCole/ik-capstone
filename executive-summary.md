# Executive Summary: Hierarchy Code Ranking System

## Project Overview

This document outlines the implementation plan for a machine learning-based system to recommend "hierarchy codes" (payment relationships) for insurance policies. The system predicts the most appropriate hierarchy code for a new policy based on historical assignments.

## Business Context

Insurance policies require assignment to payment relationships (hierarchy codes) for paying commissions. Currently, this assignment is a manual process that:

- Is time-consuming and error-prone
- Lacks consistency across different agents
- Has no clear confidence measure for assignments
- Becomes increasingly complex as the number of possible codes grows

Our proposed system addresses these challenges by automatically recommending the most appropriate hierarchy codes with confidence scores, significantly improving efficiency and consistency.

## Technical Approach

### 1. Data Representation

We'll represent insurance policies as feature vectors combining:

- **Categorical Features**: Product types, carriers, agents
- **Numerical Features**: Coverage amounts, terms, limits
- **Text Features**: Policy descriptions and details

For the prototype, we'll use the Amazon Berkeley Objects (ABO) dataset as an analog, with product listings as an analog for insurance policies. We will extend the dataset to create a synthetic hierarchy code analog that our system will predict.

### 2. Clustering-Based Hierarchy Code Generation

Since we are adapting the ABO dataset as an analog for policy -> hierarchy mappings, we generate a synthetic hierarchy code column based on the ABO product attributes.

1. **Embedding Creation**:

   - Generate embeddings that capture semantics of product listings
   - Combine text embeddings, categorical features, and numerical features

2. **Hierarchical Agglomerative Clustering**:

   - Apply multi-level clustering to create a natural hierarchy
   - Level 1: Broad categories (~8 clusters)
   - Level 2: Subcategories (~15 clusters)
   - Level 3: Detailed groupings (~30 clusters)

3. **Hierarchy Code Assignment**:
   - Create structured codes following the pattern `[TYPE]-[L1][L2]-[L3]`
   - Example: `SEA-0304-12` where:
     - `SEA`: Product type group (e.g., Seating)
     - `0304`: Category and subcategory from clustering levels 1 and 2
     - `12`: Detailed level from clustering level 3

### 3. Ranking Model Development

We'll train an XGBoost ranking model to recommend hierarchy codes:

1. **Training Data Preparation**:

   - Split data into train/validation/test sets using stratified sampling
   - Ensure all clusters are represented across all splits
   - Create ranking examples where each policy is associated with candidate codes

2. **Model Training**:

   - Train using `rank:ndcg` objective function
   - Group samples by query (similar policies)
   - Optimize for ranking the correct hierarchy codes higher

3. **Confidence Scoring**:
   - Calculate confidence based on similarity to historical examples
   - Consider consistency of assignments among similar policies
   - Provide lower confidence scores for unusual or borderline cases

## Conclusion

The proposed Hierarchy Code Ranking System provides a robust, data-driven approach to automating payment relationship assignments. The system demonstrates a recommendation system that ranks predictions based on historical examples. The system's confidence scoring will help guide human reviewers, focusing their attention on cases requiring additional expertise.
