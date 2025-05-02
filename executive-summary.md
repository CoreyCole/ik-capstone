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

### 2. Clustering-Based Hierarchy Generation

We'll implement a structured, hierarchical clustering approach:

1. **Embedding Creation**:

   - Generate embeddings that capture semantic relationships between policies
   - Combine text embeddings (60%), categorical features (30%), and numerical features (10%)

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

## Implementation Timeline

| Phase                     | Duration | Key Deliverables                                        |
| ------------------------- | -------- | ------------------------------------------------------- |
| 1. Data Preparation       | 2 weeks  | Data preprocessing pipeline, embedding generation       |
| 2. Clustering Development | 3 weeks  | Hierarchical clustering implementation, code generation |
| 3. Ranking Model          | 4 weeks  | XGBoost ranking model, similarity search                |
| 4. Testing & Tuning       | 3 weeks  | Model evaluation, parameter optimization                |
| 5. Integration            | 2 weeks  | API development, documentation                          |

## Expected Benefits

1. **Efficiency**: Reduce manual assignment time by 70-80%
2. **Consistency**: Standardize hierarchy code assignments across the organization
3. **Confidence**: Provide quantitative measures of recommendation reliability
4. **Scalability**: Handle growing numbers of policies and hierarchy codes
5. **Learning**: Continuously improve from new assignments

## Evaluation Metrics

The system's performance will be measured using:

1. **Accuracy@1**: Percentage of cases where the top recommendation is correct
2. **Accuracy@3**: Percentage of cases where the correct code is in the top 3
3. **Accuracy@5**: Percentage of cases where the correct code is in the top 5
4. **Mean Reciprocal Rank (MRR)**: Average position of the correct code
5. **Confidence Correlation**: Relationship between confidence scores and accuracy

## Next Steps

1. Finalize data representation approach
2. Implement embedding generation pipeline
3. Develop and evaluate clustering models
4. Train initial ranking models
5. Create evaluation framework for ongoing assessment

## Conclusion

The proposed Hierarchy Code Ranking System provides a robust, data-driven approach to automating payment relationship assignments. By combining hierarchical clustering with modern ranking algorithms, we can significantly improve efficiency while maintaining high accuracy. The system's confidence scoring will help guide human reviewers, focusing their attention on cases requiring additional expertise.
