### Project-6: ShopTalk - Conversational Shopping Assistant Adaptation Proposal

#### Setup

Use the dockerfile to try the demo. To build the docker image, some files are git ignored or in git-lfs

```bash
# pull all git lfs files
git lfs pull

# run the data setup script
uv run abo-dataset/pre_process_data.py
```

#### Project Adaptation: Hierarchy Code Ranking System

**Original Project Overview:**
ShopTalk is designed to revolutionize online shopping through AI-powered conversational interfaces that understand complex product queries and provide precise recommendations. The project focuses on natural language understanding, retrieval augmented generation (RAG), and real-time response generation.

**Proposed Adaptation:**
We propose adapting the ShopTalk framework to build a Hierarchy Code Ranking System that predicts appropriate hierarchy codes for insurance policies based on their attributes. This adaptation maintains the core technical challenges of the original project while applying them to a different domain with significant business impact.

**Motivation:**
My company will need to build a system along these lines for policy -> hierarchy code predictions/ranking in the future. I'd love to get a head start on this as a side project. The ABO dataset suggested for ShopTalk project should be a decent analog for policy data.

**Technical Alignment:**
Our adaptation will utilize the same Amazon Berkeley Objects (ABO) dataset as ShopTalk, but with a different end goal/output:

- The RAG system will be adapted to retrieve similar historical policy assignments
- The ranking system will be implemented using XGBoost's learning-to-rank capabilities
- The output of the system will be ranked hierarchy code predictions instead of a back-and-forth chat experience

**Key Components:**

1. **Data Processing Pipeline**

   - Load and preprocess ABO data
   - Generate synthetic hierarchy codes through clustering

2. **Embedding Generation**

   - Create vector representations of products
   - Combine text, categorical, and numerical features

3. **Hierarchical Clustering**

   - Implement 3-level hierarchical clustering
   - Generate synthetic hierarchy codes

4. **XGBoost Ranking Model**

   - Train model using learning-to-rank approach

5. **Dagster Pipeline**

   - Create data pipeline for training dataset upload (labels must be provided; system will not generate synthetic hierarchy codes for uploaded data)
   - Implement web interface for training data uploading, model management and configuration, and inference submission
   - Build feedback loop for model improvement

**Deliverables:**

1. Data preprocessing pipeline

   - Extend the ABO dataset with new synthetic 'hierarchy code' column
   - Segment data using hierarchical clustering to generate hierarchy codes that relate to the product information

2. Working prototype with:

   - Embedding generation and vector database search system
   - XGBoost ranking model
   - Basic web interface for predictions

3. DAGSTER pipeline extension:

   - Training dataset upload functionality
   - Prediction submission interface
   - Model retraining capabilities
