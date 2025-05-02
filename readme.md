### Project-6: ShopTalk - Conversational Shopping Assistant Adaptation Proposal

#### Project Adaptation: Hierarchy Code Ranking System

**Original Project Overview:**
ShopTalk is designed to revolutionize online shopping through AI-powered conversational interfaces that understand complex product queries and provide precise recommendations. The project focuses on natural language understanding, retrieval augmented generation (RAG), and real-time response generation.

**Proposed Adaptation:**
We propose adapting the ShopTalk framework to build a Hierarchy Code Ranking System that predicts appropriate hierarchy codes for insurance policies based on their attributes. This adaptation maintains the core technical challenges of the original project while applying them to a different domain with significant business impact.

**Motivation:**
My company will be building a system like this in the future and I would like to get a headstart on this and build a prototype system.

**Technical Alignment:**
Our adaptation will utilize the same Amazon Berkeley Objects (ABO) dataset as ShopTalk, but with a different mapping strategy:

- The RAG system will be adapted to retrieve similar historical policy assignments
- The ranking system will be implemented using XGBoost's learning-to-rank capabilities

**Key Components:**

1. **Data Processing Pipeline**

   - Load and preprocess ABO data
   - Generate synthetic hierarchy codes through hierarchical clustering

2. **Embedding Generation**

   - Create vector representations of policies
   - Combine text, categorical, and numerical features
   - Implement weighted embedding concatenation

3. **Hierarchical Clustering**

   - Implement 3-level hierarchical clustering
   - Generate synthetic hierarchy codes

4. **XGBoost Ranking Model**

   - Train model using learning-to-rank approach

5. **DAGSTER Pipeline**

   - Create data pipeline for training dataset upload
   - Implement web interface for training uploads, configuration management, and prediction submission
   - Build feedback loop for model improvement

**Deliverables:**

1. Extended ABO dataset with a new synthetic column 'hierarchy code'

   - Data preprocessing pipeline
   - Segment the data using hierarchical clustering to generate synthetic 'hierarchy codes' that relate to the ABO product attributes

2. Working prototype with:

   - Embedding generation system
   - XGBoost ranking model
   - Basic web interface for predictions

3. DAGSTER pipeline for new data sets:
   - Training dataset upload functionality (labels must be provided; system will not generate syntehtic 'hierarchy codes' for uploaded training data)
   - Prediction submission interface
   - Model retraining capabilities
