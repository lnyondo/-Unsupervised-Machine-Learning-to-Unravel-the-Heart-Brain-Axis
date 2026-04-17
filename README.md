# Unsupervised Machine Learning to Unravel the Heart–Brain Axis

## Project Overview
This project applies unsupervised machine learning to uncover hidden relationships between cardiovascular disease (CVD) and Alzheimer’s disease (AD) using real-world electronic health record (EHR) data.

The study is based on the **heart–brain axis**, which describes how vascular, metabolic, and inflammatory processes contribute to neurodegeneration and cognitive decline.

## Objective
To identify **latent patient phenotypes** that capture interactions between cardiovascular, metabolic, and neurodegenerative conditions using a data-driven approach.


## Dataset
- Source: All of Us Research Program  
- Standard: OMOP Common Data Model  
- Features: Diagnosis-based binary indicators (CVD, metabolic, neurodegenerative conditions)

## Methodology
The project uses an unsupervised machine learning pipeline:

1. **TF-IDF Transformation**
   - Reduces dominance of common conditions (e.g., hypertension)

2. **Dimensionality Reduction (SVD)**
   - Compresses high-dimensional EHR data

3. **MiniBatch K-Means Clustering**
   - Identifies patient subgroups (phenotypes)

4. **Evaluation**
   - Silhouette Score
   - Cluster distribution
   - Clinical interpretability

---

## Results

### Optimal Clustering
- Best number of clusters: **k = 4**
- Silhouette score: ~0.15 (expected for clinical data)

### Key Findings
- One dominant “general population” cluster
- Smaller clusters represent higher-risk subgroups
- Clusters are primarily driven by **cardiovascular and metabolic conditions**

### Critical Insight
- **No distinct neurodegenerative cluster was identified**
- Dementia-related conditions are:
  - Low in prevalence
  - Distributed across clusters

👉 This indicates that:
> Neurodegeneration is embedded within cardiometabolic disease patterns rather than forming an independent phenotype.

---

## Heart–Brain Axis Interpretation
- Cardiovascular burden dominates all clusters  
- Metabolic conditions introduce variation  
- Neurodegenerative burden remains low and diffuse  

These findings support the concept that:
> Brain disease is strongly influenced by cardiovascular health.

---

## Clinical Implications
- Cardiovascular risk management may reduce dementia risk  
- Supports a **systems-level approach to healthcare**  
- Enables identification of high-risk patient groups  
- Demonstrates value of AI in precision medicine  
