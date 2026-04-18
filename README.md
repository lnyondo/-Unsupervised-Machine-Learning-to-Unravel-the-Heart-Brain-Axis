# Unsupervised Machine Learning to Unravel the Heart–Brain Axis

## Project Authors

 Kudakwashe Blessing Mukumbi

 Lusubilo Nyondo

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

## Link to Video
https://1drv.ms/v/c/53d0442c54396c33/IQDLm4w8okgYRqsUewwNgPkUAQ9-hHeebJaW60V6b-TVfiI?e=WE3wX8

[Unsupervised Machine Learning to Unravel the Heart–.pptx](https://github.com/user-attachments/files/26844884/Unsupervised.Machine.Learning.to.Unravel.the.Heart.pptx)

[Artificial Intelligence in Healthcare Project.pdf](https://github.com/user-attachments/files/26845477/Artificial.Intelligence.in.Healthcare.Project.pdf)


[Artificial Intelligence in Healthcare Project- MTU.ipynb](https://github.com/user-attachments/files/26845580/Artificial.Intelligence.in.Healthcare.Project-.MTU.ipynb)
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Heart–Brain Phenotype Clustering from De-identified Clinical Feature Matrices\n",
    "\n",
    "## Project overview\n",
    "This notebook presents a public reproducibility workflow for unsupervised clustering of heart–brain related clinical phenotypes using de-identified, precomputed feature matrices.\n",
    "\n",
    "## Important data access note\n",
    "This public notebook does **not** connect directly to the All of Us Researcher Workbench or query protected participant-level data. Instead, it starts from approved derivative inputs prepared within the secure environment and reviewed for dissemination compliance.\n",
    "\n",
    "## Public inputs expected\n",
    "- `X_tfidf.npz`: sparse TF-IDF feature matrix\n",
    "- `target_flags.csv`: de-identified patient-level condition flags and cluster-ready validation variables\n",
    "- `concept_map.csv`: concept ID to concept name mapping approved for public use\n",
    "\n",
    "## Outputs\n",
    "This notebook generates:\n",
    "- cluster evaluation summaries\n",
    "- cluster-level prevalence summaries\n",
    "- aggregate figures for publication or GitHub documentation\n",
    "\n",
    "No participant-level outputs are generated in the public version."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 0 — Environment Setup and Reproducibility\n",
    "# ---------------------------------------------------------\n",
    "# Import required libraries and define shared paths and\n",
    "# reproducibility settings used throughout the analysis.\n",
    "# =========================================================\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from pathlib import Path\n",
    "from scipy.sparse import load_npz\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "from sklearn.cluster import MiniBatchKMeans\n",
    "from sklearn.metrics import silhouette_score\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "np.random.seed(RANDOM_STATE)\n",
    "\n",
    "DATA_DIR = Path(\"data\")\n",
    "OUTPUT_DIR = Path(\"outputs\")\n",
    "OUTPUT_DIR.mkdir(exist_ok=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 1 — Load De-identified Inputs\n",
    "\n",
    "This analysis begins from preprocessed, de-identified datasets:\n",
    "\n",
    "- TF-IDF feature matrix (diagnostic features)\n",
    "- Clinical flag table (binary disease indicators)\n",
    "- Concept mapping table\n",
    "\n",
    "No participant-level raw data is accessed in this workflow."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 1A — Load public-safe derivative inputs\n",
    "# =========================================================\n",
    "\n",
    "X_tfidf = load_npz(DATA_DIR / \"X_tfidf.npz\")\n",
    "target_flags = pd.read_csv(DATA_DIR / \"target_flags.csv\")\n",
    "concept_map_df = pd.read_csv(DATA_DIR / \"concept_map.csv\")\n",
    "\n",
    "print(\"TF-IDF shape:\", X_tfidf.shape)\n",
    "print(\"Target flags shape:\", target_flags.shape)\n",
    "print(\"Concept map shape:\", concept_map_df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 1B — Validate required analysis columns\n",
    "# ---------------------------------------------------------\n",
    "# Confirm that the cluster-validation table contains the\n",
    "# expected patient identifier and disease indicator columns.\n",
    "# =========================================================\n",
    "\n",
    "required_cols = [\n",
    "    \"patient_id\", \"htn\", \"cad\", \"hf\", \"af\", \"stroke_tia\", \"pad\",\n",
    "    \"t2d\", \"hld\", \"ad\", \"mci\", \"dementia\", \"vascular_dementia\"\n",
    "]\n",
    "\n",
    "missing_cols = [col for col in required_cols if col not in target_flags.columns]\n",
    "if missing_cols:\n",
    "    raise ValueError(f\"Missing required columns in target_flags.csv: {missing_cols}\")\n",
    "\n",
    "print(\"All required target flag columns are present.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 2 — Feature Representation\n",
    "\n",
    "Diagnoses were aggregated into a patient-by-concept frequency matrix and transformed using TF-IDF weighting. This approach reduces the influence of highly prevalent conditions and increases the relative importance of more informative features, improving discrimination between patient phenotypes. The public workflow uses diagnosis-derived, de-identified feature matrices and cluster-validation flags prepared from structured EHR data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# PRIMARY ANALYSIS — Full Cohort Clustering\n",
    "# ========================================================="
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 3 — Dimensionality Reduction\n",
    "\n",
    "Given the high dimensionality of TF-IDF features, Truncated Singular Value Decomposition (SVD) is applied to obtain a compact representation suitable for clustering."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)\n",
    "X_reduced = svd.fit_transform(X_tfidf)\n",
    "\n",
    "print(\"Reduced feature space:\", X_reduced.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 4 — Cluster Number Selection\n",
    "\n",
    "We evaluate candidate cluster sizes using silhouette scores to identify an optimal balance between cohesion and separation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 4A — Evaluate candidate numbers of clusters\n",
    "# =========================================================\n",
    "\n",
    "k_values = [4, 5, 6, 7, 8, 10]\n",
    "results = []\n",
    "\n",
    "rng = np.random.RandomState(RANDOM_STATE)\n",
    "sample_idx = rng.choice(\n",
    "    X_reduced.shape[0],\n",
    "    size=min(20000, X_reduced.shape[0]),\n",
    "    replace=False\n",
    ")\n",
    "\n",
    "for k in k_values:\n",
    "    model = MiniBatchKMeans(\n",
    "        n_clusters=k,\n",
    "        random_state=RANDOM_STATE,\n",
    "        batch_size=4096,\n",
    "        n_init=10\n",
    "    )\n",
    "    labels = model.fit_predict(X_reduced)\n",
    "\n",
    "    score = silhouette_score(\n",
    "        X_reduced[sample_idx],\n",
    "        labels[sample_idx]\n",
    "    )\n",
    "\n",
    "    results.append((k, score))\n",
    "\n",
    "results_df = pd.DataFrame(results, columns=[\"k\", \"silhouette_score\"])\n",
    "results_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 4B — Plot silhouette score versus number of clusters\n",
    "# =========================================================\n",
    "\n",
    "plt.figure(figsize=(7, 4))\n",
    "plt.plot(results_df[\"k\"], results_df[\"silhouette_score\"], marker=\"o\")\n",
    "plt.xlabel(\"Number of clusters (k)\")\n",
    "plt.ylabel(\"Silhouette score\")\n",
    "plt.title(\"Cluster number selection (silhouette vs k)\")\n",
    "plt.tight_layout()\n",
    "plt.savefig(OUTPUT_DIR / \"fig1_silhouette_vs_k.png\", dpi=300, bbox_inches=\"tight\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 5 — Final Clustering Model\n",
    "\n",
    "Based on silhouette evaluation and interpretability, k = 4 was selected as the primary clustering solution for the full cohort."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 5A — Fit the full-cohort clustering model\n",
    "# =========================================================\n",
    "\n",
    "k = 4\n",
    "\n",
    "model = MiniBatchKMeans(\n",
    "    n_clusters=k,\n",
    "    random_state=RANDOM_STATE,\n",
    "    batch_size=4096,\n",
    "    n_init=10\n",
    ")\n",
    "clusters = model.fit_predict(X_reduced)\n",
    "\n",
    "cluster_labels = pd.DataFrame({\n",
    "    \"patient_id\": target_flags[\"patient_id\"],\n",
    "    \"cluster\": clusters\n",
    "})\n",
    "\n",
    "cluster_labels.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 5B — Cluster size distribution\n",
    "# =========================================================\n",
    "\n",
    "cluster_sizes = cluster_labels[\"cluster\"].value_counts().sort_index()\n",
    "\n",
    "plt.figure(figsize=(7, 4))\n",
    "plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values)\n",
    "plt.xlabel(\"Cluster\")\n",
    "plt.ylabel(\"Number of patients\")\n",
    "plt.title(\"Cluster sizes (full cohort)\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 5C — 2D embedding for visualization\n",
    "# =========================================================\n",
    "\n",
    "svd2 = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)\n",
    "X_2d = svd2.fit_transform(X_tfidf)\n",
    "\n",
    "rng = np.random.RandomState(RANDOM_STATE)\n",
    "plot_idx = rng.choice(X_2d.shape[0], size=min(20000, X_2d.shape[0]), replace=False)\n",
    "\n",
    "plt.figure(figsize=(7, 5))\n",
    "plt.scatter(\n",
    "    X_2d[plot_idx, 0],\n",
    "    X_2d[plot_idx, 1],\n",
    "    c=clusters[plot_idx],\n",
    "    s=6,\n",
    "    alpha=0.6\n",
    ")\n",
    "plt.xlabel(\"SVD component 1\")\n",
    "plt.ylabel(\"SVD component 2\")\n",
    "plt.title(\"2D embedding of patients colored by cluster\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 6 — Cluster Validation Using Clinical Prevalence\n",
    "\n",
    "Clusters are evaluated by examining the prevalence of major cardiovascular, metabolic, and neurodegenerative conditions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 6A — Merge cluster assignments with validation flags\n",
    "# =========================================================\n",
    "\n",
    "target_flags = target_flags.merge(cluster_labels, on=\"patient_id\")\n",
    "\n",
    "target_cols = [\n",
    "    \"htn\", \"cad\", \"hf\", \"af\", \"stroke_tia\", \"pad\",\n",
    "    \"t2d\", \"hld\", \"ad\", \"mci\", \"dementia\", \"vascular_dementia\"\n",
    "]\n",
    "\n",
    "cluster_prevalence = (\n",
    "    target_flags.groupby(\"cluster\")[target_cols]\n",
    "    .mean()\n",
    "    .round(3)\n",
    ")\n",
    "\n",
    "cluster_prevalence"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 7 — Composite Domain Scores\n",
    "\n",
    "To summarize disease burden, composite scores are constructed for:\n",
    "\n",
    "- Cardiovascular disease\n",
    "- Metabolic disease\n",
    "- Neurodegenerative disease"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 7A — Compute full-cohort composite domain scores\n",
    "# =========================================================\n",
    "\n",
    "target_flags[\"cvd_score\"] = target_flags[\n",
    "    [\"htn\", \"cad\", \"hf\", \"af\", \"stroke_tia\", \"pad\"]\n",
    "].sum(axis=1)\n",
    "\n",
    "target_flags[\"metabolic_score\"] = target_flags[\n",
    "    [\"t2d\", \"hld\"]\n",
    "].sum(axis=1)\n",
    "\n",
    "target_flags[\"neuro_score\"] = target_flags[\n",
    "    [\"ad\", \"mci\", \"dementia\", \"vascular_dementia\"]\n",
    "].sum(axis=1)\n",
    "\n",
    "cluster_scores = (\n",
    "    target_flags.groupby(\"cluster\")[[\"cvd_score\", \"metabolic_score\", \"neuro_score\"]]\n",
    "    .mean()\n",
    "    .round(3)\n",
    ")\n",
    "\n",
    "cluster_scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 8 — Phenotype Interpretation\n",
    "\n",
    "Clusters are assigned provisional phenotype labels based on dominant domain patterns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 8A — Assign provisional full-cohort phenotype labels\n",
    "# =========================================================\n",
    "\n",
    "def label_cluster_full(row):\n",
    "    if row[\"cvd_score\"] >= 3.0:\n",
    "        return \"High Cardiovascular Burden\"\n",
    "    elif row[\"metabolic_score\"] >= 1.5:\n",
    "        return \"Cardiometabolic\"\n",
    "    elif row[\"cvd_score\"] >= 2.0:\n",
    "        return \"Mixed Chronic Disease\"\n",
    "    else:\n",
    "        return \"General Population\"\n",
    "\n",
    "cluster_scores[\"label\"] = cluster_scores.apply(label_cluster_full, axis=1)\n",
    "cluster_scores"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 9 — Visualization of Cluster Profiles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 9A — Plot full-cohort heart–brain domain scores\n",
    "# =========================================================\n",
    "\n",
    "cluster_scores[[\"cvd_score\", \"metabolic_score\", \"neuro_score\"]].plot(\n",
    "    kind=\"bar\",\n",
    "    figsize=(8, 5)\n",
    ")\n",
    "plt.title(\"Heart–Brain Axis Across Clusters (Full Cohort)\")\n",
    "plt.ylabel(\"Score\")\n",
    "plt.xticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.savefig(OUTPUT_DIR / \"fig4_cluster_scores_full.png\", dpi=300, bbox_inches=\"tight\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# SECONDARY ANALYSIS — Focused Heart–Brain Subcohort\n",
    "# ========================================================="
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "## Secondary analysis: focused heart–brain subcohort\n",
    "\n",
    "The primary full-cohort clustering identified broad multimorbidity patterns but was dominated by common cardiometabolic conditions. To improve sensitivity to clinically relevant heart–brain phenotypes, a secondary clustering analysis was performed in a subcohort enriched for major cardiovascular and neurodegenerative conditions."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 10 — Focused Heart–Brain Subcohort\n",
    "\n",
    "To better capture clinically relevant patterns, a subcohort enriched for cardiovascular and neurodegenerative conditions is defined."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 10A — Define the focused heart–brain subcohort\n",
    "# =========================================================\n",
    "\n",
    "tf = target_flags.set_index(\"patient_id\")\n",
    "\n",
    "mask = (\n",
    "    (tf[\"dementia\"] == 1) |\n",
    "    (tf[\"ad\"] == 1) |\n",
    "    (tf[\"stroke_tia\"] == 1) |\n",
    "    (tf[\"hf\"] == 1) |\n",
    "    (tf[\"cad\"] == 1)\n",
    ")\n",
    "\n",
    "filtered_ids = tf.index[mask]\n",
    "\n",
    "print(\"Focused heart–brain subcohort size:\", len(filtered_ids))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 11 — Secondary Clustering in Focused Cohort\n",
    "\n",
    "Clustering is repeated in the restricted cohort to improve phenotype resolution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "row_idx = tf.index.get_indexer(filtered_ids)\n",
    "\n",
    "X_filtered = X_tfidf[row_idx]\n",
    "\n",
    "kmeans = MiniBatchKMeans(n_clusters=6, random_state=RANDOM_STATE)\n",
    "clusters_filtered = kmeans.fit_predict(X_filtered)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 11A — Fit the focused-cohort clustering model\n",
    "# =========================================================\n",
    "\n",
    "row_idx = tf.index.get_indexer(filtered_ids)\n",
    "X_filtered = X_tfidf[row_idx]\n",
    "\n",
    "kmeans = MiniBatchKMeans(\n",
    "    n_clusters=6,\n",
    "    random_state=RANDOM_STATE,\n",
    "    batch_size=4096,\n",
    "    n_init=10\n",
    ")\n",
    "clusters_filtered = kmeans.fit_predict(X_filtered)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 11B — Attach focused-cohort cluster labels\n",
    "# =========================================================\n",
    "\n",
    "tf_filtered = tf.loc[filtered_ids].copy()\n",
    "tf_filtered[\"cluster\"] = clusters_filtered\n",
    "\n",
    "print(tf_filtered[\"cluster\"].value_counts().sort_index())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 11C — Cluster prevalence in focused heart–brain cohort\n",
    "# =========================================================\n",
    "\n",
    "cluster_prevalence_filtered = (\n",
    "    tf_filtered.groupby(\"cluster\")[target_cols]\n",
    "    .mean()\n",
    "    .round(3)\n",
    ")\n",
    "\n",
    "cluster_prevalence_filtered"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 11D — Composite heart–brain domain scores\n",
    "# =========================================================\n",
    "\n",
    "tf_filtered[\"cvd_score\"] = tf_filtered[\n",
    "    [\"htn\", \"cad\", \"hf\", \"af\", \"stroke_tia\", \"pad\"]\n",
    "].sum(axis=1)\n",
    "\n",
    "tf_filtered[\"metabolic_score\"] = tf_filtered[\n",
    "    [\"t2d\", \"hld\"]\n",
    "].sum(axis=1)\n",
    "\n",
    "tf_filtered[\"neuro_score\"] = tf_filtered[\n",
    "    [\"ad\", \"mci\", \"dementia\", \"vascular_dementia\"]\n",
    "].sum(axis=1)\n",
    "\n",
    "cluster_scores_filtered = (\n",
    "    tf_filtered.groupby(\"cluster\")[[\"cvd_score\", \"metabolic_score\", \"neuro_score\"]]\n",
    "    .mean()\n",
    "    .round(3)\n",
    ")\n",
    "\n",
    "cluster_scores_filtered"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 11E — Focused-cohort phenotype labels\n",
    "# =========================================================\n",
    "\n",
    "cluster_label_map = {\n",
    "    0: \"Metabolic dominant\",\n",
    "    1: \"Cardiometabolic\",\n",
    "    2: \"General population\",\n",
    "    3: \"Moderate mixed\",\n",
    "    4: \"High CVD\",\n",
    "    5: \"Severe CVD\"\n",
    "}\n",
    "\n",
    "cluster_scores_filtered[\"phenotype_label\"] = (\n",
    "    cluster_scores_filtered.index.map(cluster_label_map)\n",
    ")\n",
    "\n",
    "cluster_scores_filtered"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 11F — Plot focused-cohort heart–brain domain scores\n",
    "# =========================================================\n",
    "\n",
    "ax = cluster_scores_filtered[[\"cvd_score\", \"metabolic_score\", \"neuro_score\"]].plot(\n",
    "    kind=\"bar\",\n",
    "    figsize=(8, 5)\n",
    ")\n",
    "ax.set_title(\"Heart–Brain Axis Across Clusters (Focused Cohort)\")\n",
    "ax.set_xlabel(\"Cluster\")\n",
    "ax.set_ylabel(\"Mean Composite Score\")\n",
    "plt.xticks(rotation=0)\n",
    "plt.tight_layout()\n",
    "plt.savefig(OUTPUT_DIR / \"fig5_cluster_scores_focused.png\", dpi=300, bbox_inches=\"tight\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## STEP 12 — Export Public Results\n",
    "\n",
    "Only aggregate-level outputs are exported to ensure compliance with data dissemination policies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# =========================================================\n",
    "# STEP 12A — Save aggregate outputs\n",
    "# =========================================================\n",
    "\n",
    "cluster_scores.to_csv(OUTPUT_DIR / \"cluster_scores_full.csv\")\n",
    "cluster_prevalence.to_csv(OUTPUT_DIR / \"cluster_prevalence_full.csv\")\n",
    "\n",
    "cluster_scores_filtered.to_csv(OUTPUT_DIR / \"cluster_scores_focused.csv\")\n",
    "cluster_prevalence_filtered.to_csv(OUTPUT_DIR / \"cluster_prevalence_focused.csv\")\n",
    "\n",
    "print(\"Saved public aggregate outputs to:\", OUTPUT_DIR.resolve())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.16"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
