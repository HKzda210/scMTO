# Topic-Guided Multi-View Deep Wasserstein Clustering for Single-cell Transcriptomic Analysis

Single-cell RNA sequencing (scRNA-seq) provides unprecedented resolution for genome-wide gene expression profiling at the single-cell level, unveiling intricate details of cellular transcriptomes. However, many existing deep clustering methods often overlook the rich multi-view information inherent in scRNA-seq data, leading to pseudo-variability and potential misinterpretation of cellular states. To address this issue, we introduce scMTO, a novel topic-guided deep multi-view Wasserstein clustering method that seamlessly integrates a deep **M**ulti-view autoencoder with **T**opic modeling and **O**ptimal transport theory to enhance scRNA-seq interpretation. scMTO employs a multi-view transcriptional encoder to capture comprehensive cell embeddings from diverse data perspectives, paired with a graph-sparse regularized topic model to extract key topic features directly from sparse data. Subsequently, guided by these topics, scMTO incorporates an optimal transport regularized topic decoder to align sparse cell topic patterns in the latent space, complemented by a topic-guided zero-inflated negative binomial (ZINB) model to encapsulate the global probability distribution. By jointly optimizing ZINB, the optimal transport, and the Kullback–Leibler divergence losses, scMTO achieves substantial improvements in clustering performance, preserving a latent structure enriched with multi-view information and crucial cell topic patterns. Our comprehensive evaluation on 12 diverse scRNA-seq datasets across multiple sequencing platforms demonstrated scMTO's superiority over existing state-of-the-art clustering methods.

![frame](frame.jpg)

## Installation:
```
$ conda create -n scmto python=3.9.19
$ conda activate scmto
$ pip install -r requirements.txt
```

## Usage
We provide demos showing the clustering performance of scMTO: 

- Single-cell RNA-seq data clustering analysis: [demo.ipynb](https://github.com/HKzda210/scMTO/blob/master/tutorial/demo.ipynb)
- Analysis of the large scale scRNA-seq dataset MRCA_BC: [demo_scale.ipynb](https://github.com/HKzda210/scMTO/blob/master/tutorial/demo.ipynb)

## Data Availability
The real data sets we used can be download [here](https://drive.google.com/drive/folders/1BIZxZNbouPtGf_cyu7vM44G5EcbxECeu). [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17346572.svg)](https://doi.org/10.5281/zenodo.17346572)

