# 🎬 SceneIt: A Multimodal Movie Recommendation System

This repository contains the code, datasets, and report for our OMSCS Deep Learning final project — a movie recommendation system that integrates multiple modalities including user ratings, metadata, plot descriptions, and poster images.

## 🧠 Overview

Inspired by the Deep Learning Recommendation Model (DLRM), our system combines:

- **User ratings** from [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
- **Structured metadata** from the IMDB 5000 dataset
- **Plot summaries** from the Wikipedia Movie Plots dataset
- **Poster images** from the Movie Genre from its Poster dataset

We trained a multimodal model that fuses these sources to predict how a user would rate a movie. Our model outperforms classical baselines like Singular Value Decomposition (SVD) and Collaborative Filtering (CF) in terms of RMSE on the MovieLens dataset.

## 📊 Results

- Achieved a **denormalized RMSE of 0.7042** on MovieLens 100k
- Mean prediction error: **0.0235**, median: **-0.0023**
- Outperformed baselines:
  - SVD: 1.301
  - User-based CF: 1.435
  - Item-based CF: 1.356
- Ablation insights:
  - Best unimodal input: **metadata** (RMSE = 0.871)
  - Best dual modality: **metadata + posters** (RMSE = 0.845)
  - Using all three modalities yielded **higher RMSE (0.864)**, indicating naive fusion has limitations

See [`paper.pdf`](./paper.pdf) for full details.

## 🏗️ Architecture

Our model includes:
- A **learned embedding** for user IDs
- A **pretrained BERT** model to encode plot summaries
- A **pretrained ResNet-50** feature extractor for poster images
- A **fully connected layer** for metadata
- A **fusion MLP** to combine all features and predict ratings

## 🧪 Experiments

We compare our model against:
- **SVD**
- **User-based Collaborative Filtering**
- **Item-based Collaborative Filtering**

We also run ablation experiments with different combinations of modalities (metadata, text, images) to measure their individual contributions.

## 📁 Repository Structure

```bash
.
├── charts/                 # Saved result plots (loss curves, RMSE, etc.)
├── clean_data/             # Cleaned datasets (metadata, text, images) and post-processed datasets
├── models/                 # For trained model checkpoints (empty) and data scripts 
├── notebooks/              # Training and analysis notebooks
├── posters/                # Poster images
├── predictions/            # Predictions on test set
├── raw_data/               # Raw datasets (metadata, text, images, user ratings)
├── environment.yml         # Conda environment file
├── main.ipynb              # Main notebook
├── paper.pdf               # Final project paper
├── requirements.txt        # Python dependencies
└── README.md               # This file
