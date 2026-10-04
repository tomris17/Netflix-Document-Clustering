# Netflix Text and Document Clustering Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains an unsupervised text clustering project designed to group Netflix movies and TV shows based on the semantic themes of their descriptions[cite: 13].

---

## Dataset Notice
*Note: The dataset (`Netflix_movies_and_tv_shows_clustering.csv`) used in this project[cite: 13] is publicly available on Kaggle.*

---

## Dataset Features
* **description**: Textual plot summary of movies and TV shows on Netflix[cite: 13].

---

## Project Workflow
1. **Text Preprocessing**: Dropping missing descriptions[cite: 13] and extracting features using `TfidfVectorizer` (with English stop words removal and a maximum feature limit of 1000)[cite: 13].
2. **Optimal Cluster Detection**: Leveraging Yellowbrick's `KElbowVisualizer` alongside `KMeans` to determine the ideal number of clusters[cite: 13].
3. **Model Training**: Fitting the `KMeans` clustering algorithm on the TF-IDF feature array[cite: 13].
4. **Model Persistence**: Exporting the trained model using `joblib` into `document_clustering_model.pkl`[cite: 13].
5. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/netflix-text-clustering.git](https://github.com/YOUR_USERNAME/netflix-text-clustering.git)
   cd netflix-text-clustering
