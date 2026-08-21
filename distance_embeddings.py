import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"

def compute_distances_from_chroma() -> pd.DataFrame | None:
    """ Berechnet Cosinsus Distanz Matrix der Dokumente zueinander

    Returns:
        pd.DataFrame | None: Cosinsus distanzmatrix
    """    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )

  
    data = vector_store.get(include=["embeddings", "metadatas"])
    
    all_embeddings = data["embeddings"]
    all_metadatas = data["metadatas"]

    if all_embeddings is None:
        print("Keine Daten in ChromaDB gefunden.")
        return

    # Embeddings nach Quelldatei gruppieren
    doc_vectors = defaultdict(list)
    for emb, meta in zip(all_embeddings, all_metadatas):
        source = os.path.basename(meta.get("source", "Unbekannt"))
        doc_vectors[source].append(emb)

    doc_names = list(doc_vectors.keys())
    print(f"Gefundene Dokumente in ChromaDB ({len(doc_names)}): {doc_names}")

    # Mittelwert-Vektor pro Dokument berechnen
    mean_embeddings = []
    for name in doc_names:
        avg_vector = np.mean(doc_vectors[name], axis=0)
        mean_embeddings.append(avg_vector)

    # Cosinus-Distanzmatrix berechnen
    num_docs = len(doc_names)
    distance_matrix = np.zeros((num_docs, num_docs))

    for i in range(num_docs):
        for j in range(num_docs):
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                distance_matrix[i][j] = cosine(mean_embeddings[i], mean_embeddings[j])

    
    df = pd.DataFrame(distance_matrix, index=doc_names, columns=doc_names)
    print("\n--- Cosinus-Distanzmatrix ---")
    print(df.round(4))
    return df


def plot_mds(distance_matrix_df):
    # MDS initialisieren
    # dissimilarity="precomputed", da Distanzmatrix in embedings schon berechnet
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=2,
        normalized_stress="auto",
    )

    # 2D-Koordinaten aus den Distanzen berechnen
    coords = mds.fit_transform(distance_matrix_df.values)

    # Plot erstellen
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")

    plt.scatter(
        coords[:, 0], coords[:, 1], color="royalblue", s=200, zorder=3, alpha=0.8
    )

    # Dokumentennamen als Beschriftung hinzufügen
    for i, name in enumerate(distance_matrix_df.index):
        name = name.removesuffix('.pdf') 

        plt.annotate(
            name,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(10, 8),
            ha="left",
            fontsize=11,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "lightgray",
                "alpha": 0.8,
            },
        )

    plt.title("MDS-Projektion der Embedings", fontsize=14, pad=15)
    plt.xlabel("MDS-Dimension 1", fontsize=11)
    plt.ylabel("MDS-Dimension 2", fontsize=11)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()

    plt.show()


def plot_pca_documents():
   
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
    )

    data = vector_store.get(include=["embeddings", "metadatas"])
    all_embeddings = data["embeddings"]
    all_metadatas = data["metadatas"]

    if all_embeddings is None or len(all_embeddings) == 0:
        print("Keine Vektoren in ChromaDB gefunden.")
        return

    # 2. Embeddings nach Dokument gruppieren
    doc_vectors = defaultdict(list)
    for emb, meta in zip(all_embeddings, all_metadatas):
        source = os.path.basename(meta.get("source", "Unbekannt"))
        doc_vectors[source].append(emb)

    doc_names = list(doc_vectors.keys())

    # Mittelwert-Vektor pro Dokument berechnen
    mean_embeddings = np.array(
        [np.mean(doc_vectors[name], axis=0) for name in doc_names]
    )

    
    pca = PCA(n_components=2, random_state=1)
    coords_2d = pca.fit_transform(mean_embeddings)

    # Erklärung der Varianz 
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(
        f"Erklärte Varianz durch die ersten 2 Hauptkomponenten: {explained_variance:.2f}%"
    )

    
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")

    plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        color="crimson",
        s=200,
        zorder=3,
        alpha=0.8,
    )

    for i, name in enumerate(doc_names):
        name = name.removesuffix('.pdf') 
        plt.annotate(
            name,
            (coords_2d[i, 0], coords_2d[i, 1]),
            textcoords="offset points",
            xytext=(10, 8),
            ha="left",
            fontsize=11,
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "fc": "white",
                "ec": "lightgray",
                "alpha": 0.8,
            },
        )

    plt.title("PCA-Projektion der Embeddings", fontsize=14, pad=15)
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% Varianz)",
        fontsize=11,
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% Varianz)",
        fontsize=11,
    )
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    df = compute_distances_from_chroma()
    plot_mds(df)
    plot_pca_documents()