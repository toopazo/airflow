"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# from sklearn.datasets import make_blobs


class Silhouette:
    def __init__(
        self, labels: np.ndarray, barycenters: np.ndarray, vectors: np.ndarray
    ):
        # Generating the sample data from make_blobs
        # This particular setting has one distinct cluster and 3 clusters placed close
        # together.
        # X, y = make_blobs(
        #     n_samples=500,
        #     n_features=2,
        #     centers=4,
        #     cluster_std=1,
        #     center_box=(-10.0, 10.0),
        #     shuffle=True,
        #     random_state=1,
        # )  # For reproducibility

        self.labels: np.ndarray = labels
        self.barycenters: np.ndarray = barycenters
        self.vectors: np.ndarray = vectors

        self.n_clusters = len(set(self.labels))

        shape0 = self.vectors.shape
        self.n_samples = shape0[0]
        self.n_features = shape0[1]
        print(f"Num of vectors {self.n_samples} Num of features {self.n_features}")

    def eval(self, output_path: Path):

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.vectors) + (self.n_clusters + 1) * 10])

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(self.vectors, self.labels)
        print(
            "For n_clusters =",
            self.n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.vectors, self.labels)

        y_lower = 10
        for i in range(self.n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[self.labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / self.n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(self.labels.astype(float) / self.n_clusters)
        ax2.scatter(
            self.vectors[:, 0],
            self.vectors[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        centers = self.barycenters
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette: n_clusters = {self.n_clusters}, silhouette_avg={round(silhouette_avg, 4)}",
            fontsize=14,
            fontweight="bold",
        )

        # plt.show()
        fig.savefig(output_path)

        return sample_silhouette_values

    def fit_and_eval(self, n_clusters: int, output_path: Path):

        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.vectors) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(self.vectors)
        print(f"cluster_labels {cluster_labels} len {len(cluster_labels)}")

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(self.vectors, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.vectors, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            self.vectors[:, 0],
            self.vectors[:, 1],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # print(f"centers.shape {centers.shape}")
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            # "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            # % n_clusters,
            f"Silhouette: n_clusters = {n_clusters}, silhouette_avg={round(silhouette_avg, 4)}",
            fontsize=14,
            fontweight="bold",
        )

        # plt.show()
        fig.savefig(output_path)


class SklearnNN:
    def __init__(
        self, labels: np.ndarray, barycenters: np.ndarray, vectors: np.ndarray
    ):

        # samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
        # neigh_samples = []
        # neigh_names = []
        # for j, vectors in enumerate(clu_vectors_list):
        #     cluster_name = clu_name_list[j]
        #     shape0 = vectors.shape
        #     sdim = shape0[0]
        #     # vdim = shape0[1]
        #     for i in range(0, sdim):
        #         vector_i = vectors[i, :]
        #         neigh_samples.append(deepcopy(vector_i))
        #         neigh_names.append(cluster_name)

        neigh = NearestNeighbors(n_neighbors=5, algorithm="brute")
        neigh.fit(vectors)

        self.sklearn_nn = neigh
        self.labels = labels
        self.barycenters = barycenters
        self.vectors = vectors  # neigh_samples

        self.n_clusters = len(set(self.labels))

    # def knn_init(self, clu_name_list: list, clu_vectors_list: list):

    #     # samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
    #     neigh_samples = []
    #     neigh_index = []
    #     for j, vectors in enumerate(clu_vectors_list):
    #         cluster_name = clu_name_list[j]
    #         shape0 = vectors.shape
    #         sdim = shape0[0]
    #         # vdim = shape0[1]
    #         for i in range(0, sdim):
    #             vector_i = vectors[i, :]
    #             neigh_samples.append(deepcopy(vector_i))
    #             neigh_index.append(cluster_name)

    #     neigh = NearestNeighbors(n_neighbors=5)
    #     neigh.fit(neigh_samples)
    #     # NearestNeighbors(...)

    #     # neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
    #     return neigh, neigh_samples, neigh_index

    def nearest_neighbors(
        self, q_vector: np.ndarray, q_label: int, barycenters: np.ndarray
    ):
        nn_indexes, nn_vectors, nn_distances, nn_labels = self.run_fit(
            q_vector=q_vector, q_label=q_label
        )
        nn_distances_round = [round(float(e), 8) for e in nn_distances]

        print("nearest_neighbors")
        print(f"  nn_indexes            {[int(e) for e in nn_indexes]}")
        print(f"  nn_vectors shape      {nn_vectors.shape}")
        print(f"  nn_distances          {nn_distances_round}")
        print(f"  nn_labels             {[int(e) for e in nn_labels]}")

        for clu_i, bary in enumerate(barycenters):
            dist_to_bary = np.linalg.norm(q_vector - bary)
            print(f"  dist to barycenter {clu_i}  {round(dist_to_bary, 4)}")

    def run_fit(self, q_vector: np.ndarray, q_label: int) -> tuple:
        nn_distances, nn_indexes = self.sklearn_nn.kneighbors(
            [q_vector], return_distance=True
        )
        nn_distances = np.array(nn_distances).squeeze()
        nn_indexes = np.array(nn_indexes).squeeze()

        nn_vectors: list[np.ndarray] = []
        neigh_distances = []
        nn_labels = []
        for ix in nn_indexes:
            neigh_vector = self.vectors[ix]
            nn_vectors.append(neigh_vector)
            dist = np.linalg.norm(neigh_vector - q_vector)
            neigh_distances.append(float(dist))
            nn_labels.append(self.labels[ix])

        nn_vectors = np.array(nn_vectors)

        # print(f"query vector shape  {q_vector.shape}")
        # print(f"knn distance        {neigh_dist}")
        # print(f"knn index           {neigh_ind}")
        # print(f"knn neighbors shape {neigh_neighbors.shape}")
        # print(f"knn distances       {neigh_distances}")
        # print(f"knn names           {neigh_names}")

        return nn_indexes, nn_vectors, nn_distances, nn_labels

    def tsne(self, output_path: Path):
        vectors_2d = TSNE(
            n_components=2, learning_rate="auto", init="random", perplexity=3
        ).fit_transform(self.vectors)

        fig, ax1 = plt.subplots()
        fig.set_size_inches(18, 7)

        cmap = mpl.colormaps["plasma"]  # colormaps['viridis']
        colors = cmap(np.linspace(0, 1, self.n_clusters))
        print(colors.shape)
        print(len(self.labels))
        for i, lb in enumerate(self.labels):
            # lb = self.labels[i]
            cl = colors[lb]
            vx = vectors_2d[i, 0]
            vy = vectors_2d[i, 1]

            ax1.scatter(
                vx,  # vectors_2d[:, 0],
                vy,  # vectors_2d[:, 1],
                marker=".",
                s=30 * 5,
                lw=0,
                alpha=0.7,
                c=cl,
                edgecolor="k",
            )

        # plt.show()
        fig.savefig(output_path)

        return vectors_2d

    def pca(self, output_path: Path):
        # Supongamos que tienes tus datos en una matriz llamada `datos` de forma (num_muestras, 512)
        # Por ejemplo:
        # datos = np.random.rand(100, 512)

        # Paso 1: Reducir la dimensionalidad a 2D
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(self.vectors)

        # Paso 2: Graficar
        # plt.figure(figsize=(10, 7))
        fig, ax1 = plt.subplots()
        fig.set_size_inches(18, 7)

        # ax1.scatter(
        #     vectors_2d[:, 0],
        #     vectors_2d[:, 1],
        #     c="blue",
        #     edgecolor="k",
        #     alpha=0.7,
        # )
        cmap = mpl.colormaps["plasma"]  # colormaps['viridis']
        colors = cmap(np.linspace(0, 1, self.n_clusters))
        print(colors.shape)
        print(len(self.labels))
        for i, lb in enumerate(self.labels):
            # lb = self.labels[i]
            cl = colors[lb]
            vx = vectors_2d[i, 0]
            vy = vectors_2d[i, 1]

            ax1.scatter(
                vx,  # vectors_2d[:, 0],
                vy,  # vectors_2d[:, 1],
                marker=".",
                s=30 * 5,
                lw=0,
                alpha=0.7,
                c=cl,
                edgecolor="k",
            )

        plt.title("Visualización de Datos Reducidos con PCA")
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.grid(True)
        # plt.show()
        fig.savefig(output_path)
