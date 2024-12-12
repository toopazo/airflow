"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from copy import deepcopy

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors

# from sklearn.datasets import make_blobs


class Silhouette:
    def __init__(self, vectors: np.ndarray):
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

        shape0 = vectors.shape
        self.sdim = shape0[0]
        self.vdim = shape0[1]
        print(f"Num of vectors {self.sdim} Num of features {self.vdim}")

        self.vectors = vectors

    def eval(
        self,
        n_clusters,
        cluster_labels: np.ndarray,
        barycenters: np.ndarray,
        output_path: Path,
    ):
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
        centers = barycenters
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
            "Silhouette analysis on sample data with n_clusters = %d" % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        # plt.show()
        fig.savefig(output_path)

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
        print(f"centers.shape {centers.shape}")
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
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

        # plt.show()
        fig.savefig(output_path)


class KnnExtended:
    def __init__(
        self, clu_name_list: list, clu_barycenter_list: list, clu_vectors_list: list
    ):

        # samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
        neigh_samples = []
        neigh_names = []
        for j, vectors in enumerate(clu_vectors_list):
            cluster_name = clu_name_list[j]
            shape0 = vectors.shape
            sdim = shape0[0]
            # vdim = shape0[1]
            for i in range(0, sdim):
                vector_i = vectors[i, :]
                neigh_samples.append(deepcopy(vector_i))
                neigh_names.append(cluster_name)

        neigh = NearestNeighbors(n_neighbors=5, algorithm="brute")
        neigh.fit(neigh_samples)

        self.neigh = neigh
        self.neigh_samples = neigh_samples
        self.neigh_names = neigh_names

    def knn_init(self, clu_name_list: list, clu_vectors_list: list):

        # samples = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
        neigh_samples = []
        neigh_index = []
        for j, vectors in enumerate(clu_vectors_list):
            cluster_name = clu_name_list[j]
            shape0 = vectors.shape
            sdim = shape0[0]
            # vdim = shape0[1]
            for i in range(0, sdim):
                vector_i = vectors[i, :]
                neigh_samples.append(deepcopy(vector_i))
                neigh_index.append(cluster_name)

        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(neigh_samples)
        # NearestNeighbors(...)

        # neigh.kneighbors([[0, 0, 1.3]], 2, return_distance=False)
        return neigh, neigh_samples, neigh_index

    def run_fit(self, q_vector: np.ndarray):
        neigh_dist, neigh_ind = self.neigh.kneighbors([q_vector], return_distance=True)
        # knn = np.array(knn)

        neigh_neighbors: list[np.ndarray] = []
        neigh_distances = []
        neigh_names = []
        neigh_dist = np.array(neigh_dist).squeeze()
        neigh_ind = np.array(neigh_ind).squeeze()
        for ix in neigh_ind:
            neigh_vector = self.neigh_samples[ix]
            neigh_neighbors.append(neigh_vector)
            dist = np.linalg.norm(neigh_vector - q_vector)
            neigh_distances.append(float(dist))
            neigh_names.append(self.neigh_names[ix])

        neigh_neighbors = np.array(neigh_neighbors)

        # print(f"query vector shape  {q_vector.shape}")
        # print(f"knn distance        {neigh_dist}")
        # print(f"knn index           {neigh_ind}")
        # print(f"knn neighbors shape {neigh_neighbors.shape}")
        # print(f"knn distances       {neigh_distances}")
        # print(f"knn names           {neigh_names}")

        return (neigh_neighbors, neigh_dist, neigh_names)
