"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

import os
import sys
from pathlib import Path
import numpy as np
from scipy import stats
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

from airflow.face_reider.sequence_cluster import SequenceCluster
from airflow.face_reider.sequence_cluster_sciikit import (
    SklSilhouette,
    SklNearestNeighbors,
)


class ClusterReId:
    def __init__(self, weak_silhouette: float, strong_silhouette: float):
        self.weak_silhouette = weak_silhouette
        self.strong_silhouette = strong_silhouette
        self.seq_clu = SequenceCluster()

    def reid_clusters(self, output_dir: Path):
        self.seq_clu.print_info()
        names, labels, barycenters, vectors = self.seq_clu.get_cluster_data()
        df, weak_clusters = self.evaluate_using_silhouette(
            names, labels, barycenters, vectors, output_dir / "silhouette_seq_clu"
        )

        print(f"weak_clusters {weak_clusters}")
        for label_a in weak_clusters:
            label_a_df = df[df["label"] == label_a]
            # print(label_a_df)
            sbl_2nd = np.array(
                [e[1] for e in label_a_df["sorted_barycenter_label"].values]
            )
            # print(f"sbl_2nd {sbl_2nd}")
            mode_of_sbl_2nd = stats.mode(sbl_2nd).mode
            # print(f"mode_of_sbl_2nd {mode_of_sbl_2nd}")
            label_b = mode_of_sbl_2nd

            arr = np.array([label_a, label_b])
            label_a = np.min(arr)
            label_b = np.max(arr)
            labels, barycenters = self.merge_clusters(
                labels, barycenters, vectors, label_a, label_b
            )
            self.evaluate_using_silhouette(
                names,
                labels,
                barycenters,
                vectors,
                output_dir / f"silhouette_merged_{label_a}_{label_b}",
            )
            # I am not sure if running more than once will keep the previous modifications
            return

        # label_a = 7
        # label_b = 8
        # labels, barycenters = self.merge_clusters(
        #     labels, barycenters, vectors, label_a, label_b
        # )
        # self.evaluate_using_silhouette(
        #     names, labels, barycenters, vectors, output_dir / "silhouette_post"
        # )

    def merge_clusters(
        self,
        labels: np.ndarray,
        barycenters: np.ndarray,
        vectors: np.ndarray,
        label_a: int,
        label_b: int,
    ):
        assert label_a < label_b
        print(f"Merging cluster {label_b} into {label_a}")

        print(f"  labels.shape      {labels.shape}")
        print(f"  barycenters.shape {barycenters.shape}")

        label_a_indexes = np.where(labels == label_a)[0]
        label_a_vectors = vectors[label_a_indexes, :]

        label_b_indexes = np.where(labels == label_b)[0]
        label_b_vectors = vectors[label_b_indexes, :]

        merged_vectors = np.concatenate((label_a_vectors, label_b_vectors), axis=0)

        # print(label_a_indexes)
        print(f"  label_a_vectors.shape {label_a_vectors.shape}")
        print(f"  label_b_vectors.shape {label_b_vectors.shape}")
        print(f"  merged_vectors.shape  {merged_vectors.shape}")

        #                     a  a  a     b  b  b  b
        #     labels = [0, 0, 1, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        # new_labels = [0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 3, 3]

        new_labels = []
        for i, label in enumerate(labels):
            new_label = label
            if label == label_b:
                new_label = label_a
                # print(f"Changing name {name} to {new_name}")
            if label > label_b:
                new_label = new_label - 1

            new_labels.append(new_label)
        # print(f"Old set len {len(set(labels))} New set len {len(set(new_labels))}")
        # print(f"Old set len {len(set(names))} New set len {len(set(new_names))}")

        new_barycenters = []
        for i, bc in enumerate(barycenters):
            new_bc = bc
            if i == label_a:
                new_bc = self.seq_clu.estimate_cluster_barycenter(merged_vectors)
            if i == label_b:
                continue
            new_barycenters.append(new_bc)

        new_labels_npy = np.array(new_labels)
        new_barycenters_npy = np.array(new_barycenters)

        print(f"  new_labels.shape      {new_labels_npy.shape}")
        print(f"  new_barycenters.shape {new_barycenters_npy.shape}")

        return new_labels_npy, new_barycenters_npy

    def evaluate_using_silhouette(
        self,
        names: np.ndarray,
        labels: np.ndarray,
        barycenters: np.ndarray,
        vectors: np.ndarray,
        output_dir: Path,
    ):
        # Update df with labels info
        df = self.seq_clu.get_dataframe()
        df["label"] = labels

        # Create output dir if necessary
        os.makedirs(output_dir, exist_ok=True)

        clu_sil = SklSilhouette(names, labels, barycenters, vectors)
        clu_sil.plot_silhouette(
            output_path=output_dir / "sequence_cluster_silhouette.png"
        )
        silhouettes, silhouette_avg = clu_sil.calculate_silhouette()
        _ = silhouette_avg
        assert isinstance(silhouettes, np.ndarray)

        # Add silhouette info to df
        df["silhouette"] = silhouettes

        df = self.add_barycenter_info(barycenters, vectors, df)
        print(df)

        # Calculate weak and strong clusters
        weak_clusters, weak_clusters_avg = self.get_weak_clusters(df)
        strong_clusters, strong_clusters_avg = self.get_strong_clusters(df)

        # Plot and save cluster info
        prefix_name = "weak_cluster"
        self.plot_clusters(
            df, weak_clusters, weak_clusters_avg, output_dir, prefix_name
        )
        prefix_name = "strong_cluster"
        self.plot_clusters(
            df, strong_clusters, strong_clusters_avg, output_dir, prefix_name
        )

        df.to_csv(output_dir / "sequence_to_cluster.csv")

        self.plot_custom_clusters(df, list(set(df["label"].values)), output_dir)

        return df, weak_clusters

    def add_barycenter_info(
        self,
        barycenters: np.ndarray,
        vectors: np.ndarray,
        df: pd.DataFrame,
    ):
        # Add columns about the barycenter distance
        # n_argsort = len(barycenters)
        # for ni in range(0, n_argsort):
        #     df[f"barycenter_dist_argsort_{ni}"] = df.index * 1.0
        #     df[f"barycenter_label_argsort_{ni}"] = df.index * 1.0
        df["sorted_barycenter_label"] = [[] for _ in range(len(df))]
        df["sorted_barycenter_dist"] = [[] for _ in range(len(df))]
        for ix in range(0, vectors.shape[0]):
            q_vector = vectors[ix, :]
            barycenter_list = []
            barycenter_dist = []
            for clu_i, bc in enumerate(barycenters):
                dist_to_bc = np.linalg.norm(q_vector - bc)
                barycenter_list.append(clu_i)
                barycenter_dist.append(dist_to_bc)
            # barycenter_list = np.array(barycenter_list)
            # barycenter_dist = np.array(barycenter_dist)
            argsort_list = np.argsort(barycenter_dist)
            df.at[ix, "sorted_barycenter_label"] = deepcopy(
                [int(barycenter_list[e]) for e in argsort_list]
            )
            df.at[ix, "sorted_barycenter_dist"] = deepcopy(
                [float(barycenter_dist[e]) for e in argsort_list]
            )
            # argsort_list = np.argsort(barycenter_dist)
            # for ni in range(0, n_argsort):
            #     argsort_i = argsort_list[ni]
            #     df.at[ix, f"barycenter_label_argsort_{ni}"] = deepcopy(
            #         barycenter_list[argsort_i]
            #     )
            #     df.at[ix, f"barycenter_dist_argsort_{ni}"] = deepcopy(
            #         barycenter_dist[argsort_i]
            #     )
            # print(f"  dist to barycenter {clu_i}  {round(dist_to_bary, 4)}")
        return df

    def get_weak_clusters(self, df: pd.DataFrame):
        weak_clusters = []
        weak_clusters_avg = []
        unique_labels = set(df["label"].values)
        for label in unique_labels:
            label_df = df[df["label"] == label]
            silhouettes = label_df["silhouette"].values
            avg_sil = np.mean(silhouettes)
            max_sil = np.max(silhouettes)
            if avg_sil <= self.weak_silhouette and max_sil < self.strong_silhouette:
                weak_clusters.append(label)
                weak_clusters_avg.append(avg_sil)
        return weak_clusters, weak_clusters_avg

    def get_strong_clusters(self, df: pd.DataFrame):
        strong_clusters = []
        strong_clusters_avg = []
        unique_labels = set(df["label"].values)
        for label in unique_labels:
            label_df = df[df["label"] == label]
            silhouettes = label_df["silhouette"].values
            avg_sil = np.mean(silhouettes)
            min_sil = np.min(silhouettes)
            if avg_sil >= self.strong_silhouette and min_sil > self.weak_silhouette:
                strong_clusters.append(label)
                strong_clusters_avg.append(avg_sil)

        return strong_clusters, strong_clusters_avg

    def plot_custom_clusters(
        self, df: pd.DataFrame, labels: list[int], output_dir: Path
    ):
        avg_sils = []
        for label in labels:
            label_df = df[df["label"] == label]
            silhouettes = label_df["silhouette"].values
            avg_sil = np.mean(silhouettes)
            avg_sils.append(avg_sil)

        prefix_name = "cluster"
        self.plot_clusters(df, labels, avg_sils, output_dir, prefix_name)

    def plot_cluster(self, df: pd.DataFrame, label: int, avg_sil: float):
        fig, ax = plt.subplots()

        max_ydata = 0
        for ix, row in df.iterrows():
            _ = ix
            xdata = row["sorted_barycenter_label"]
            ydata = row["sorted_barycenter_dist"]
            if np.max(ydata) > max_ydata:
                max_ydata = np.max(ydata)

            ax.plot(xdata, ydata)
            # ax.plot(ydata)

        ax.set(
            xlabel="Cluster label",
            xlim=[0, len(xdata)],
            ylabel="Distance to barycenter",
            ylim=[0, round(max_ydata, 0) + 1],
            title=f"Cluster label {label} avg silhouette {round(avg_sil, 2)}",
            xticks=range(0, len(xdata)),
            # xticklabels=xdata,
        )
        ax.grid()

        return fig

    def plot_clusters(
        self,
        df: pd.DataFrame,
        labels: list,
        avg_sils: list,
        output_dir: Path,
        outut_prefix: str,
    ):
        for label, avg_sil in zip(labels, avg_sils):
            sub_df = df[df["label"] == label]
            assert isinstance(sub_df, pd.DataFrame)
            fig = self.plot_cluster(sub_df, label, avg_sil)
            output_path = output_dir / f"{outut_prefix}_{label}.png"
            print(f"Saving {output_path}")
            fig.savefig(output_path)
            plt.close(fig)

    def skl_nearest_neighbors(
        self,
        skl_nn: SklNearestNeighbors,
        q_vector: np.ndarray,
        barycenters: np.ndarray,
    ):
        nn_indexes, nn_vectors, nn_distances, nn_labels = skl_nn.run_fit(
            q_vector=q_vector
        )
        nn_distances_round = [round(float(e), 8) for e in nn_distances]

        print(f"SklNearestNeighbors k = {skl_nn.n_neighbors}")
        print(f"  nn_indexes            {[int(e) for e in nn_indexes]}")
        # print(f"  nn_vectors shape      {nn_vectors.shape}")
        print(f"  nn_distances          {nn_distances_round}")
        print(f"  nn_labels             {[int(e) for e in nn_labels]}")

        for clu_i, bary in enumerate(barycenters):
            dist_to_bary = np.linalg.norm(q_vector - bary)
            print(f"  dist to barycenter {clu_i}  {round(dist_to_bary, 4)}")


if __name__ == "__main__":
    u_output_dir = Path(sys.argv[1])
    assert u_output_dir.is_dir()

    seq_clu = ClusterReId(weak_silhouette=0.35, strong_silhouette=0.55)
    seq_clu.reid_clusters(output_dir=u_output_dir)
