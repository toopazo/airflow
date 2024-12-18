"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

import os
import sys
from copy import deepcopy
from pathlib import Path
from pprint import pprint
import numpy as np
from collections import Counter

from airflow.face_reider.sequence_cluster import SequenceCluster
from airflow.face_reider.sequence_cluster_sciikit import (
    SklSilhouette,
    SklNearestNeighbors,
)


class ClusterEval:
    def __init__(self, seq_names: list[str]):
        self.seq_clu = SequenceCluster(sequences=seq_names)

    def skl_evaluate(
        self,
        names: np.ndarray,
        labels: np.ndarray,
        barycenters: np.ndarray,
        vectors: np.ndarray,
        output_dir: Path,
    ):

        # ce = clusteval()
        # results = ce.fit(vector_data)
        # print(results)

        # # Plot
        # ce.plot()
        # ce.plot_silhouette()
        # ce.scatter()
        # ce.dendrogram()

        clu_sil = SklSilhouette(labels, barycenters, vectors)
        # clu_sil.fit_and_eval(
        #     n_clusters=3, output_path=output_dir / "cluster_fit_and_eval.png"
        # )
        clu_sil.calc_plot(output_path=output_dir / "cluster_eval.png")

        skl_nn = SklNearestNeighbors(labels, barycenters, vectors)

        # np.argwhere(clu_name_list==)
        for ix in range(0, len(labels), 5):
            # ix = 0
            print(ix)
            q_vector = skl_nn.vectors[ix]
            self.skl_nearest_neighbors(skl_nn, q_vector, barycenters)

        skl_nn.tsne(output_path=output_dir / "cluster_tsne.png")
        skl_nn.pca(output_path=output_dir / "cluster_pca.png")

    def evaluate(self, output_dir: Path):

        names, labels, barycenters, vectors = self.seq_clu.get_cluster_data()
        self.skl_evaluate(names, labels, barycenters, vectors, output_dir)

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

        print("nearest_neighbors")
        print(f"  nn_indexes            {[int(e) for e in nn_indexes]}")
        print(f"  nn_vectors shape      {nn_vectors.shape}")
        print(f"  nn_distances          {nn_distances_round}")
        print(f"  nn_labels             {[int(e) for e in nn_labels]}")

        for clu_i, bary in enumerate(barycenters):
            dist_to_bary = np.linalg.norm(q_vector - bary)
            print(f"  dist to barycenter {clu_i}  {round(dist_to_bary, 4)}")


if __name__ == "__main__":
    u_output_dir = Path(sys.argv[1])
    assert u_output_dir.is_dir()

    useqs = [
        "frame_id_000049_active_seq_000000",
        "frame_id_000049_active_seq_000001",
        "frame_id_000049_active_seq_000002",
    ]

    seq_clu = ClusterEval(seq_names=useqs)
    # for useq in useqs:
    #     print(useq)
    #     print(seq_clu.clusters[useq].print_info())

    seq_clu.evaluate(u_output_dir)
