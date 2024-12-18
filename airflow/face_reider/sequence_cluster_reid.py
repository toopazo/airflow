"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

import sys
from pathlib import Path
import numpy as np

from airflow.face_reider.sequence_cluster import SequenceCluster
from airflow.face_reider.sequence_cluster_sciikit import (
    SklSilhouette,
    SklNearestNeighbors,
)


class ClusterReId:
    def __init__(self, min_silhouette: float):
        self.min_silhouette = min_silhouette
        self.seq_clu = SequenceCluster()

    def cluster_similarity(self, output_dir: Path):
        self.seq_clu.print_info()

        names, labels, barycenters, vectors = self.seq_clu.get_cluster_data()
        self.skl_evaluate(names, labels, barycenters, vectors, output_dir)

    def skl_evaluate(
        self,
        names: np.ndarray,
        labels: np.ndarray,
        barycenters: np.ndarray,
        vectors: np.ndarray,
        output_dir: Path,
    ):
        clu_sil = SklSilhouette(names, labels, barycenters, vectors)
        # print(
        #     f"Num of vectors {clu_sil.n_samples} Num of features {clu_sil.n_features}"
        # )

        skl_nn = SklNearestNeighbors(names, labels, barycenters, vectors)

        silhouettes, silhouette_avg = clu_sil.calculate_silhouette()
        assert isinstance(silhouettes, np.ndarray)
        # print(silhouettes)
        for ix, sil in enumerate(silhouettes):
            if sil < self.min_silhouette:
                min_sil = self.min_silhouette
                print(
                    f"Vector {ix} in sequence {labels[ix]} has a silhouette {sil} < {min_sil}"
                )

                self.skl_nearest_neighbors(
                    skl_nn=skl_nn, q_vector=vectors[ix, :], barycenters=barycenters
                )

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

    seq_clu = ClusterReId(min_silhouette=0.35)
    seq_clu.cluster_similarity(output_dir=u_output_dir)
