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

from airflow.database.table_sequence import Sequence
from airflow.database.table_inference import Inference
from airflow.face_reider.sequence_cluster_sciikit import SklSilhouette, SklNN


class SequenceCluster:
    def __init__(self, seq_names: list[str]):

        self.clusters = {}
        for name in seq_names:
            print()
            print(f"Loading data for sequence_name {name} from database")
            sequence = Sequence()
            sequence.load_from_database(name)
            # print(data)
            self.clusters[name] = sequence

    def get_cluster_data_list(self):
        clu_name_list = []
        clu_barycenter_list = []
        clu_vectors_list = []
        for k, v in self.clusters.items():
            clu_name = k  # k.replace("frame_id_000049_active_", "")
            sequence = v
            assert isinstance(sequence, Sequence)
            inferid_list = sequence.get_inference_id_list()

            vector_list = []
            for infer_id in inferid_list:
                infer = Inference()
                infer.load_from_database(inference_id=infer_id)
                vector_list.append(infer.embedding)
            clu_vectors = np.array(vector_list)

            clu_barycenter = self.estimate_cluster_barycenter(clu_vectors)

            clu_name_list.append(clu_name)
            clu_barycenter_list.append(deepcopy(clu_barycenter))
            clu_vectors_list.append(deepcopy(clu_vectors))

        return clu_name_list, clu_barycenter_list, clu_vectors_list

    def get_cluster_data_npy(
        self, clu_name_list: list, clu_barycenter_list: list, clu_vectors_list: list
    ) -> tuple:
        vect_list = []
        label_list = []
        prev_name = ""
        label = -1

        for clu_j, clu_vectors in enumerate(clu_vectors_list):
            name = clu_name_list[clu_j]
            if prev_name != name:
                label = label + 1
                prev_name = name

            shape0 = clu_vectors.shape
            sdim = shape0[0]
            # vdim = shape0[1]

            for i in range(0, sdim):
                vector_i = clu_vectors[i, :]
                vect_list.append(deepcopy(vector_i))
                label_list.append(label)

        vectors = np.array(vect_list)
        labels = np.array(label_list)
        barycenters = np.array(clu_barycenter_list)
        # barycenters = clu_barycenter_list

        return labels, barycenters, vectors

    def skl_evaluate(
        self,
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
        clu_sil.eval(output_path=output_dir / "cluster_eval.png")

        skl_nn = SklNN(labels, barycenters, vectors)

        # np.argwhere(clu_name_list==)
        for ix in range(0, len(labels), 5):
            # ix = 0
            print(ix)
            q_vector = skl_nn.vectors[ix]
            self.skl_nearest_neighbors(skl_nn, q_vector, barycenters)

        skl_nn.tsne(output_path=output_dir / "cluster_tsne.png")
        skl_nn.pca(output_path=output_dir / "cluster_pca.png")

    def evaluate(self, output_dir: Path):

        clu_name_list, clu_barycenter_list, clu_vectors_list = (
            self.get_cluster_data_list()
        )
        print(f"  clu_name_list len         {len(clu_name_list)}")
        print(f"  clu_barycenter_list len   {len(clu_barycenter_list)}")
        print(f"  clu_vectors_list len      {len(clu_vectors_list)}")

        labels, barycenters, vectors = self.get_cluster_data_npy(
            clu_name_list, clu_barycenter_list, clu_vectors_list
        )
        print(f"  labels              {labels}")
        print(f"  labels shape        {labels.shape}")
        print(f"  labels set          {set(labels)}")
        print(f"  labels count        {Counter(labels)}")
        print(f"  barycenters shape   {barycenters.shape}")
        print(f"  vectors shape       {vectors.shape}")

        self.skl_evaluate(labels, barycenters, vectors, output_dir)

    def skl_nearest_neighbors(
        self,
        skl_nn: SklNN,
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

    def estimate_cluster_barycenter(self, vectors: np.ndarray):
        # print("calculate_baricenter")
        # shape0 = vectors.shape
        # sdim = shape0[0]
        # vdim = shape0[1]

        # print(f"There are {sdim} samples of dimension {vdim}")
        vmedian = np.median(vectors, axis=0)
        # print(vmedian.shape)
        return vmedian


if __name__ == "__main__":
    u_output_dir = Path(sys.argv[1])

    useqs = [
        "frame_id_000049_active_seq_000000",
        "frame_id_000049_active_seq_000001",
        "frame_id_000049_active_seq_000002",
    ]

    seq_clu = SequenceCluster(seq_names=useqs)
    # for useq in useqs:
    #     print(useq)
    #     print(seq_clu.clusters[useq].print_info())

    seq_clu.evaluate(u_output_dir)

    # python -m  airflow.face_reider.sequence_cluster_eval "/home/${USER}/repos_git/airflow/output/inauguracion_metro_santiago"
