"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint
import numpy as np
from airflow.database.table_sequence import Sequence
from airflow.database.table_inference import Inference
from airflow.face_reider.sequence_cluster_sciikit import Silhouette, KnnExtended


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

    def scikit_learn_eval(
        self,
        clu_name_list: list,
        clu_barycenter_list: list,
        clu_vectors_list: list,
        output_dir: Path,
    ):
        vect_list = []
        indx_list = []
        prev_name = ""
        label = -1

        for j, vectors in enumerate(clu_vectors_list):
            name = clu_name_list[j]
            if prev_name != name:
                label = label + 1
                prev_name = name

            shape0 = vectors.shape
            sdim = shape0[0]
            # vdim = shape0[1]

            for i in range(0, sdim):
                vector_i = vectors[i, :]
                vect_list.append(deepcopy(vector_i))
                indx_list.append(label)

        vector_data = np.array(vect_list)
        label_list = np.array(indx_list)
        barycenters = np.array(clu_barycenter_list)
        # barycenters = clu_barycenter_list

        print(f"vector_data.shape {vector_data.shape}")
        print(f"label_list {label_list} len {len(label_list)}")
        print(f"barycenters.shape {barycenters.shape}")

        # ce = clusteval()
        # results = ce.fit(vector_data)
        # print(results)

        # # Plot
        # ce.plot()
        # ce.plot_silhouette()
        # ce.scatter()
        # ce.dendrogram()

        clu_sil = Silhouette(vectors=vector_data)
        clu_sil.fit_and_eval(
            n_clusters=3, output_path=output_dir / "cluster_fit_and_eval.png"
        )
        clu_sil.eval(
            n_clusters=3,
            cluster_labels=label_list,
            barycenters=barycenters,
            output_path=output_dir / "cluster_eval.png",
        )

    def evaluate(self, output_dir: Path):

        clu_name_list, clu_barycenter_list, clu_vectors_list = self.get_cluster_data()

        self.scikit_learn_eval(
            clu_name_list, clu_barycenter_list, clu_vectors_list, output_dir
        )

        # knn_ext = KnnExtended(clu_name_list, clu_barycenter_list, clu_vectors_list)

        # self.custom_evaluation(clu_name_list, clu_barycenter_list, clu_vectors_list)

    def custom_evaluation(
        self, clu_name_list: list, clu_barycenter_list: list, clu_vectors_list: list
    ):
        cluster_data = self.add_intra_inter_data(
            clu_name_list, clu_barycenter_list, clu_vectors_list
        )

        for cluster_i, cluster_name in enumerate(clu_name_list):
            # https://en.wikipedia.org/wiki/Dunn_index
            max_intra_cluster_distance = cluster_data[cluster_name]["intra"][
                "max_intra_distance"
            ]
            min_b2b_distance = cluster_data[cluster_name]["inter"]["min_b2b_distance"]
            min_b2b_dunn_index = min_b2b_distance / max_intra_cluster_distance
            cluster_data[cluster_name]["min_b2b_dunn_index"] = min_b2b_dunn_index

            # Worst case scenario Dunn Index
            min_b2b_name = cluster_data[cluster_name]["inter"]["min_b2b_name"]
            max_intra_cluster_distance_b2b = cluster_data[min_b2b_name]["intra"][
                "max_intra_distance"
            ]
            worst_case_min_inter_cluster_distance = (
                min_b2b_distance
                - max_intra_cluster_distance
                - max_intra_cluster_distance_b2b
            )
            worst_case_dunn_index = (
                worst_case_min_inter_cluster_distance / max_intra_cluster_distance
            )
            cluster_data[cluster_name]["worst_case_dunn_index"] = worst_case_dunn_index

            # https://en.wikipedia.org/wiki/Silhouette_(clustering)

            # Threshold of 2 times norm std
            # th2std_intra_cluster_distance = cluster_data[cluster_name]["intra"][
            #     "th2std_intra_cluster_distance"
            # ]
            # th2std_intra_cluster_distance_b2b = cluster_data[min_b2b_name]["intra"][
            #     "th2std_intra_cluster_distance"
            # ]
            # th2std_min_inter_cluster_distance = (
            #     min_b2b_distance
            #     - th2std_intra_cluster_distance
            #     - th2std_intra_cluster_distance_b2b
            # )
            # th2std_dunn_index = (
            #     th2std_min_inter_cluster_distance / max_intra_cluster_distance
            # )
            # cluster_data[cluster_name]["th2std_dunn_index"] = th2std_dunn_index

            # Threshold of 1 times norm std
            th1std_intra_cluster_distance = cluster_data[cluster_name]["intra"][
                "th1std_intra_cluster_distance"
            ]
            th1std_intra_cluster_distance_b2b = cluster_data[min_b2b_name]["intra"][
                "th1std_intra_cluster_distance"
            ]
            th1std_min_inter_cluster_distance = (
                min_b2b_distance
                - th1std_intra_cluster_distance
                - th1std_intra_cluster_distance_b2b
            )
            th1std_dunn_index = (
                th1std_min_inter_cluster_distance / th1std_intra_cluster_distance
            )
            cluster_data[cluster_name]["th1std_dunn_index"] = th1std_dunn_index

            print(f"Cluster name {cluster_name}")

            _key = "max_intra_cluster_distance"
            _val = max_intra_cluster_distance
            print(f"  {_key.ljust(40)}    {_val}")

            _key = "th1std_intra_cluster_distance"
            _val = th1std_intra_cluster_distance
            print(f"  {_key.ljust(40)}    {_val}")

            # _key = "th2std_intra_cluster_distance"
            # _val = th2std_intra_cluster_distance
            # print(f"  {_key.ljust(40)}    {_val}")

            _key = "min_b2b_distance"
            _val = min_b2b_distance
            print(f"  {_key.ljust(40)}    {_val}")

            _key = "min_b2b_name"
            _val = min_b2b_name
            print(f"  {_key.ljust(40)}    {_val}")

            _key = "min_b2b_dunn_index"
            _val = min_b2b_dunn_index
            print(f"  {_key.ljust(40)}    {_val}")

            # _key = "worst_case_min_inter_cluster_distance"
            # _val = worst_case_min_inter_cluster_distance
            # print(f"  {_key.ljust(40)}    {_val}")

            # _key = "worst_case_dunn_index"
            # _val = worst_case_dunn_index
            # print(f"  {_key.ljust(40)}    {_val}")

            # _key = "th2std_min_inter_cluster_distance"
            # _val = th2std_min_inter_cluster_distance
            # print(f"  {_key.ljust(40)}    {_val}")

            # _key = "th2std_dunn_index"
            # _val = th2std_dunn_index
            # print(f"  {_key.ljust(40)}    {_val}")

            _key = "th1std_min_inter_cluster_distance"
            _val = th1std_min_inter_cluster_distance
            print(f"  {_key.ljust(40)}    {_val}")

            _key = "th1std_dunn_index"
            _val = th1std_dunn_index
            print(f"  {_key.ljust(40)}    {_val}")

            barycenter = cluster_data[cluster_name]["intra"]["barycenter"]
            neigh_neighbors, neigh_dist, neigh_names = knn_ext.run_fit(barycenter)

            # print(f"  query vector shape  {barycenter.shape}")
            _key = "knn query shape"
            _val = barycenter.shape
            print(f"  {_key.ljust(40)}    {_val}")

            # print(f"  knn distance        {neigh_dist}")
            _key = "knn distances"
            _val = neigh_dist
            print(f"  {_key.ljust(40)}    {_val}")

            # print(f"  knn neighbors shape {neigh_neighbors.shape}")
            _key = "knn neighbors"
            _val = neigh_neighbors.shape
            print(f"  {_key.ljust(40)}    {_val}")

            # print(f"  knn names           {neigh_names}")
            _key = "knn names"
            _val = neigh_names
            print(f"  {_key.ljust(40)}    {_val}")

            # vectors = clu_vectors_list[cluster_i]
            # shape0 = vectors.shape
            # sdim = shape0[0]
            # # vdim = shape0[1]
            # for si in range(0, sdim):
            #     vector_i = vectors[si, :]
            #     dist = np.linalg.norm(vector_i - barycenter)
            #     print(dist)

    def add_intra_inter_data(
        self, clu_name_list, clu_barycenter_list, clu_vectors_list
    ):
        cluster_data = {}
        for i, cluster_name in enumerate(clu_name_list):
            cluster_barycenter = clu_barycenter_list[i]
            cluster_vectors = clu_vectors_list[i]

            cluster_data[cluster_name] = {}
            cluster_data[cluster_name]["intra"] = self.intra_cluster_calculations(
                cluster_vectors, cluster_barycenter
            )

            cluster_data[cluster_name]["inter"] = self.inter_cluster_calculations(
                cluster_barycenter, clu_barycenter_list, clu_name_list
            )
        return cluster_data

    def get_cluster_data(self):
        clu_name_list = []
        clu_barycenter_list = []
        clu_vectors_list = []
        for k, v in self.clusters.items():
            clu_name = k.replace("frame_id_000049_active_", "")
            sequence = v
            assert isinstance(sequence, Sequence)
            inferid_list = sequence.get_inference_id_list()

            vector_list = []
            for infer_id in inferid_list:
                infer = Inference()
                infer.load_from_database(inference_id=infer_id)
                vector_list.append(infer.embedding)
            clu_vectors = np.array(vector_list)
            # vector_matrix = vector_matrix[:, 0:10]

            clu_barycenter = self.estimate_cluster_barycenter(clu_vectors)

            clu_name_list.append(clu_name)
            clu_barycenter_list.append(deepcopy(clu_barycenter))
            clu_vectors_list.append(deepcopy(clu_vectors))

        return clu_name_list, clu_barycenter_list, clu_vectors_list

    def inter_cluster_calculations(
        self, barycenter: np.ndarray, barycenter_list: list, cluster_list: list
    ):
        # print(f"barycenter_list len {len(barycenter_list)}")
        # for ix, ix_bc in enumerate(barycenter_list):
        min_b2b_distance = -1.0
        min_i = -1
        # cluster_name = cluster_list[ix]

        for i, barycenter_i in enumerate(barycenter_list):
            # if ix == jx:
            #     continue
            dist = float(np.linalg.norm(barycenter_i - barycenter))
            if dist == 0:
                continue
            if min_b2b_distance == -1:
                min_b2b_distance = dist
                min_i = i
            if dist < min_b2b_distance:
                min_b2b_distance = dist
                min_i = i
            # print(f"dist     {dist}")

            # print()
            # print(f"i             {i}")
            # print(f"min_dist      {min_dist}")
            # print(f"min_dist_i    {min_dist_i}")
        min_b2b_name = cluster_list[min_i]
        return {"min_b2b_distance": min_b2b_distance, "min_b2b_name": min_b2b_name}

    def estimate_cluster_barycenter(self, vectors: np.ndarray):
        # print("calculate_baricenter")
        shape0 = vectors.shape
        sdim = shape0[0]
        vdim = shape0[1]
        # print(f"There are {sdim} samples of dimension {vdim}")
        vmedian = np.median(vectors, axis=0)
        # print(vmedian.shape)
        return vmedian

    def intra_cluster_calculations(self, vectors: np.ndarray, barycenter: np.ndarray):
        # print("calculate_baricenter")
        shape0 = vectors.shape
        sdim = shape0[0]
        # vdim = shape0[1]
        # print(f"There are {sdim} samples of dimension {vdim}")
        # barycenter = self.estimate_cluster_barycenter(vector_matrix)
        norm_list = []
        for i in range(0, sdim):
            vector_i = vectors[i, :]
            delta_norm = np.linalg.norm(vector_i - barycenter)
            norm_list.append(delta_norm)
        mean_intra_distance_to_barycenter = np.mean(norm_list)
        print(
            f"Mean intra distance to barycenter is {mean_intra_distance_to_barycenter}"
        )

        std_intra_distance = np.std(vectors, axis=0)
        th1std_intra_cluster_distance = 1 * np.linalg.norm(std_intra_distance)
        print(f"Threshold 1 x norm std deviation {th1std_intra_cluster_distance}")
        # th2std_intra_cluster_distance = 2 * np.linalg.norm(std_intra_distance)
        # print(f"Threshold 2 x norm std deviation {th2std_intra_cluster_distance}")
        # print(
        #     f"Standard deviation of the scalars in the standard deviation vector {np.std(std_intra_distance)}"
        # )

        max_intra_distance, max_intra_ij = self.max_distance(vectors)
        print(f"Max intra distance {max_intra_distance} between points {max_intra_ij}")

        return {
            "barycenter": barycenter,
            "mean_intra_distance_to_barycenter": mean_intra_distance_to_barycenter,
            "th1std_intra_cluster_distance": th1std_intra_cluster_distance,
            # "th2std_intra_cluster_distance": th2std_intra_cluster_distance,
            "max_intra_distance": max_intra_distance,
            "max_intra_ij": max_intra_ij,
        }

    def max_distance(self, vectors: np.ndarray):
        # Find maximum distance between points
        # https://stackoverflow.com/questions/2736290/how-to-find-two-most-distant-points
        # https://pypi.org/project/rotating-calipers/
        shape0 = vectors.shape
        sdim = shape0[0]
        # vdim = shape0[1]

        max_dist = -1.0
        max_ij = (-1, -1)
        for i in range(0, sdim):
            vector_i = vectors[i, :]
            for j in range(0, sdim):
                vector_j = vectors[j, :]
                delta = vector_i - vector_j
                delta_norm = float(np.linalg.norm(delta))
                if max_dist == -1:
                    max_dist = delta_norm
                    max_ij = (i, j)
                if max_dist < delta_norm:
                    max_dist = delta_norm
                    max_ij = (i, j)
        return max_dist, max_ij


if __name__ == "__main__":
    useqs = [
        "frame_id_000049_active_seq_000000",
        "frame_id_000049_active_seq_000001",
        "frame_id_000049_active_seq_000002",
    ]

    seq_clu = SequenceCluster(seq_names=useqs)
    for useq in useqs:
        print(useq)
        print(seq_clu.clusters[useq].print_info())
        # if useq == "frame_id_000049_active_seq_000000":
        #     seq_clu.clusters[useq].drop_data(0, 10)
        #     seq_clu.clusters[useq].drop_data(-10, -0)
        #     print(seq_clu.clusters[useq].print_info())

    user = os.environ["USER"]
    u_output_dir = Path(
        f"/home/{user}/repos_git/airflow/output/inauguracion_metro_santiago"
    )
    seq_clu.evaluate(u_output_dir)
