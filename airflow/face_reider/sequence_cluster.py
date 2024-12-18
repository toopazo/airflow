"""
Código para evaluar que tan buena es la relación secuencia-cluster
en el espacio vectorial de embeddings
"""

import sys
from copy import deepcopy
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

from airflow.database.table_sequence import Sequence
from airflow.database.table_inference import Inference


class SequenceCluster:
    """
    Class to load from the database all the vectors and the cluster/sequence they belong to.
    """

    def __init__(self, sequences: list | None = None):
        seq_names = sequences or Sequence().get_database_sequences()

        self.clusters = {}
        for name in seq_names:
            # print()
            # print(f"Loading data for sequence_name {name} from database")
            sequence = Sequence()
            sequence.load_from_database(name)
            # print(data)
            self.clusters[name] = sequence

        clu_name_list, clu_barycenter_list, clu_vectors_list = (
            self._get_cluster_data_list()
        )
        self.clu_name_list = clu_name_list
        self.clu_barycenter_list = clu_barycenter_list
        self.clu_vectors_list = clu_vectors_list

        names, labels, barycenters, vectors = self._get_cluster_data_npy(
            clu_name_list, clu_barycenter_list, clu_vectors_list
        )
        self.names = names
        self.labels = labels
        self.barycenters = barycenters
        self.vectors = vectors

    def get_cluster_data(self):
        """
        Get numpy arrays of cluster data:
            name of clusters
            label of cluster (numerical version of cluster names)
            barycenter of cluster
            vectors in the dataset
        """
        return self.names, self.labels, self.barycenters, self.vectors

    def print_info(self):
        """
        print info about the object
        """

        datad = {"names": self.names, "labels": self.labels}
        dataf = pd.DataFrame(datad)
        print(dataf)

        # print(f"  clu_name_list len         {len(self.clu_name_list)}")
        # print(f"  clu_barycenter_list len   {len(self.clu_barycenter_list)}")
        # print(f"  clu_vectors_list len      {len(self.clu_vectors_list)}")

        # print(f"  names               {self.names}")
        # print(f"  names shape         {self.names.shape}")
        # print(f"  labels              {self.labels}")
        # print(f"  labels shape        {self.labels.shape}")
        # print(f"  labels set          {set(self.labels)}")
        print(f"  labels count        {Counter(self.labels)}")
        print(f"  barycenters shape   {self.barycenters.shape}")
        print(f"  vectors shape       {self.vectors.shape}")

    def _get_cluster_data_list(self):
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

            clu_barycenter = self._estimate_cluster_barycenter(clu_vectors)

            clu_name_list.append(clu_name)
            clu_barycenter_list.append(deepcopy(clu_barycenter))
            clu_vectors_list.append(deepcopy(clu_vectors))

        return clu_name_list, clu_barycenter_list, clu_vectors_list

    def _get_cluster_data_npy(
        self, clu_name_list: list, clu_barycenter_list: list, clu_vectors_list: list
    ) -> tuple:
        name_list = []
        label_list = []
        vect_list = []
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

                name_list.append(name)
                label_list.append(label)
                vect_list.append(deepcopy(vector_i))

        names = np.array(name_list)
        vectors = np.array(vect_list)
        labels = np.array(label_list)
        barycenters = np.array(clu_barycenter_list)
        # barycenters = clu_barycenter_list

        return names, labels, barycenters, vectors

    def _estimate_cluster_barycenter(self, vectors: np.ndarray):
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
    assert u_output_dir.is_dir()

    seq_clu = SequenceCluster()
    seq_clu.print_info()
