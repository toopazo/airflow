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

        (
            clu_name_list,
            clu_barycenter_list,
            clu_vectors_list,
            clu_inferids_list,
            clu_frameids_list,
        ) = self._get_cluster_data_list()
        self.clu_names = clu_name_list
        self.clu_barycenter_list = clu_barycenter_list
        self.clu_vectors_list = clu_vectors_list
        self.clu_infer_ids = clu_inferids_list
        self.clu_frame_ids = clu_frameids_list

        names, labels, barycenters, vectors, infer_ids, frame_ids = (
            self._get_cluster_data_npy()
        )
        self.names = names
        self.labels = labels
        self.barycenters = barycenters
        self.vectors = vectors
        self.infer_ids = infer_ids
        self.frame_ids = frame_ids

    def get_cluster_data(self):
        """
        Get numpy arrays of cluster data:
            name of clusters
            label of cluster (numerical version of cluster names)
            barycenter of cluster
            vectors in the dataset
        """
        return self.names, self.labels, self.barycenters, self.vectors

    def get_dataframe(self):
        datad = {
            "name": self.names,
            "label": self.labels,
            "frame_id": self.frame_ids,
            "inference_id": self.infer_ids,
        }
        dataf = pd.DataFrame(datad)
        return dataf

    def print_info(self):
        """
        print info about the object
        """

        df = self.get_dataframe()
        print(df)

        # print(f"  clu_name_list len         {len(self.clu_name_list)}")
        # print(f"  clu_barycenter_list len   {len(self.clu_barycenter_list)}")
        # print(f"  clu_vectors_list len      {len(self.clu_vectors_list)}")

        # print(f"  names               {self.names}")
        # print(f"  names shape         {self.names.shape}")
        # print(f"  labels              {self.labels}")
        # print(f"  labels shape        {self.labels.shape}")
        # print(f"  labels set          {set(self.labels)}")
        print(f"  labels count        {Counter([int(e) for e in self.labels])}")
        print(f"  barycenters shape   {self.barycenters.shape}")
        print(f"  vectors shape       {self.vectors.shape}")

    def _get_cluster_data_list(self):
        clu_name_list = []
        clu_barycenter_list = []
        clu_vectors_list = []
        clu_inferids_list = []
        clu_frameids_list = []
        for k, v in self.clusters.items():
            clu_name = k  # k.replace("frame_id_000049_active_", "")
            sequence = v
            assert isinstance(sequence, Sequence)
            clu_frameids = sequence.get_frame_id_list()
            clu_inferids = sequence.get_inference_id_list()

            assert len(clu_frameids) == len(clu_inferids)

            vector_list = []
            for infer_id in clu_inferids:
                infer = Inference()
                infer.load_from_database(inference_id=infer_id)
                vector_list.append(infer.embedding)
            clu_vectors = np.array(vector_list)

            clu_barycenter = self.estimate_cluster_barycenter(clu_vectors)

            clu_name_list.append(clu_name)
            clu_barycenter_list.append(deepcopy(clu_barycenter))
            clu_vectors_list.append(deepcopy(clu_vectors))
            clu_inferids_list.append(deepcopy(clu_inferids))
            clu_frameids_list.append(deepcopy(clu_frameids))

        return (
            clu_name_list,
            clu_barycenter_list,
            clu_vectors_list,
            clu_inferids_list,
            clu_frameids_list,
        )

    def _get_cluster_data_npy(self) -> tuple:
        name_list = []
        label_list = []
        vector_list = []
        infer_id_list = []
        frame_id_list = []

        prev_name = ""
        label = -1

        for clu_j, clu_j_vectors in enumerate(self.clu_vectors_list):
            clu_j_name = self.clu_names[clu_j]
            clu_j_infer_ids = self.clu_infer_ids[clu_j]
            clu_j_frame_ids = self.clu_frame_ids[clu_j]

            if prev_name != clu_j_name:
                label = label + 1
                prev_name = clu_j_name

            shape0 = clu_j_vectors.shape
            num_vectors = shape0[0]
            # num_features = shape0[1]

            assert len(clu_j_infer_ids) == num_vectors

            for i in range(0, num_vectors):
                vector = clu_j_vectors[i, :]
                infer_id = clu_j_infer_ids[i]
                frame_id = clu_j_frame_ids[i]

                name_list.append(clu_j_name)
                label_list.append(label)
                vector_list.append(deepcopy(vector))
                infer_id_list.append(infer_id)
                frame_id_list.append(frame_id)

        names = np.array(name_list)
        vectors = np.array(vector_list)
        labels = np.array(label_list)
        barycenters = np.array(self.clu_barycenter_list)
        infer_ids = np.array(infer_id_list)
        frame_ids = np.array(frame_id_list)

        return names, labels, barycenters, vectors, infer_ids, frame_ids

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
    assert u_output_dir.is_dir()

    seq_clu = SequenceCluster()
    seq_clu.print_info()
