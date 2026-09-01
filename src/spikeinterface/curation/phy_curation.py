from pathlib import Path

import numpy as np

from spikeinterface.core import BaseSorting, SortingAnalyzer
from spikeinterface.core.sorting_tools import spike_vector_to_indices

from .curation_model import Curation, SequentialCuration


def curation_from_phy(
    sorting_or_analyzer: BaseSorting | SortingAnalyzer,
    phy_folder: str | Path,
) -> Curation | SequentialCuration:
    """
    Reconstruct split and merge decisions made in Phy.

    The sorting must be the one originally opened in Phy. Arbitrary spike
    reassignments are represented by first splitting original units into their
    final partitions and then merging partitions assigned to the same cluster.
    Unchanged units retain their IDs; changed units use collision-free IDs.

    Parameters
    ----------
    sorting_or_analyzer : BaseSorting | SortingAnalyzer
        The sorting used to create or open the Phy session, or its analyzer.
    phy_folder : str | Path
        Folder containing Phy's ``spike_times.npy`` and curated
        ``spike_clusters.npy`` files.

    Returns
    -------
    curation : Curation | SequentialCuration
        Curation model that can be passed to :func:`apply_curation`.

    Notes
    -----
    Cluster labels and deleted clusters are not imported. Phy keeps every spike
    assigned to a cluster, so labels such as ``noise`` or ``mua`` should be
    filtered separately after applying the structural curation.
    """
    if isinstance(sorting_or_analyzer, SortingAnalyzer):
        sorting = sorting_or_analyzer.sorting
    elif isinstance(sorting_or_analyzer, BaseSorting):
        sorting = sorting_or_analyzer
    else:
        raise TypeError("sorting_or_analyzer must be a BaseSorting or SortingAnalyzer")

    if sorting.get_num_segments() != 1:
        raise ValueError("Phy curation can only be reconstructed for a single-segment sorting")

    phy_folder = Path(phy_folder)
    spike_times_path = phy_folder / "spike_times.npy"
    spike_clusters_path = phy_folder / "spike_clusters.npy"
    required_paths = (spike_times_path, spike_clusters_path)
    if not all(path.is_file() for path in required_paths):
        raise FileNotFoundError(f"{phy_folder} must contain spike_times.npy and spike_clusters.npy")

    phy_spike_times = np.load(spike_times_path, mmap_mode="r").reshape(-1)
    phy_cluster_ids = np.load(spike_clusters_path, mmap_mode="r").reshape(-1)
    if phy_spike_times.size != phy_cluster_ids.size:
        raise ValueError("Phy spike_times.npy and spike_clusters.npy have different numbers of spikes")

    spikes = sorting.to_spike_vector()
    spike_times_match = spikes.size == phy_spike_times.size and np.array_equal(spikes["sample_index"], phy_spike_times)
    if not spike_times_match:
        raise ValueError(
            "The sorting spike vector does not match Phy's original spike times and ordering. "
            "Pass the exact sorting that was originally opened in Phy."
        )

    unit_ids = sorting.unit_ids.tolist()
    spike_indices = spike_vector_to_indices(
        [spikes],
        sorting.unit_ids,
        absolute_index=True,
    )[0]

    source_partitions = {}
    final_cluster_sources = {}
    removed_unit_ids = []
    for unit_id in unit_ids:
        unit_spike_indices = spike_indices[unit_id]
        if unit_spike_indices.size == 0:
            removed_unit_ids.append(unit_id)
            source_partitions[unit_id] = []
            continue

        unit_final_cluster_ids = phy_cluster_ids[unit_spike_indices]
        partitions = _group_indices_by_cluster(unit_final_cluster_ids)
        for final_cluster_id, _ in partitions:
            final_cluster_sources.setdefault(final_cluster_id, []).append(unit_id)
        source_partitions[unit_id] = partitions

    id_allocator = _UnitIdAllocator(unit_ids, final_cluster_sources)
    final_unit_ids = {}
    for final_cluster_id, source_unit_ids in final_cluster_sources.items():
        source_unit_id = source_unit_ids[0]
        source_is_unchanged = len(source_unit_ids) == 1 and len(source_partitions[source_unit_id]) == 1
        if source_is_unchanged:
            final_unit_ids[final_cluster_id] = source_unit_id
        else:
            final_unit_ids[final_cluster_id] = id_allocator.allocate(final_cluster_id)

    splits = []
    partition_unit_ids = {}
    for source_unit_id, partitions in source_partitions.items():
        if len(partitions) == 1:
            final_cluster_id, _ = partitions[0]
            partition_unit_ids[(source_unit_id, final_cluster_id)] = source_unit_id
            continue

        # Let Split.get_full_spike_indices() infer the largest partition.
        partitions = sorted(partitions, key=lambda item: item[1].size)
        split_indices = []
        split_unit_ids = []
        for partition_index, (final_cluster_id, local_spike_indices) in enumerate(partitions):
            is_last_partition = partition_index == len(partitions) - 1
            if not is_last_partition:
                split_indices.append(local_spike_indices.tolist())

            if len(final_cluster_sources[final_cluster_id]) == 1:
                partition_unit_id = final_unit_ids[final_cluster_id]
            else:
                partition_unit_id = id_allocator.allocate()
            split_unit_ids.append(partition_unit_id)
            partition_unit_ids[(source_unit_id, final_cluster_id)] = partition_unit_id

        splits.append(
            {
                "unit_id": source_unit_id,
                "mode": "indices",
                "indices": split_indices,
                "new_unit_ids": split_unit_ids,
            }
        )

    split_step = Curation(
        format_version="2",
        unit_ids=unit_ids,
        removed=removed_unit_ids,
        splits=splits,
    )
    ids_after_splits = split_step.get_final_ids_from_new_unit_ids()

    merges = []
    for final_cluster_id, source_unit_ids in final_cluster_sources.items():
        if len(source_unit_ids) < 2:
            continue
        merge_unit_ids = [partition_unit_ids[(source_unit_id, final_cluster_id)] for source_unit_id in source_unit_ids]
        merges.append(
            {
                "unit_ids": merge_unit_ids,
                "new_unit_id": final_unit_ids[final_cluster_id],
            }
        )

    if not merges:
        return split_step

    merge_step = Curation(
        format_version="2",
        unit_ids=ids_after_splits,
        merges=merges,
    )
    if not splits:
        return merge_step
    return SequentialCuration(curation_steps=[split_step, merge_step])


def _group_indices_by_cluster(cluster_ids):
    unique_cluster_ids, inverse = np.unique(cluster_ids, return_inverse=True)
    grouped_indices = np.argsort(inverse, kind="stable")
    boundaries = np.cumsum(np.bincount(inverse, minlength=unique_cluster_ids.size))[:-1]
    return [
        (int(cluster_id), indices)
        for cluster_id, indices in zip(unique_cluster_ids, np.split(grouped_indices, boundaries))
    ]


class _UnitIdAllocator:
    def __init__(self, original_unit_ids, phy_cluster_ids):
        self._use_integers = all(isinstance(unit_id, int) for unit_id in original_unit_ids)
        self._used = set(original_unit_ids)
        self._next = max([*original_unit_ids, *phy_cluster_ids], default=-1) + 1 if self._use_integers else 0

    def allocate(self, preferred=None):
        if preferred is not None:
            candidate = preferred if self._use_integers else f"phy_{preferred}"
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate

        while True:
            candidate = self._next if self._use_integers else f"phy_{self._next}"
            self._next += 1
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate
