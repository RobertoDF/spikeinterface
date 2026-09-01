import numpy as np
import pytest

from spikeinterface.core import NumpyRecording, NumpySorting, create_sorting_analyzer
from spikeinterface.curation import apply_curation, curation_from_phy
from spikeinterface.curation.curation_model import Curation, SequentialCuration


def _write_phy_assignments(phy_folder, sorting, cluster_ids):
    spikes = sorting.to_spike_vector()
    np.save(phy_folder / "spike_times.npy", spikes["sample_index"])
    np.save(phy_folder / "spike_clusters.npy", np.asarray(cluster_ids))


def test_curation_from_phy_merge(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {
            0: np.array([10, 30]),
            1: np.array([20, 40]),
            2: np.array([50, 60]),
        },
        sampling_frequency=30_000,
    )
    _write_phy_assignments(tmp_path, sorting, [3, 3, 3, 3, 2, 2])

    curation = curation_from_phy(sorting, tmp_path)
    curated_sorting = apply_curation(sorting, curation)

    assert isinstance(curation, Curation)
    assert set(curated_sorting.unit_ids) == {2, 3}
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(3), [10, 20, 30, 40])
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(2), [50, 60])


def test_curation_from_phy_realigns_simultaneous_spikes(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {
            0: np.array([10, 20]),
            1: np.array([10, 30]),
        },
        sampling_frequency=30_000,
    )
    spikes = sorting.to_spike_vector()
    phy_spike_order = np.array([1, 0, 2, 3])
    np.save(tmp_path / "spike_times.npy", spikes["sample_index"][phy_spike_order])
    np.save(tmp_path / "spike_templates.npy", spikes["unit_index"][phy_spike_order])
    np.save(tmp_path / "spike_clusters.npy", spikes["unit_index"][phy_spike_order])

    curation = curation_from_phy(sorting, tmp_path)
    curated_sorting = apply_curation(sorting, curation)

    assert len(curation.splits) == 0
    assert len(curation.merges) == 0
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(0), [10, 20])
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(1), [10, 30])


def test_curation_from_phy_split_then_merge(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {
            0: np.array([10, 30, 50, 70]),
            1: np.array([20, 40]),
            2: np.array([60, 80]),
        },
        sampling_frequency=30_000,
    )
    _write_phy_assignments(tmp_path, sorting, [3, 3, 4, 3, 4, 2, 4, 2])

    curation = curation_from_phy(sorting, tmp_path)
    curated_sorting = apply_curation(sorting, curation)

    assert isinstance(curation, SequentialCuration)
    assert set(curated_sorting.unit_ids) == {2, 3, 4}
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(3), [10, 20, 40])
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(4), [30, 50, 70])
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(2), [60, 80])


def test_curation_from_phy_accepts_analyzer(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {
            0: np.array([10, 30, 50, 70]),
            1: np.array([20, 40]),
        },
        sampling_frequency=30_000,
    )
    recording = NumpyRecording(
        traces_list=[np.zeros((100, 2), dtype="float32")],
        sampling_frequency=30_000,
    )
    recording.set_dummy_probe_from_locations(np.array([[0, 0], [0, 20]]))
    analyzer = create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
        sparse=False,
        main_channel_indices=np.zeros(2, dtype="int64"),
    )
    analyzer.compute("random_spikes", max_spikes_per_unit=10)
    _write_phy_assignments(tmp_path, sorting, [2, 2, 3, 2, 3, 3])

    curation = curation_from_phy(analyzer, tmp_path)
    curated_analyzer = apply_curation(analyzer, curation, n_jobs=1)

    assert isinstance(curation, SequentialCuration)
    assert set(curated_analyzer.unit_ids) == {2, 3}
    assert curated_analyzer.has_extension("random_spikes")
    np.testing.assert_array_equal(curated_analyzer.sorting.get_unit_spike_train(2), [10, 20, 40])
    np.testing.assert_array_equal(curated_analyzer.sorting.get_unit_spike_train(3), [30, 50, 70])


def test_curation_from_phy_rejects_another_sorting(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {0: np.array([10, 20])},
        sampling_frequency=30_000,
    )
    _write_phy_assignments(tmp_path, sorting, [0, 0])
    np.save(tmp_path / "spike_times.npy", [10, 21])

    with pytest.raises(ValueError, match="exact sorting"):
        curation_from_phy(sorting, tmp_path)


def test_curation_from_phy_single_spike(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {0: np.array([10])},
        sampling_frequency=30_000,
    )
    _write_phy_assignments(tmp_path, sorting, [0])

    curation = curation_from_phy(sorting, tmp_path)
    curated_sorting = apply_curation(sorting, curation)

    assert list(curated_sorting.unit_ids) == [0]
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train(0), [10])


def test_curation_from_phy_string_unit_ids(tmp_path):
    sorting = NumpySorting.from_unit_dict(
        {
            "a": np.array([10, 30]),
            "b": np.array([20, 40]),
        },
        sampling_frequency=30_000,
    )
    _write_phy_assignments(tmp_path, sorting, [2, 2, 3, 2])

    curation = curation_from_phy(sorting, tmp_path)
    curated_sorting = apply_curation(sorting, curation)

    assert set(curated_sorting.unit_ids) == {"phy_2", "phy_3"}
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train("phy_2"), [10, 20, 40])
    np.testing.assert_array_equal(curated_sorting.get_unit_spike_train("phy_3"), [30])
