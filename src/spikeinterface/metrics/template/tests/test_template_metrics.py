import numpy as np
import pytest

from spikeinterface.postprocessing.tests.common_extension_tests import AnalyzerExtensionCommonTestSuite
from spikeinterface.metrics.template import (
    ComputeTemplateMetrics,
    compute_template_metrics,
    get_single_channel_template_metric_names,
)
from spikeinterface.metrics.template.metrics import (
    single_channel_metrics,
    multi_channel_metrics,
    get_trough_and_peak_idx,
    get_peak_to_trough_duration,
    _compute_halfwidth,
)

template_metrics = get_single_channel_template_metric_names()


def _gaussian(num_samples, center, amplitude, sigma=2.0):
    x = np.arange(num_samples)
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def test_missing_peak_after_is_none():
    """A trough at the right edge leaves no window for the peak after it: the index must be None, not -1."""
    sampling_frequency = 30_000.0
    num_samples = 100
    template = _gaussian(num_samples, 50, 30.0) + _gaussian(num_samples, 95, -100.0)

    peaks_info = get_trough_and_peak_idx(template, sampling_frequency, edge_exclusion_ms=0.1)

    assert peaks_info["trough_index"] == 95
    assert peaks_info["peak_before_index"] == 50
    assert peaks_info["peak_after_index"] is None
    assert peaks_info["peak_after_width_left"] is None
    assert peaks_info["peak_after_half_width_left"] is None
    assert np.isnan(get_peak_to_trough_duration(peaks_info, sampling_frequency))


def test_no_edge_exclusion_detects_extrema():
    """With edge_exclusion_ms=0 the search starts at sample 0 and must still detect the trough."""
    sampling_frequency = 30_000.0
    num_samples = 100
    template = _gaussian(num_samples, 30, -100.0) + _gaussian(num_samples, 60, 40.0)

    peaks_info = get_trough_and_peak_idx(template, sampling_frequency, edge_exclusion_ms=0)

    assert peaks_info["trough_index"] == 30
    assert peaks_info["peak_after_index"] == 60


def test_compute_halfwidth_missing_extremum():
    hw, left, right = _compute_halfwidth(np.zeros(10), None, 30_000.0)
    assert np.isnan(hw)
    assert left is None and right is None


def test_different_params_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics using different params, and check that they are
    actually calculated using the different params.
    """
    compute_template_metrics(
        sorting_analyzer=small_sorting_analyzer,
        metric_names=["exp_decay", "spread", "half_width"],
        metric_params={"exp_decay": {"peak_function": "min"}, "spread": {"spread_smooth_um": 15}},
    )

    tm_extension = small_sorting_analyzer.get_extension("template_metrics")
    tm_params = tm_extension.params["metric_params"]

    assert tm_params["exp_decay"]["peak_function"] == "min"
    assert tm_params["spread"]["spread_smooth_um"] == 15


def test_compute_new_template_metrics(small_sorting_analyzer):
    """
    Computes template metrics then computes a subset of template metrics, and checks
    that the old template metrics are not deleted.

    Then computes template metrics with new parameters and checks that old metrics
    are deleted.
    """

    small_sorting_analyzer.delete_extension("template_metrics")

    # calculate just exp_decay
    small_sorting_analyzer.compute({"template_metrics": {"metric_names": ["exp_decay"]}})
    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")

    assert "exp_decay" in list(template_metric_extension.get_data().keys())
    assert "half_width" not in list(template_metric_extension.get_data().keys())

    # calculate all template metrics
    small_sorting_analyzer.compute("template_metrics")
    # calculate just exp_decay - this should not delete any other metrics
    small_sorting_analyzer.compute({"template_metrics": {"metric_names": ["exp_decay"]}})
    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")

    set(template_metrics) == set(template_metric_extension.get_data().keys())

    # calculate just exp_decay with delete_existing_metrics
    small_sorting_analyzer.compute(
        {"template_metrics": {"metric_names": ["exp_decay"], "delete_existing_metrics": True}}
    )
    template_metric_extension = small_sorting_analyzer.get_extension("template_metrics")
    computed_metric_names = template_metric_extension.get_data().keys()

    for metric_name in template_metrics:
        if metric_name == "exp_decay":
            assert metric_name in computed_metric_names
        else:
            assert metric_name not in computed_metric_names

    # check that, when parameters are changed, the old metrics are deleted
    small_sorting_analyzer.compute(
        {
            "template_metrics": {
                "metric_names": ["exp_decay"],
                "metric_params": {"recovery_slope": {"recovery_window_ms": 0.6}},
            }
        }
    )


def test_metric_names_in_same_order(small_sorting_analyzer):
    """
    Computes sepecified template metrics and checks order is propagated.
    """
    specified_metric_names = ["waveform_ratios", "half_width", "recovery_slope"]
    small_sorting_analyzer.compute(
        "template_metrics", metric_names=specified_metric_names, delete_existing_metrics=True
    )
    tm_columns = small_sorting_analyzer.get_extension("template_metrics").get_data().columns
    specified_metric_column_names = ComputeTemplateMetrics.get_metric_columns(specified_metric_names)

    for specified_name, column in zip(specified_metric_column_names, tm_columns):
        assert specified_name == column


def test_save_template_metrics(small_sorting_analyzer, create_cache_folder):
    """
    Computes template metrics in binary folder format. Then computes subsets of template
    metrics and checks if they are saved correctly.
    """
    import pandas as pd

    column_names = []
    for m in single_channel_metrics:
        column_names.extend(list(m.metric_columns.keys()))
    small_sorting_analyzer.compute("template_metrics")

    cache_folder = create_cache_folder
    output_folder = cache_folder / "sorting_analyzer"

    folder_analyzer = small_sorting_analyzer.save_as(format="binary_folder", folder=output_folder)
    template_metrics_filename = output_folder / "extensions" / "template_metrics" / "metrics.csv"

    saved_metrics = pd.read_csv(template_metrics_filename)
    metric_names = saved_metrics.columns

    for metric_name in column_names:
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=False)

    saved_metrics = pd.read_csv(template_metrics_filename)
    metric_names = saved_metrics.columns
    for metric_name in column_names:
        assert metric_name in metric_names

    folder_analyzer.compute("template_metrics", metric_names=["half_width"], delete_existing_metrics=True)

    saved_metrics = pd.read_csv(template_metrics_filename)
    metric_names = saved_metrics.columns
    for metric_name in column_names:
        if "half_width" in metric_name:
            assert metric_name in metric_names
        else:
            assert metric_name not in metric_names


class TestTemplateMetrics(AnalyzerExtensionCommonTestSuite):

    @pytest.mark.parametrize(
        "params",
        [
            dict(),
            dict(upsampling_factor=2),
            dict(include_multi_channel_metrics=True),
        ],
    )
    def test_extension(self, params):
        self.run_extension_tests(ComputeTemplateMetrics, params)
