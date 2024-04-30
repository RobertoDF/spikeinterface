from spikeinterface.core.template import Templates
from spikeinterface.core import generate_sorting, InjectTemplatesRecording
import zarr


def fetch_templates_from_database(dataset="test_templates"):

    s3_path = f"s3://spikeinterface-template-database/{dataset}/"
    zarr_group = zarr.open_consolidated(s3_path, storage_options={"anon": False})

    templates_object = Templates.from_zarr_group(zarr_group)

    return templates_object
