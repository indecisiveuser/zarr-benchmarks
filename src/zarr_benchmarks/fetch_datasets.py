import numpy.typing as npt
from numpy import squeeze
import pooch

from zarr_benchmarks.read_write_zarr import read_write_zarr

ZENODO = pooch.create(
    # Use the default cache folder for the operating system
    path=pooch.os_cache("zarr-benchmarks"),
    base_url="doi:10.5281/zenodo.15544055",
    registry=None,
)
ZENODO.load_registry_from_doi()


def _fetch_from_file_system(image_name: str, zarr: bool = True) -> npt.NDArray:
    """Fetch zarr image from zenodo (if not already cached), and return as a 3D numpy array"""
    image_path = image_name

    if zarr:
        # open zarr
        image = squeeze(read_write_zarr.read_zarr_array(image_path, zarr_spec=2)).astype('uint16')

    else:
        # open n5
        image = read_write_zarr.read_n5_array(image_path).astype('uint16')

    return image


def get_data(zarr: bool = True) -> npt.NDArray:
    """Fetch image of a heart from the human organ atlas."""
    if zarr:
        return _fetch_from_file_system('/Users/schweinfurthl/Downloads/benchmarking_test_dataset/dataset.ome.zarr/s14-t0.zarr/1', zarr)
    else:
        return _fetch_from_file_system('/Users/schweinfurthl/Downloads/benchmarking_test_dataset/dataset.n5/setup14/timepoint0/s1', zarr)
    


# def get_dense_segmentation() -> npt.NDArray:
#     """Fetch small subset of C3 segmentation data from the H01 release"""
#     return _fetch_from_zenodo("H01-c3-subset.zarr")


# def get_sparse_segmentation() -> npt.NDArray:
#     """Fetch small subset of '104 proofread cells' segmentation data from the H01 release"""
#     return _fetch_from_zenodo("H01-proofread-104-subset.zarr")
