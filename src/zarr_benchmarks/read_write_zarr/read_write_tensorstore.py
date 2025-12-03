import pathlib
from typing import Literal

import numpy.typing as npt
import tensorstore as ts

from zarr_benchmarks import utils


def _adjust_inner_chunk_shape(
    outer_chunk_shape: tuple[int, ...], desired_inner_chunk_shape: tuple[int, ...]
) -> tuple[int, ...]:
    """
    Adjust the inner chunk shape to evenly divide the outer chunk shape.

    For each dimension, finds the largest divisor of the outer chunk
    that is <= the desired inner chunk size.

    Args:
        outer_chunk_shape: The shape of the outer chunk (e.g., full array or chunk grid)
        desired_inner_chunk_shape: The desired inner chunk (shard) shape

    Returns:
        Adjusted inner chunk shape that evenly divides the outer chunk shape
    """
    adjusted = []
    for outer, desired_inner in zip(outer_chunk_shape, desired_inner_chunk_shape):
        if desired_inner >= outer:
            # If desired inner chunk is >= outer, just use outer dimension
            adjusted.append(outer)
        else:
            # Find the largest divisor of outer that is <= desired_inner
            for candidate in range(desired_inner, 0, -1):
                if outer % candidate == 0:
                    adjusted.append(candidate)
                    break
    return tuple(adjusted)


def get_compression_ratio(store_path: pathlib.Path, zarr_spec: Literal[2, 3]) -> float:
    zarr_array = _open_zarr_array(store_path, zarr_spec)
    item_size = zarr_array.dtype.numpy_dtype.itemsize
    nbytes = item_size * zarr_array.size
    nbytes_stored = utils.get_directory_size(store_path)
    return nbytes / nbytes_stored

def _open_zarr_array(
    store_path: pathlib.Path, zarr_spec: Literal[2, 3]
) -> ts.TensorStore:
    if zarr_spec == 2:
        driver = "zarr"
    else:
        driver = "zarr3"

    return ts.open(
        {
            "driver": driver,
            "kvstore": {
                "driver": "file",
                "path": str(store_path),
            },
        },
    ).result()

def _open_n5_array(store_path: pathlib.Path) -> ts.TensorStore:
    return ts.open(
        {
            "driver": 'n5',
            "kvstore": {
                "driver": "file",
                "path": str(store_path),
            },
        },
    ).result()

def read_zarr_array(store_path: pathlib.Path, zarr_spec: Literal[2, 3]) -> npt.NDArray:
    """Read the v2/v3 zarr spec with tensorstore"""
    zarr_read = _open_zarr_array(store_path, zarr_spec)
    read_image = zarr_read[:].read().result()
    return read_image

def read_n5_array(store_path: pathlib.Path) -> npt.NDArray:
    """Read the n5 spec with tensorstore"""
    n5_read = _open_n5_array(store_path)
    read_image = n5_read[:].read().result()
    return read_image

def _write_n5(
    image: npt.NDArray,
    store_path: pathlib.Path,
    *,
    chunks: tuple[int],
    compressor: dict | None,
    write_empty_chunks: bool = True,
) -> None:
    dataset = ts.open(
        {
            "driver": "n5",
            "kvstore": {
                "driver": "file",
                "path": str(store_path),
            },
            "metadata": {
                "dataType": str(image.dtype),
                "dimensions": image.shape,
                "compression": {"type": "raw"} if compressor is None else compressor,
                "blockSize": chunks
            },
            "create": True,
            "delete_existing": True,
            "store_data_equal_to_fill_value": write_empty_chunks,
        },
    ).result()

    write_future = dataset[:].write(image)
    write_future.result()


def _write_zarr_array_v2(
    image: npt.NDArray,
    store_path: pathlib.Path,
    *,
    chunks: tuple[int],
    compressor: dict | None,
    write_empty_chunks: bool = True,
) -> None:
    dataset = ts.open(
        {
            "driver": "zarr",
            "kvstore": {
                "driver": "file",
                "path": str(store_path),
            },
            "metadata": {
                "dtype": image.dtype.str,
                "shape": image.shape,
                "chunks": chunks,
                "compressor": compressor,
                "fill_value": 0,
            },
            "create": True,
            "delete_existing": False,
            "store_data_equal_to_fill_value": write_empty_chunks,
        },
    ).result()

    write_future = dataset[:].write(image)
    write_future.result()


def _write_zarr_array_v3(
    image: npt.NDArray,
    store_path: pathlib.Path,
    *,
    chunks: tuple[int],
    compressor: dict | None,
    write_empty_chunks: bool = True,
) -> None:
    # Adjust inner chunk shape to evenly divide the outer chunk shape (image.shape)
    adjusted_chunks = _adjust_inner_chunk_shape(image.shape, chunks)

    index_codecs = [
        {
            "name": "bytes",
            "configuration": {"endian": "little"}
        }
    ]

    codecs = [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": adjusted_chunks,
                "codecs": [compressor] if compressor is not None else [],
                "index_codecs": index_codecs,
                "index_location": "end"
                } 
        }
    ]

    dataset = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {
                "driver": "file",
                "path": str(store_path),
            },
            "metadata": {
                "zarr_format": 3,
                "node_type": "array",
                "data_type": str(image.dtype),
                "shape": image.shape,
                "chunk_grid": {
                    "name": "regular",
                    # "configuration": {"chunk_shape": chunks},
                    "configuration": {"chunk_shape": image.shape},
                },
                # "codecs": (
                #     [
                #         {"name": "bytes", "configuration": {"endian": "little"}},
                #         compressor,
                #     ]
                #     if compressor is not None
                #     else [{"name": "bytes", "configuration": {"endian": "little"}}]
                # ),
                "codecs": codecs,
                "fill_value": 0,
            },
            "create": True,
            "delete_existing": False,
            "store_data_equal_to_fill_value": write_empty_chunks,
        },
    ).result()

    with ts.Transaction() as txn:
        write_future = dataset.with_transaction(txn).write(image)
        write_future.result()


def write_zarr_array(
    image: npt.NDArray,
    store_path: pathlib.Path,
    *,
    overwrite: bool,
    chunks: tuple[int],
    compressor: dict | None,
    write_empty_chunks: bool = True,
    zarr_spec: Literal[2, 3],
) -> None:
    """Write the v2/v3 zarr spec with tensorstore"""
    if overwrite:
        utils.remove_output_dir(store_path)

    if zarr_spec == 2:
        _write_zarr_array_v2(
            image,
            store_path,
            chunks=chunks,
            compressor=compressor,
            write_empty_chunks=write_empty_chunks,
        )
    else:
        _write_zarr_array_v3(
            image,
            store_path,
            chunks=chunks,
            compressor=compressor,
            write_empty_chunks=write_empty_chunks,
        )


def get_blosc_compressor(
    cname: str,
    clevel: int,
    shuffle: Literal["shuffle", "noshuffle", "bitshuffle"],
    zarr_spec: Literal[2, 3, 'n5'],
) -> dict:
    # see the zarr shuffle docs: https://google.github.io/tensorstore/driver/zarr/index.html#json-driver/zarr/Compressor/blosc.shuffle
    match shuffle:
        case "noshuffle":
            shuffle_int = 0
        case "shuffle":
            shuffle_int = 1
        case "bitshuffle":
            shuffle_int = 2
        case _:
            raise ValueError(f"invalid shuffle value for blosc {shuffle}")

    if zarr_spec == 2:
        return {"id": "blosc", "cname": cname, "clevel": clevel, "shuffle": shuffle_int}
    elif zarr_spec == 'n5':
        return {"type": "blosc", "cname": cname, "clevel": clevel, "shuffle": shuffle_int}
    else:
        return {
            "name": "blosc",
            "configuration": {"cname": cname, "clevel": clevel, "shuffle": shuffle},
        }


def get_gzip_compressor(level: int, zarr_spec: Literal[2, 3, 'n5']) -> dict:
    if zarr_spec == 2:
        return {"id": "gzip", "level": level}
    if zarr_spec == 'n5':
        return {"type": "gzip", "level": level}
    else:
        return {"name": "gzip", "configuration": {"level": level}}


def get_zstd_compressor(level: int, zarr_spec: Literal[2, 3, 'n5']) -> dict:
    if zarr_spec == 2:
        return {"id": "zstd", "level": level}
    elif zarr_spec == 'n5':
        return {"type": "zstd", "level": level}
    else:
        return {"name": "zstd", "configuration": {"level": level}}
