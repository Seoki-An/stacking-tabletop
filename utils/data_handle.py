import os
import h5py
import torch
import numpy as np

from typing import Union, List


def convert_h5py_group_to_dict(
    group: h5py.Group, n_vertex: int = None, to_list: bool = False
):
    data = {}
    for key, val in group.items():
        if isinstance(val, h5py.Group):
            data[key] = convert_h5py_group_to_dict(val, n_vertex, to_list)
        else:
            val = val[()]
            if key in ["pending_bodies", "stacked_bodies"] and n_vertex:
                assert n_vertex >= val.shape[-2]
                if val.ndim > 3:
                    points = np.inf * np.ones([val.shape[0], val.shape[1], n_vertex, 3])
                    points[:, :, : val.shape[-2], :] = val
                else:
                    points = np.inf * np.ones([val.shape[0], n_vertex, 3])
                    points[:, : val.shape[-2], :] = val
                val = points

            if val.dtype == np.float64:
                val = val.astype(np.float32)

            if to_list:
                data[key] = [val]
            else:
                data[key] = val

    return data


def convert_dict_to_h5py_group(data: Union[dict, np.ndarray], group: h5py.Group):
    for key, val in data.items():
        if isinstance(val, dict):
            subgroup = group.create_group(key)
            convert_dict_to_h5py_group(val, subgroup)
        else:
            group.create_dataset(key, data=val)


def extend_dict_of_list(
    target_data: dict, data: Union[dict, list], merge: bool = False
):
    if isinstance(data, dict):
        for key, val in data.items():
            target_data[key] = extend_dict_of_list(target_data[key], val, merge)
        return target_data
    else:
        if isinstance(target_data, list):
            if merge:
                target_data.extend(data)
            else:
                target_data.append(data)
            return target_data
        else:
            return [target_data, data]


def dict_list_to_nparray(data: Union[dict, list], stack: bool = True):
    if isinstance(data, dict):
        for key in data.keys():
            try:
                data[key] = dict_list_to_nparray(data[key], stack)
            except Exception as e:
                print(f"Error occurred while processing key '{key}': {e}")
        return data
    else:
        if stack:
            try:
                data = np.stack(data)
            except Exception as e:
                print(f"Error occurred while stacking data: {e}")
                print(f"Data shape: {[d.shape for d in data]}")
                raise e
        else:
            data = np.concatenate(data)
        return data


def stream_merge_h5(path_list: List[str], new_dir: str, name: str) -> str:
    """Memory-efficient counterpart of `merge_data`.

    Streams data from each input file into a single output file with resizable
    leaf datasets, appending along axis 0 — never holds the union in RAM. Skips
    paths that don't exist on disk.

    Returns the merged file path.
    """
    paths = [p for p in path_list if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError(
            f"stream_merge_h5: no input files exist among {path_list}"
        )

    new_dir = os.path.normpath(new_dir)
    os.makedirs(new_dir, exist_ok=True)
    out_path = os.path.join(new_dir, name)

    with h5py.File(out_path, "w") as out:
        with h5py.File(paths[0], "r") as src:
            _h5_create_resizable_like(src, out)
        for path in paths:
            with h5py.File(path, "r") as src:
                _h5_append_into(src, out)

    return out_path


def _h5_create_resizable_like(src: h5py.Group, dst: h5py.Group) -> None:
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            _h5_create_resizable_like(item, dst.create_group(name))
        else:
            shape = (0,) + tuple(item.shape[1:])
            maxshape = (None,) + tuple(item.shape[1:])
            dst.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                dtype=item.dtype,
                chunks=True,
            )


def _h5_append_into(src: h5py.Group, dst: h5py.Group) -> None:
    for name, item in src.items():
        if isinstance(item, h5py.Group):
            _h5_append_into(item, dst[name])
        else:
            data = item[:]
            dset = dst[name]
            n = dset.shape[0]
            k = data.shape[0]
            if k == 0:
                continue
            dset.resize(n + k, axis=0)
            dset[n:] = data


class StreamingH5Merger:
    """Incremental counterpart of `stream_merge_h5`.

    Opens the output file once, lazy-initializes the resizable-leaf schema from
    the first non-empty input, and folds in each subsequent input via `append`.
    Lets callers merge files as they arrive (e.g. from `ray.wait`) without
    holding all of them on disk simultaneously.
    """

    def __init__(self, out_path: str):
        self.out_path = out_path
        self._file = None

    def append(self, src_path: str) -> None:
        if not os.path.exists(src_path):
            return
        with h5py.File(src_path, "r") as src:
            if self._file is None:
                os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
                self._file = h5py.File(self.out_path, "w")
                _h5_create_resizable_like(src, self._file)
            _h5_append_into(src, self._file)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def has_data(self) -> bool:
        return self._file is not None

    def __enter__(self) -> "StreamingH5Merger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def merge_data(path_list: List[str], new_dir: str, name: str, n_vertex: int = None):

    dataset = {}
    with h5py.File(path_list[0], "r") as file:
        for key, val in file.items():
            if isinstance(val, h5py.Group):
                dataset[key] = convert_h5py_group_to_dict(val, n_vertex, to_list=True)
            else:
                dataset[key] = [val[()]]

    for path in path_list[1:]:
        dataset_ = {}
        with h5py.File(path, "r") as file:
            for key, val in file.items():
                if isinstance(val, h5py.Group):
                    dataset_[key] = convert_h5py_group_to_dict(
                        val, n_vertex, to_list=True
                    )
                else:
                    dataset_[key] = [val[()]]
        dataset = extend_dict_of_list(dataset, dataset_, merge=True)

    dataset = dict_list_to_nparray(dataset, stack=False)

    for val in dataset.values():
        if isinstance(val, np.ndarray):
            print(f"The length of data: {val.shape[0]}")
            break

    new_path = os.path.join(new_dir, name)
    os.makedirs(os.path.normpath(new_dir), exist_ok=True)
    with h5py.File(new_path, "w") as file:
        for key, val in dataset.items():
            if isinstance(val, dict):
                group = file.create_group(key)
                convert_dict_to_h5py_group(val, group)
            else:
                file.create_dataset(key, data=val)


def get_dict_masked(
    data: Union[dict, np.ndarray, torch.Tensor], mask: Union[np.ndarray, torch.Tensor]
):
    if isinstance(data, dict):
        for key, val in data.items():
            data[key] = get_dict_masked(val, mask)
    else:
        data = data[mask]

    return data


def isnan_in_dict(data: Union[dict, np.ndarray]):
    isnan_dict = {}
    if isinstance(data, dict):
        for key, val in data.items():
            isnan_dict[key] = isnan_in_dict(val)
        return isnan_dict
    else:
        return np.argwhere(np.isnan(data))
