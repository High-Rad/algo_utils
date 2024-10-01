import json
import os
from glob import glob
from typing import Callable

import jsonschema
import numpy as np
from nibabel import load, as_closest_canonical


def load_nifti_data(nifti_file_name: str):
    """
    Loading data from a nifti file.

    :param nifti_file_name: The path to the desired nifti file.

    :return: A tuple in the following form: (data, file), where:
        • data is a ndarray containing the loaded data.
        • file is the file object that was loaded.
    """

    # loading nifti file
    nifti_file = load(nifti_file_name)
    nifti_file = as_closest_canonical(nifti_file)

    # extracting the data of the file
    data = nifti_file.get_fdata(dtype=np.float32)

    return data, nifti_file


def replace_in_file_name(file_name, old_part, new_part, dir_file_name=False, dst_file_exist=True):
    if old_part not in file_name:
        raise Exception(f'The following file/dir doesn\'t contain the part "{old_part}": {file_name}')
    new_file_name = file_name.replace(old_part, new_part)
    check_if_exist = os.path.isdir if dir_file_name else os.path.isfile
    if dst_file_exist and (not check_if_exist(new_file_name)):
        raise Exception(f'It looks like the following file/dir doesn\'t exist: {new_file_name}')
    return new_file_name


def symlink_for_inner_files_in_a_dir(src: str, dst: str, map_file_basename: Callable = None,
                                     filter_file_basename: Callable = None):
    """makes a symbolic link of files in a directory"""
    if not os.path.isdir(src):
        raise Exception("symlink_for_inner_files works only for directories")
    if src.endswith('/'):
        src = src[:-1]
    if dst.endswith('/'):
        dst = dst[:-1]
    os.makedirs(dst, exist_ok=True)
    map_file_basename = (lambda x: x) if map_file_basename is None else map_file_basename
    filter_file_basename = (lambda x: True) if filter_file_basename is None else filter_file_basename
    for file in glob(f'{src}/*'):
        file_basename = os.path.basename(file)
        if os.path.isdir(file):
            symlink_for_inner_files_in_a_dir(file, f'{dst}/{file_basename}')
        else:
            if filter_file_basename(file_basename):
                os.symlink(file, f'{dst}/{map_file_basename(file_basename)}')


def load_and_validate_jsonschema(json_fn: str, json_format: dict) -> dict:
    """
    Load and validate a JSON file format using jsonschema.

    Parameters
    ----------
    json_fn : str
        The filename of the JSON file.
    json_format : dict
        The json format as a jsonschema.

    Returns
    -------
    dict
        A dictionary containing the loaded JSON data.

    Raises
    ------
    FileNotFoundError
        If the specified JSON file is not found.
    jsonschema.exceptions.ValidationError
        If the loaded JSON does not conform to the expected schema.
    """

    try:
        with open(json_fn, 'r') as json_file:
            # Load JSON data from the file
            d = json.load(json_file)

        # Validate the loaded JSON data against the schema
        jsonschema.validate(d, json_format)

        return d

    except FileNotFoundError as e:
        raise FileNotFoundError(f"The specified JSON file '{json_fn}' was not found.") from e

    except jsonschema.exceptions.ValidationError as e:
        raise jsonschema.exceptions.ValidationError(
            f"Validation error in JSON file '{json_fn}': {e.message}"
        ) from e
