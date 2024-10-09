"""
Algorithm Utilities: This module contains various functions for file operations.
"""

import inspect
import json
import os
from glob import glob
from typing import Callable, Tuple, Optional

import jsonschema
import numpy as np
from nibabel import load, as_closest_canonical, Nifti1Image


__all__ = ['load_nifti_data', 'replace_in_file_name', 'symlink_for_inner_files_in_a_dir',
           'load_and_validate_jsonschema', 'get_project_root', 'get_absolute_path']


def load_nifti_data(nifti_file_name: str) -> Tuple[np.ndarray, Nifti1Image]:
    """
    Loading data from a nifti file.

    Notes
    -----
    The data is loaded as a numpy array of type float32. The data is also converted to the closest canonical form.

    Parameters
    ----------
    nifti_file_name : str
        The path to the desired nifti file.

    Returns
    -------
    Tuple[np.ndarray, Nifti1Image]
        A tuple containing the loaded data and the file object.
    """

    # loading nifti file
    nifti_file = load(nifti_file_name)
    nifti_file = as_closest_canonical(nifti_file)

    # extracting the data of the file
    data = nifti_file.get_fdata(dtype=np.float32)

    return data, nifti_file


def replace_in_file_name(file_name: str, old_part: str, new_part: str, dir_file_name: bool = False,
                         dst_file_exist: bool = True) -> str:
    """
    Replaces a part of a file name with another part.

    Parameters
    ----------
    file_name : str
        The file name to be modified.
    old_part : str
        The part of the file name to be replaced.
    new_part : str
        The part to replace the old part.
    dir_file_name : bool, optional
        If True, the file name is a directory name. Default is False.
    dst_file_exist : bool, optional
        If True, the new file name must exist. Default is True.

    Returns
    -------
    str
        The new file name.

    Raises
    ------
    Exception
        If the old part is not found in the file name. If the new file name does not exist (and dst_file_exist is True).
    """
    if old_part not in file_name:
        raise Exception(f'The following file/dir doesn\'t contain the part "{old_part}": {file_name}')
    new_file_name = file_name.replace(old_part, new_part)
    check_if_exist = os.path.isdir if dir_file_name else os.path.isfile
    if dst_file_exist and (not check_if_exist(new_file_name)):
        raise Exception(f'It looks like the following file/dir doesn\'t exist: {new_file_name}')
    return new_file_name


def symlink_for_inner_files_in_a_dir(src: str, dst: str, map_file_basename: Optional[Callable[[str], str]] = None,
                                     filter_file_basename: Optional[Callable[[str], bool]] = None):
    """
    Create symlinks for all files in a directory to another directory. The files are symlinked with the same basename.

    Parameters
    ----------
    src : str
        The source directory.
    dst : str
        The destination directory.
    map_file_basename : Optional[Callable[[str], str]], optional
        A function to map the file basename. Default is None.
    filter_file_basename : Optional[Callable[[str], bool]], optional
        A function to filter the file basename. Default is None.

    Raises
    ------
    Exception
        If the source is not a directory.
    """
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


def get_project_root(project_marker_file_basename: str = "project_marker.txt", stack_index: int = 1) -> str:
    """
    Returns the project's root directory by looking for a marker file in the caller function's project.

    Notes
    -----
    This function is useful when the caller function is located in the project's subdirectory, and the project marker
    file is located in the project's root directory. The function traverses the directory tree upwards until it finds
    the project marker file. If the project marker file is not found, a FileNotFoundError is raised. The project marker
    file is assumed to be named "project_marker.txt" by default. The name of the project marker file can be changed by
    passing the `project_marker_file_basename` parameter.

    Parameters
    ----------
    project_marker_file_basename : str
        The name of the project marker file. Default is "project_marker.txt".
    stack_index : int
        The index of the stack frame to get the file path from. Default is 1. If one wants to get the project root of
        the function that called the function that called this function, the index should be 2, and so on.

    Returns
    -------
    str
        The project's root directory.

    Raises
    ------
    FileNotFoundError
        If the project marker file is not found in the caller function's project.
    """

    if stack_index < 1:
        raise IndexError("The stack index should be greater than or equal to 1.")

    stack = inspect.stack()

    if stack_index >= len(stack):
        raise IndexError("The stack index is out of range. Namely, it is greater than the number of stack frames.")

    caller_function_file_abs_path = os.path.abspath(stack[stack_index].filename)

    directory = os.path.dirname(caller_function_file_abs_path)
    while directory != "/":  # Stop when reached the root directory
        if project_marker_file_basename in os.listdir(directory):
            return directory
        directory = os.path.dirname(directory)
    raise FileNotFoundError(f"Project marker file ({project_marker_file_basename}) not found. "
                            f"Ensure you have a marker file in the project root.")


def get_absolute_path(p: str, project_marker_file_basename: str = "project_marker.txt") -> str:
    """
    Returns an absolute path of a file/dir, assuming it's located in the caller function's project.

    Notes
    -----
    This function is useful when the caller function is located in the project's subdirectory, and the project marker
    file is located in the project's root directory. The function traverses the directory tree upwards until it finds
    the project marker file. If the project marker file is not found, a FileNotFoundError is raised. The project marker
    file is assumed to be named "project_marker.txt" by default. The name of the project marker file can be changed by
    passing the `project_marker_file_basename` parameter.

    Parameters
    ----------
    p : str
        Relative path to a file/dir located in the current project. The path should be relative to the project's root
        directory.
    project_marker_file_basename : str
        The name of the project marker file. Default is "project_marker.txt".

    Returns
    -------
    str
        Absolute path to the given file/dir (assuming it's located in the caller function's project).

    Raises
    ------
    FileNotFoundError
        If the project marker file is not found in the caller function's project.
    """
    if p.startswith('/'):
        return get_absolute_path(p[1:])
    return os.path.join(get_project_root(project_marker_file_basename, 2), p)
