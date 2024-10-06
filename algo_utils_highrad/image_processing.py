"""
Algorithm Utilities: This module contains various functions for image processing.
"""

from typing import Collection, List, Tuple

import numpy as np
from tqdm.contrib.concurrent import process_map

from .algo_typing import IndexExpression3D
from .file_operations import load_nifti_data


__all__ = ['is_a_scan', 'is_a_mask', 'is_a_labeled_mask', 'get_duplicate_scans', 'find_joint_z_slices']


def is_a_scan(case: np.ndarray) -> bool:
    """
    Check if the input is a scan (and not a mask). A scan is considered to be a 3D numpy array with more than 2 unique
    values.

    Parameters
    ----------
    case : np.ndarray
        The input 3D numpy array.

    Returns
    -------
    bool
        True if the input is a scan, False otherwise.
    """
    if np.unique(case).size <= 2:
        return False
    return True


def is_a_mask(case: np.ndarray) -> bool:
    """
    Check if the input is a binary mask (and not a scan). A binary mask is considered to be a 3D numpy array with only
    2 unique values.

    Parameters
    ----------
    case : np.ndarray
        The input 3D numpy array.

    Returns
    -------
    bool
        True if the input is a binary mask, False otherwise.
    """
    if np.any((case != 0) & (case != 1)):
        return False
    return True


def is_a_labeled_mask(case: np.ndarray, relevant_labels: Collection[int]) -> bool:
    """
    Check if the input is a labeled mask (and not a scan). A labeled mask is considered to be a 3D numpy array with
    integer values that are all contained in the relevant_labels collection.

    Parameters
    ----------
    case : np.ndarray
        The input 3D numpy array.
    relevant_labels : Collection[int]
        The collection of relevant labels.

    Returns
    -------
    bool
        True if the input is a labeled mask, False otherwise.
    """
    if np.all(np.isin(case, relevant_labels)):
        return True
    return False


def __get_mean_and_std(scan: str) -> Tuple[float, float]:
    """
    Get the mean and standard deviation of the given scan.

    Parameters
    ----------
    scan : str
        The path to the scan.

    Returns
    -------
    Tuple[float, float]
        The mean and standard deviation of the scan. If the scan is for some reason not loadable, the function returns
        (np.nan, np.nan).
    """
    try:
        s, _ = load_nifti_data(scan)
    except Exception:
        return np.nan, np.nan
    return s.mean(), s.std()


def get_duplicate_scans(scans: List[str], multiprocess: bool = False) -> List[List[str]]:
    """
    Get the list of duplicate scans in the given list of scans. Two scans are considered duplicates if they have the
    same mean and standard deviation of their intensities. The function returns a list of lists, where each inner list
    contains the paths of the duplicate scans.

    Notes
    -----
    If a scan is for some reason not loadable, it is ignored.

    Parameters
    ----------
    scans : List[str]
        The list of scans. Each scan is a path to a NIfTI file.
    multiprocess : bool
        Whether to use multiprocessing to speed up the process. Default is False.

    Returns
    -------
    List[List[str]]
        The list of duplicate scans. Each inner list contains the paths of the duplicate scans. If there are no
        duplicates, the function returns an empty list.
    """
    if multiprocess:
        params = np.round(process_map(__get_mean_and_std, scans), 4)
    else:
        params = np.round([__get_mean_and_std(s) for s in scans], 4)
    params = params[~np.isnan(params).any(axis=1)]
    _, inds, counts = np.unique(params, return_inverse=True, return_counts=True, axis=0)

    duplicates = []
    for i, c in enumerate(counts):
        if c > 1:
            duplicates.append(list(np.asarray(scans)[np.where(inds == i)]))
    return duplicates


def find_joint_z_slices(im1: np.ndarray, im2: np.ndarray) -> Tuple[IndexExpression3D, IndexExpression3D]:
    """
    Find overlap joint z-slices between two images.

    Notes
    _____
    Assumes that im1 and im2 are 3D images similar in x and y shape.

    Parameters
    ----------
    im1: numpy.ndarray
        1st 3D image.
    im2: numpy.ndarray
        2nd 3D image.

    Returns
    -------
    s1, s2 : IndexExpression3D
        The 3D index expressions (Tuple[slice, slice, slice]) of the joint slices in im1 and im2, respectively. They are
        None if there is no overlap between the images.

    Raises
    ------
    AssertionError
        If the images have different x and y shapes.
    """

    # ensure images have the same shape over x and y axes
    assert np.all(np.asarray(im1.shape[0:2]) == np.asarray(im2.shape[0:2])), f'Images have different x and y shapes.' \
                                                                             f' im1 shape: {im1.shape}, im2 shape' \
                                                                             f': {im2.shape}.'

    # create the 2D window bounding box
    x_ind, y_ind = np.asarray(im1.shape[0:2]) // 2
    window_size = 20
    candidate_2d_window = np.s_[x_ind: x_ind + window_size, y_ind: y_ind + window_size]

    # there is an overlap between im1 and im2 over the z axis, iff one of the following holds:
    # 1) the top z-slice of im1 is contained in im2 (condition-1)
    # 2) the top z-slice of im2 is contained in im1

    s1, s2 = None, None

    # checking if condition-1 holds
    top_z_candidate_1 = im1[..., 0][candidate_2d_window]
    for z2 in range(im2.shape[-1]):  # iterate over whole im2's z axis and look for a similar candidate
        if np.allclose(top_z_candidate_1, im2[..., z2][candidate_2d_window], rtol=0, atol=1e-1):
            joint_z_window_size = min(im1.shape[-1], im2.shape[-1] - z2)
            s1_candidate = np.s_[:, :, :joint_z_window_size]
            s2_candidate = np.s_[:, :, z2: z2 + joint_z_window_size]
            if np.allclose(im1[s1_candidate], im2[s2_candidate], rtol=0, atol=1e-1):
                s1 = s1_candidate
                s2 = s2_candidate
                break

    # in case condition-1 doesn't hold
    if s1 is None:

        # checking if condition-2 holds
        top_z_candidate_2 = im2[..., 0][candidate_2d_window]
        for z1 in range(im1.shape[-1]):  # iterate over whole im1's z axis and look for a similar candidate
            if np.allclose(top_z_candidate_2, im1[..., z1][candidate_2d_window], rtol=0, atol=1e-1):
                joint_z_window_size = min(im2.shape[-1], im1.shape[-1] - z1)
                s2_candidate = np.s_[:, :, :joint_z_window_size]
                s1_candidate = np.s_[:, :, z1: z1 + joint_z_window_size]
                if np.allclose(im1[s1_candidate], im2[s2_candidate], rtol=0, atol=1e-1):
                    s1 = s1_candidate
                    s2 = s2_candidate
                    break

    return s1, s2
