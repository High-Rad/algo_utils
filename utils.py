import json
import os
from glob import glob
from typing import Tuple, List, Collection, Callable, Optional, Dict, Union, Iterator

import jsonschema
from skimage.measure import centroid
import nibabel
import numpy as np
import pandas as pd
from nibabel import load, as_closest_canonical, affines, save, Nifti1Image
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import disk
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, ball
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from xlsxwriter.utility import xl_col_to_name
from time import time, gmtime
from medpy.metric import hd, assd
from multiprocessing.pool import Pool
from functools import partial
from skimage.measure import regionprops
import networkx as nx


def _surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    This function is copied from medpy.metric version 0.3.0

    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def crop_to_relevant_joint_bbox(result, reference):
    relevant_joint_case = np.logical_or(result, reference)
    slc = get_slice_of_cropped_relevant_bbox(relevant_joint_case)
    return result[slc], reference[slc]


def get_slice_of_cropped_relevant_bbox(case: np.ndarray, margin=0):
    assert isinstance(margin, int) and margin >= 0

    if case.ndim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(case)
        xmax += 1
        ymax += 1
        zmax += 1
        slc = np.s_[xmin: xmax, ymin: ymax, zmin: zmax]
    else:
        xmin, xmax, ymin, ymax = bbox2_2D(case)
        xmax += 1
        ymax += 1
        slc = np.s_[xmin: xmax, ymin: ymax]

    if margin > 0:
        new_slc = []
        for i in range(case.ndim):
            min_val, max_val = slc[i].start, slc[i].stop
            min_val = max(min_val - margin, 0)
            max_val = min(max_val + margin, case.shape[i])
            new_slc.append(slice(min_val, max_val))
        slc = tuple(new_slc)

    return slc


def min_distance(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.hd version 0.3.0

    Minimum Distance.

    Computes the (symmetric) Minimum Distance between the binary objects in two images. It is defined as the minimum
    surface distance between the objects (Hausdorff Distance however, is defined as the maximum surface distance between
    the objects).

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    min_distance : float
        The symmetric Minimum Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)
    if np.any(np.logical_and(result, reference)):
        md = np.float64(0)
    else:
        md = _surface_distances(result, reference, voxelspacing, connectivity).min()
    return md


def Hausdorff(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.hd

    Hausdorff Distance.

    Computes the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. It is defined as the maximum surface distance between the objects.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)
    return hd(result, reference, voxelspacing, connectivity)


def ASSD(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd

    Average symmetric surface distance.

    Computes the average symmetric surface distance (ASD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    assd : float
        The average symmetric surface distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    This is a real metric, obtained by calling and averaging

    >>> asd(result, reference)

    and

    >>> asd(reference, result)

    The binary images can therefore be supplied in any order.
    """
    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)
    return assd(result, reference, voxelspacing, connectivity)


def assd_and_hd(result, reference, voxelspacing=None, connectivity: int = 1, crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd and medpy.metric.hd

    Average symmetric surface distance and Hausdorff Distance.

    Computes the average symmetric surface distance (ASD) and the (symmetric) Hausdorff Distance (HD) between the binary objects in
    two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    (assd, hd) : Tuple(float, float)
        The average symmetric surface distance and The symmetric Hausdorff Distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    These are real metrics. The binary images can therefore be supplied in any order.
    """

    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)

    sds1 = _surface_distances(result, reference, voxelspacing, connectivity)
    sds2 = _surface_distances(reference, result, voxelspacing, connectivity)

    assd_res = np.mean((sds1.mean(), sds2.mean()))
    hd_res = max(sds1.max(), sds2.max())

    return assd_res, hd_res


def assd_hd_and_min_distance(result, reference, voxelspacing=None, connectivity: int = 1,
                             crop_to_relevant_scope: bool = True):
    """
    The concept is taken from medpy.metric.assd and medpy.metric.hd

    Average symmetric surface distance, Hausdorff Distance and Minimum Distance.

    Computes the average symmetric surface distance (ASD), the (symmetric) Hausdorff Distance (HD) and the (symmetric)
    Minimum Distance between the binary objects in two images.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        The decision on the connectivity is important, as it can influence the results
        strongly. If in doubt, leave it as it is.
    crop_to_relevant_scope : bool
        If set to True (by default) the two cases holding the objects will be cropped to the relevant region
        to save running time.

    Returns
    -------
    (assd, hd) : Tuple(float, float)
        The average symmetric surface distance and The symmetric Hausdorff Distance between the object(s) in ``result`` and the
        object(s) in ``reference``. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    Notes
    -----
    These are real metrics. The binary images can therefore be supplied in any order.
    """

    if crop_to_relevant_scope:
        result, reference = crop_to_relevant_joint_bbox(result, reference)

    sds1 = _surface_distances(result, reference, voxelspacing, connectivity)
    sds2 = _surface_distances(reference, result, voxelspacing, connectivity)

    assd_res = np.mean((sds1.mean(), sds2.mean()))
    hd_res = max(sds1.max(), sds2.max())
    if np.any(np.logical_and(result, reference)):
        md_res = np.float64(0)
    else:
        md_res = sds1.min()

    return assd_res, hd_res, md_res


def dice(gt_seg, prediction_seg):
    """
    compute dice coefficient
    :param gt_seg:
    :param prediction_seg:
    :return: dice coefficient between gt and predictions
    """
    seg1 = np.asarray(gt_seg).astype(np.bool_)
    seg2 = np.asarray(prediction_seg).astype(np.bool_)

    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)
    denominator_of_res = seg1.sum() + seg2.sum()
    if denominator_of_res == 0:
        return 1
    return 2. * intersection.sum() / denominator_of_res


def tp_dice(gt_seg: np.ndarray, prediction_seg: np.ndarray, nan_if_no_tp: bool = False) -> float:
    """
    Computing Dice Coefficient after filtering FN and FP connected components.

    Parameters
    ----------
    gt_seg : np.ndarray
        Ground-Truth binary segmentation.
    prediction_seg : np.ndarray
        Prediction binary segmentation.
    nan_if_no_tp : bool, default False
        Whether to return numpy.nan in case there isn't TP connected components. If set to False (by default), the
        number 1.0 will be returned.

    Returns
    -------
        The Dice Coefficient of the TP connected components.
    """
    gt_seg = label((gt_seg > 0.5).astype(np.float32))
    prediction_seg = label((prediction_seg > 0.5).astype(np.float32))

    intersections = get_tumors_intersections(gt_seg, prediction_seg)

    if not intersections:
        return np.nan if nan_if_no_tp else 1.

    gt_tp_ts = list(intersections.keys())
    pred_tp_ts = [t for ts in intersections.values() for t in ts]

    return dice(np.isin(gt_seg, gt_tp_ts), np.isin(prediction_seg, pred_tp_ts))


def approximate_diameter(volume):
    r = ((3 * volume) / (4 * np.pi)) ** (1 / 3)
    diameter = 2 * r
    return diameter


def get_connected_components(Map, connectivity=None, min_cc_size: int = 11):
    """
    Remove Small connected component
    """
    label_img = label(Map, connectivity=connectivity)
    cc_num = label_img.max()
    cc_areas = ndimage.sum(Map, label_img, range(cc_num + 1))
    area_mask = (cc_areas < min_cc_size)
    label_img[area_mask[label_img]] = 0
    return_value = label(label_img)
    return return_value


def getLargestCC(segmentation, connectivity=1):
    labels = label(segmentation, connectivity=connectivity)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC.astype(segmentation.dtype)


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


def get_liver_border(liver_case: np.ndarray, selem_radius: int = 1) -> np.ndarray:
    return np.logical_xor(liver_case, binary_erosion(liver_case, np.expand_dims(disk(selem_radius), 2))).astype(
        liver_case.dtype)


def expand_labels(label_image, distance=1, voxelspacing=None, distance_cache=None, return_distance_cache=False):
    """

    This function is based on the same named function in skimage.segmentation version 0.18.3

    expand_labels is derived from code that was
    originally part of CellProfiler, code licensed under BSD license.
    Website: http://www.cellprofiler.org

    Copyright (c) 2020 Broad Institute
    All rights reserved.

    Original authors: CellProfiler team


    Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    distance_cache : a tuple with 2 ndarrays, optional
        This two ndarrays are distances calculated earlyer to use in the current calculation
        This is used, for example, if you want to run this function several times while changing only
        the ``distance`` parameter. The calculation will be more optimized.
    return_distance_cache : bool, optional
        If this is set to True, the distances cache will be returned too. By default it's False.
        See distance_cache decumentation.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    distance_cache : a tuple with 2 ndarrays
        This will be returned only if return_distance_cache is set to True.
        See distance_cache decumentation.
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """
    if distance_cache is None:
        distances, nearest_label_coords = distance_transform_edt(
            label_image == 0, return_indices=True, sampling=voxelspacing
        )
    else:
        distances, nearest_label_coords = distance_cache
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    if return_distance_cache:
        return labels_out, (distances, nearest_label_coords)
    return labels_out


def expand_per_label(label_image: np.ndarray, dists_to_expand: Union[float, np.ndarray] = 1,
                     max_relevant_distances: Union[float, np.ndarray] = None,
                     voxelspacing=None, distance_cache=None, return_distance_cache=False):
    if max_relevant_distances is None or not return_distance_cache:
        max_relevant_distances = dists_to_expand

    # in case the distance to expand is equivalent for all labels
    if np.unique(dists_to_expand).size == 1:
        return expand_labels(label_image,
                             dists_to_expand if np.asarray(dists_to_expand).size == 1 else dists_to_expand[0],
                             voxelspacing, distance_cache, return_distance_cache)

    unique_labels = np.unique(label_image)
    unique_labels = unique_labels[unique_labels != 0]

    dists_to_expand = np.asarray(dists_to_expand)
    max_relevant_distances = np.asarray(max_relevant_distances)

    assert dists_to_expand.size == unique_labels.size
    assert max_relevant_distances.size == unique_labels.size

    # calculating the distances
    if distance_cache is None:
        distances = np.empty((unique_labels.size,) + label_image.shape, dtype=np.float32)
        for i in range(unique_labels.size):
            distances[i] = distance_transform_edt_for_certain_label((unique_labels[i], max_relevant_distances[i]),
                                                                    label_image, voxelspacing, return_indices=False)
    else:
        distances = distance_cache

    dilate_mask = (distances <= dists_to_expand.reshape([-1, *([1] * (distances.ndim - 1))]))
    treated_distances = np.where(dilate_mask, distances, np.inf)

    labels_out_ind = np.argmin(treated_distances, axis=0)
    min_dist_between_label_val = np.squeeze(np.take_along_axis(treated_distances, np.expand_dims(labels_out_ind, 0), 0),
                                            axis=0)
    labels_out_ind[min_dist_between_label_val == np.inf] = -1
    labels_out_ind += 1

    labels_out = np.concatenate([[0], unique_labels])[labels_out_ind].astype(np.float32)

    if return_distance_cache:
        return labels_out, distances
    return labels_out


def distance_transform_edt_for_certain_label(label_and_max_relevant_dist, label_image, voxelspacing, return_indices):
    label, max_relevant_dist = label_and_max_relevant_dist
    return fast_distance_transform_edt(label_image == label, voxelspacing, max_relevant_dist,
                                       return_indices=return_indices)


def fast_distance_transform_edt(input, voxelspacing, max_relevant_dist, return_indices):
    input_shape = input.shape

    size_to_extend = np.ceil(max_relevant_dist / np.asarray(voxelspacing)).astype(np.int16)

    assert input.ndim in [2, 3]

    if input.ndim == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(input)
    else:
        xmin, xmax, ymin, ymax = bbox2_2D(input)

    xmax += 1
    ymax += 1

    xmin = max(0, xmin - size_to_extend[0])
    xmax = min(input.shape[0], xmax + size_to_extend[0])

    ymin = max(0, ymin - size_to_extend[1])
    ymax = min(input.shape[1], ymax + size_to_extend[1])

    if input.ndim == 3:
        zmax += 1
        zmin = max(0, zmin - size_to_extend[2])
        zmax = min(input.shape[2], zmax + size_to_extend[2])

        slc = np.s_[xmin: xmax, ymin: ymax, zmin: zmax]
    else:
        slc = np.s_[xmin: xmax, ymin: ymax]

    # cropping the input image to the relevant are
    input = input[slc]

    if return_indices:
        distances, nearest_label_coords = distance_transform_edt(input == 0, return_indices=return_indices,
                                                                 sampling=voxelspacing)

        # extending the results to the input image shape
        nearest_label_coords[0] += xmin
        nearest_label_coords[1] += ymin
        if input.ndim == 3:
            nearest_label_coords[2] += zmin

        extended_nearest_label_coords = np.zeros((input.ndim,) + input_shape)
        if input.ndim == 3:
            extended_nearest_label_coords[:, slc[0], slc[1], slc[2]] = nearest_label_coords
        else:
            extended_nearest_label_coords[:, slc[0], slc[1]] = nearest_label_coords

    else:
        distances = distance_transform_edt(input == 0, return_indices=return_indices, sampling=voxelspacing)

    # extending the results to the input image shape
    extended_distances = np.ones(input_shape) * np.inf
    extended_distances[slc] = distances

    if return_indices:
        return extended_distances, extended_nearest_label_coords

    return extended_distances


# def shrink_labels(label_image, distance=1, voxelspacing=None, distance_cache=None, return_distance_cache=False,
#                   is_the_label_image_binary=False):
#     """
#
#     This function is based on the function expand_labels in skimage.segmentation version 0.18.3, with changes made to
#     be a shrink version of it.
#
#     See decumentation of the funcion expand_labels here.
#
#     Shrink labels in label image by ``distance`` pixels.
#     Given a label image, ``shrink_labels`` decreas label regions (connected components)
#     inwards by up to ``distance`` pixels.
#
#     Parameters
#     ----------
#     label_image : ndarray of dtype int
#         label image
#     distance : float
#         Euclidean distance in pixels by which to grow the labels. Default is one.
#     voxelspacing : float or sequence of floats, optional
#         The voxelspacing in a distance unit i.e. spacing of elements
#         along each dimension. If a sequence, must be of length equal to
#         the input rank; if a single number, this is used for all axes. If
#         not specified, a grid spacing of unity is implied.
#     distance_cache : ndarray of dtype float, optional
#         The distances calculated earlier to use in the current calculation
#         This is used, for example, if you want to run this function several times while changing only
#         the ``distance`` parameter. The calculation will be more optimized.
#     return_distance_cache : bool, optional
#         If this is set to True, the distances cache will be returned too. By default it's False.
#         See distance_cache documentation.
#     is_the_label_image_binary : bool, optional
#         If this is set to True, the function assumes that the given label_image is consisted of only 0,1 labels, and
#         calculates some calculations faster.
#     Returns
#     -------
#     reduced_labels : ndarray of dtype int
#         Labeled array, where all connected regions have been reduced
#     distance_cache : ndarray of dtype float
#         This will be returned only if return_distance_cache is set to True.
#         See distance_cache documentation.
#     """
#
#     if distance_cache is None:
#         # extract borders
#         morph_func = binary_erosion if is_the_label_image_binary else erosion
#         # borders = np.logical_and(label_image, np.logical_not(morph_func(label_image, ball(1)))).astype(label_image.dtype)
#         borders = np.logical_and(label_image, np.logical_not(morph_func(label_image, np.array(
#             [[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool_)))).astype(label_image.dtype)
#         borders = label_image * borders
#
#         distances = distance_transform_edt(borders == 0, sampling=voxelspacing)
#     else:
#         distances = distance_cache
#     labels_out = np.zeros_like(label_image)
#     dilate_mask = distances >= distance
#
#     labels_out[dilate_mask] = 1
#     labels_out *= label_image
#     if return_distance_cache:
#         return labels_out, distances
#     return labels_out


# def find_pairs(baseline_moved_labeled, followup_labeled, reverse=False, voxelspacing=None, max_dilate_param=5,
#                return_iteration_indicator=False):
#     list_of_pairs = []
#     # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
#     for dilate in range(max_dilate_param):
#
#         # find intersection areas between baseline and followup
#         matched = np.logical_and((baseline_moved_labeled > 0), followup_labeled > 0)
#
#         # iterate over tumors in baseline that intersect tumors in followup
#         for i in np.unique((matched * baseline_moved_labeled).flatten()):
#             if i == 0:
#                 continue
#
#             # find intersection between current BL tumor and FU tumors
#             overlap = ((baseline_moved_labeled == i) * followup_labeled)
#
#             # get the labels of the FU tumors that intersect the current BL tumor
#             follow_up_num = np.unique(overlap.flatten())
#
#             # in case there is more than 1 FU tumor that intersect the current BL tumor
#             if follow_up_num.shape[0] > 2:
#                 sum = 0
#                 # iterate over the FU tumors that intersect the current BL tumor
#                 for j in follow_up_num[1:]:
#                     # in case the current FU tumor has the biggest found overlap with current BL tumor,
#                     # till this iteration
#                     if sum < (overlap == j).sum():
#                         sum = (overlap == j).sum()
#                         biggest = j
#
#                     # in case the overlap of the current FU tumor with the current BL tumor
#                     # is grader than 10% of the BL or FU tumor
#                     elif ((overlap == j).sum() / (followup_labeled == j).sum()) > 0.1 or (
#                             (overlap == j).sum() / (baseline_moved_labeled == i).sum()) > 0.1:
#                         # a match was found
#                         if reverse:
#                             if return_iteration_indicator:
#                                 list_of_pairs.append((dilate + 1, (j, i)))
#                             else:
#                                 list_of_pairs.append((j, i))
#                         else:
#                             if return_iteration_indicator:
#                                 list_of_pairs.append((dilate + 1, (i, j)))
#                             else:
#                                 list_of_pairs.append((i, j))
#                         # zero the current FU tumor and the current BL tumor
#                         baseline_moved_labeled[baseline_moved_labeled == i] = 0
#                         followup_labeled[followup_labeled == j] = 0
#                 # marking the FU tumor with the biggest overlap with the BL tumor as a found match
#                 if reverse:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (biggest, i)))
#                     else:
#                         list_of_pairs.append((biggest, i))
#                 else:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (i, biggest)))
#                     else:
#                         list_of_pairs.append((i, biggest))
#
#                 # zero the current BL tumor and the FU tumor that has jost been
#                 # marked as a match with the current BL tumor
#                 baseline_moved_labeled[baseline_moved_labeled == i] = 0
#                 followup_labeled[followup_labeled == biggest] = 0
#
#             # in case there is only 1 FU tumor that intersect the current BL tumor
#             elif follow_up_num.shape[0] > 1:
#                 # a match was found
#                 if reverse:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (follow_up_num[-1], i)))
#                     else:
#                         list_of_pairs.append((follow_up_num[-1], i))
#                 else:
#                     if return_iteration_indicator:
#                         list_of_pairs.append((dilate + 1, (i, follow_up_num[-1])))
#                     else:
#                         list_of_pairs.append((i, follow_up_num[-1]))
#                 # zero the current BL tumor and the FU tumor that intersects it
#                 baseline_moved_labeled[baseline_moved_labeled == i] = 0
#                 followup_labeled[followup_labeled == follow_up_num[-1]] = 0
#
#         # dilate the BL and FU tumors cases
#         # if dilate % 5 == 3:
#         #     plus = ndimage.generate_binary_structure(3, 1)
#         #     plus[1, 1, 0] = False
#         #     plus[1, 1, 2] = False
#         #     baseline_moved_labeled = dilation(baseline_moved_labeled, selem=ndimage.generate_binary_structure(3, 1))
#         #     followup_labeled = dilation(followup_labeled, selem=plus)
#         # else:
#         #     plus = ndimage.generate_binary_structure(3, 1)
#         #     plus[1, 1, 0] = False
#         #     plus[1, 1, 2] = False
#         #     baseline_moved_labeled = dilation(baseline_moved_labeled, selem=plus)
#         #     followup_labeled = dilation(followup_labeled, selem=plus)
#
#         # dilation without overlap and considering resolution
#         baseline_moved_labeled = expand_labels(baseline_moved_labeled, distance=1, voxelspacing=voxelspacing)
#         followup_labeled = expand_labels(followup_labeled, distance=1, voxelspacing=voxelspacing)
#
#     return (list_of_pairs)

def find_pairs(baseline_moved_labeled, followup_labeled, reverse=False, voxelspacing=None, max_dilate_param=5,
               return_iteration_and_reverse_indicator=False):
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    bl_matched_tumors, fu_matched_tumors = [], []

    list_of_pairs = []
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find intersection areas between baseline and followup
        matched = np.logical_and((working_baseline_moved_labeled > 0), working_followup_labeled > 0)

        # iterate over tumors in baseline that intersect tumors in followup
        for i in np.unique((matched * working_baseline_moved_labeled).flatten()):
            if i == 0:
                continue

            # find intersection between current BL tumor and FU tumors
            overlap = ((working_baseline_moved_labeled == i) * working_followup_labeled)

            # get the labels of the FU tumors that intersect the current BL tumor
            follow_up_num = np.unique(overlap.flatten())

            # in case there is more than 1 FU tumor that intersect the current BL tumor
            if follow_up_num.shape[0] > 2:
                sum = 0
                # iterate over the FU tumors that intersect the current BL tumor
                for j in follow_up_num[1:]:
                    # in case the current FU tumor has the biggest found overlap with current BL tumor,
                    # till this iteration
                    if sum < (overlap == j).sum():
                        sum = (overlap == j).sum()
                        biggest = j

                    # in case the overlap of the current FU tumor with the current BL tumor
                    # is grader than 10% of the BL or FU tumor
                    elif ((overlap == j).sum() / (working_followup_labeled == j).sum()) > 0.1 or (
                            (overlap == j).sum() / (working_baseline_moved_labeled == i).sum()) > 0.1:
                        # a match was found
                        if reverse:
                            if return_iteration_and_reverse_indicator:
                                list_of_pairs.append((int(reverse), dilate + 1, (j, i)))
                            else:
                                list_of_pairs.append((j, i))
                        else:
                            if return_iteration_and_reverse_indicator:
                                list_of_pairs.append((int(reverse), dilate + 1, (i, j)))
                            else:
                                list_of_pairs.append((i, j))
                        # zero the current FU tumor and the current BL tumor
                        bl_matched_tumors.append(i)
                        fu_matched_tumors.append(j)
                        working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                        working_followup_labeled[working_followup_labeled == j] = 0
                # marking the FU tumor with the biggest overlap with the current BL tumor as a found match
                if reverse:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (biggest, i)))
                    else:
                        list_of_pairs.append((biggest, i))
                else:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (i, biggest)))
                    else:
                        list_of_pairs.append((i, biggest))

                # zero the current BL tumor and the FU tumor that has jost been
                # marked as a match with the current BL tumor
                bl_matched_tumors.append(i)
                fu_matched_tumors.append(biggest)
                working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                working_followup_labeled[working_followup_labeled == biggest] = 0

            # in case there is only 1 FU tumor that intersect the current BL tumor
            elif follow_up_num.shape[0] > 1:
                # a match was found
                if reverse:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (follow_up_num[-1], i)))
                    else:
                        list_of_pairs.append((follow_up_num[-1], i))
                else:
                    if return_iteration_and_reverse_indicator:
                        list_of_pairs.append((int(reverse), dilate + 1, (i, follow_up_num[-1])))
                    else:
                        list_of_pairs.append((i, follow_up_num[-1]))

                # zero the current BL tumor and the FU tumor that intersects it
                bl_matched_tumors.append(i)
                fu_matched_tumors.append(follow_up_num[-1])
                working_baseline_moved_labeled[working_baseline_moved_labeled == i] = 0
                working_followup_labeled[working_followup_labeled == follow_up_num[-1]] = 0

        if dilate == (max_dilate_param - 1) or np.all(working_baseline_moved_labeled == 0) or np.all(
                working_followup_labeled == 0):
            break

        # dilation without overlap and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, bl_matched_tumors)] = 0
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)
        working_followup_labeled[np.isin(working_followup_labeled, fu_matched_tumors)] = 0

    return (list_of_pairs)


def match_2_cases(BL_tumors_labels, FU_tumors_labels, voxelspacing=None, max_dilate_param=5,
                  return_iteration_and_reverse_indicator=False):
    """
    • This version removes the tumors immediately after their match, not at the end of the iteration (as a result,
        the number of the tumors may affect the final results. Additionally, it requires one check as (bl=BL, fu=FU) and
        one check as (bl=FU, fu=BL)).
    • This version works with python 'for' iterations and not with numpy optimizations.
    """
    first = BL_tumors_labels.copy()
    second = FU_tumors_labels.copy()

    list_of_pairs = find_pairs(first, second, voxelspacing=voxelspacing, max_dilate_param=max_dilate_param,
                               return_iteration_and_reverse_indicator=return_iteration_and_reverse_indicator)

    first = BL_tumors_labels.copy()
    second = FU_tumors_labels.copy()
    list_of_pairs2 = find_pairs(second, first, reverse=True, voxelspacing=voxelspacing,
                                max_dilate_param=max_dilate_param,
                                return_iteration_and_reverse_indicator=return_iteration_and_reverse_indicator)

    resulting_list = list(list_of_pairs)
    if return_iteration_and_reverse_indicator:
        if len(resulting_list) > 0:
            _, _, resulting_matches = zip(*resulting_list)
            resulting_list.extend(x for x in list_of_pairs2 if x[2] not in resulting_matches)
        else:
            resulting_list = list(list_of_pairs2)
    else:
        resulting_list.extend(x for x in list_of_pairs2 if x not in resulting_list)

    return resulting_list


def match_2_cases_v2(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    """
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    if return_iteration_indicator:
        pairs = np.array([]).reshape([0, 3])
    else:
        pairs = np.array([]).reshape([0, 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            # take as a match any maximum intersection with bl tumors
            current_pairs_inds = np.stack(
                [np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack(
                [np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if dilate == (max_dilate_param - 1):
            break

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def match_2_cases_v3(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    """
    working_baseline_moved_labeled = baseline_moved_labeled
    working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            # take as a match any maximum intersection with bl tumors
            # current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            # current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if dilate == (max_dilate_param - 1):
            break

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def match_2_cases_v4(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    • This version expands the tumors after each iteration relative to the tumors size.
    """

    def dists_to_expand(diameters: np.ndarray, i_1: float = 6, i_10: float = 3) -> np.ndarray:
        """
        The distance to expand is a function of the diameter.

        i_1 is the number of mm to expand for a tumor with a diameter of 1
        i_10 is the number of mm to expand for a tumor with a diameter of 10
        See here the visual graph: https://www.desmos.com/calculator/lznykxikim
        """

        b = (10 * i_10 - i_1) / 9
        c = ((i_1 ** 10) / i_10) ** (1 / 9)

        if isinstance(diameters, (int, float)):
            diameters = np.array([diameters])

        diameters = np.clip(diameters, 1, 20)
        dists = np.empty(diameters.shape, np.float32)

        # for tumors with diameter less or equal to 10, the distance to expand
        # as a function of the diameter is (i_1 - b)/x + b, where x is the diameter, and b and i_1 are defined above.
        dists[diameters <= 10] = ((i_1 - b) / diameters[diameters <= 10]) + b

        # for tumors with diameter greater than 10, the distance to expand
        # as a function of the diameter is c * (i_1/c)^x, where x is the diameter, and c and i_1 are defined above.
        dists[diameters > 10] = c * ((i_1 / c) ** diameters[diameters > 10])

        return dists

    def dists_to_expand_v2(diameters: np.ndarray, i: float = 5, j: float = 3, k: float = 0.05) -> np.ndarray:
        """
        The distance (number of mm) to expand is a function of the diameter. The following functions are in use:

        func_1: a/x + b, where x is the diameter.
        func_2: c/e^(dx), where x is the diameter.

        For a diameter between 1 and 10, func_1 is used, and for a diameter between 10 and 20, func_2 is used.
        All the diameters is clipped to range [1,20]

        The parameters, {a,b} for func_1 and {c,d} for func_2 is decided due to the given arguments.

        i is the number of mm to expand for a tumor with a diameter of 1
        j is the number of mm to expand for a tumor with a diameter of 10
        k is the number of mm to expand for a tumor with a diameter of 20
        See here the visual graph: https://www.desmos.com/calculator/dvokawlytl
        """

        if isinstance(diameters, (int, float)):
            diameters = np.array([diameters])

        diameters = np.clip(diameters, 1, 20)
        dists = np.empty(diameters.shape, np.float32)

        # for tumors with diameter less or equal to 10, the distance to expand
        # as a function of the diameter is 10(i-j)/9x + (10j-i)/9, where x is the diameter, and i and j are defined above.
        dists[diameters <= 10] = 10 * (i - j) / (9 * diameters[diameters <= 10]) + (10 * j - i) / 9

        # for tumors with diameter greater than 10, the distance to expand
        # as a function of the diameter is j * (j/k)^((10-x)/10), where x is the diameter, and j and k are defined above.
        dists[diameters > 10] = j * ((j / k) ** ((10 - diameters[diameters > 10]) / 10))

        return dists

    def extract_diameters(tumors_labeled_case: np.ndarray) -> np.ndarray:
        tumors_labels, tumors_vol = np.unique(tumors_labeled_case, return_counts=True)
        tumors_vol = tumors_vol[tumors_labels > 0] * np.asarray(voxelspacing).prod()
        return (6 * tumors_vol / np.pi) ** (1 / 3)

    bl_dists_to_expand = dists_to_expand_v2(extract_diameters(baseline_moved_labeled))
    bl_max_relevant_distances = max_dilate_param * bl_dists_to_expand

    fu_dists_to_expand = dists_to_expand_v2(extract_diameters(followup_labeled))
    fu_max_relevant_distances = max_dilate_param * fu_dists_to_expand

    # working_baseline_moved_labeled = baseline_moved_labeled
    # working_followup_labeled = followup_labeled

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_per_label(baseline_moved_labeled,
                                                                             dists_to_expand=(
                                                                                                         dilate + 1) * bl_dists_to_expand,
                                                                             max_relevant_distances=bl_max_relevant_distances,
                                                                             voxelspacing=voxelspacing,
                                                                             distance_cache=distance_cache_bl,
                                                                             return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_per_label(followup_labeled,
                                                                       dists_to_expand=(
                                                                                                   dilate + 1) * fu_dists_to_expand,
                                                                       max_relevant_distances=fu_max_relevant_distances,
                                                                       voxelspacing=voxelspacing,
                                                                       distance_cache=distance_cache_fu,
                                                                       return_distance_cache=True)

        # zero the BL tumor and the FU tumor in the matches
        working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
        working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            # take as a match any maximum intersection with bl tumors
            # current_pairs_inds = np.stack([np.arange(relevant_bl_tumors.size), np.argmax(intersection_matrix_overlap_vol, axis=1)]).T

            # take as a match any maximum intersection with fu tumors
            # current_pairs_inds = np.unique(np.concatenate([current_pairs_inds, np.stack([np.argmax(intersection_matrix_overlap_vol, axis=0), np.arange(relevant_fu_tumors.size)]).T]), axis=0)

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

        if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
            break

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def match_2_cases_v5(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                     return_iteration_indicator=False):
    """
    • This version removes the tumors only at the end of the iterations.
    • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
    • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
        intersection as a match.
    • This version dilates the images ones in the beginning.
    """

    if np.all(baseline_moved_labeled == 0) or np.all(followup_labeled == 0):
        return []

    distance_cache_bl, distance_cache_fu = None, None

    pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
    # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
    for dilate in range(max_dilate_param):

        # dilation without overlap, and considering resolution
        working_baseline_moved_labeled, distance_cache_bl = expand_labels(baseline_moved_labeled, distance=dilate + 1,
                                                                          voxelspacing=voxelspacing,
                                                                          distance_cache=distance_cache_bl,
                                                                          return_distance_cache=True)
        working_followup_labeled, distance_cache_fu = expand_labels(followup_labeled, distance=dilate + 1,
                                                                    voxelspacing=voxelspacing,
                                                                    distance_cache=distance_cache_fu,
                                                                    return_distance_cache=True)

        if dilate > 0:
            # zero the BL tumor and the FU tumor in the matches
            working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
            working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

            if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
                break

        # find pairs of intersection of tumors
        pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(np.int16)
        pairs_of_intersection, overlap_vol = np.unique(
            pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

        if pairs_of_intersection.size > 0:

            relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs_of_intersection[:, 0])],
                return_counts=True)
            relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                return_counts=True)

            # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
            intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
            intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                            np.searchsorted(relevant_fu_tumors,
                                                            pairs_of_intersection[:, 1])] = overlap_vol

            # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
            intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                [-1, 1])

            # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
            intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                [1, -1])

            current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
            # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T]), axis=0)
            current_pairs_inds = np.unique(np.concatenate(
                [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T]), axis=0)

            if return_iteration_indicator:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                     np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
            else:
                current_pairs = np.stack(
                    [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

            pairs = np.concatenate([pairs, current_pairs])

    if return_iteration_indicator:
        return [(p[2], (p[0], p[1])) for p in pairs]
    return [tuple(p) for p in pairs]


def pre_process_segmentation(seg, remove_small_obs=True, min_obj_size: int = 20):
    # fill holes over 2D slices
    res = binary_fill_holes(seg, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool_))

    # removing small objects
    if remove_small_obs:
        res = remove_small_objects(res, min_size=min_obj_size)

    return res.astype(seg.dtype)


def replace_in_file_name(file_name, old_part, new_part, dir_file_name=False, dst_file_exist=True):
    if old_part not in file_name:
        raise Exception(f'The following file/dir doesn\'t contain the part "{old_part}": {file_name}')
    new_file_name = file_name.replace(old_part, new_part)
    check_if_exist = os.path.isdir if dir_file_name else os.path.isfile
    if dst_file_exist and (not check_if_exist(new_file_name)):
        raise Exception(f'It looks like the following file/dir doesn\'t exist: {new_file_name}')
    return new_file_name


def write_to_excel(df, writer, columns_order, column_name_as_index, sheet_name='Sheet1',
                   f1_scores: Optional[Dict[str, Tuple[str, str]]] = None):
    df = df.set_index(column_name_as_index)

    def add_agg_to_df(_df):
        # Separate the numerical and non-numerical columns
        num_df = _df.select_dtypes(include=[np.number])
        non_num_df = _df.select_dtypes(exclude=[np.number])

        # Perform aggregation on the numerical columns
        agg_df = num_df.agg(['mean', 'std', 'min', 'max', 'sum'])

        # Create a new dataframe with NaN for the non-numerical columns
        nan_df = pd.DataFrame(np.nan, index=agg_df.index, columns=non_num_df.columns)

        # Concatenate the NaN dataframe with the aggregated numerical dataframe
        agg_full_df = pd.concat([nan_df, agg_df], axis=1)

        # Concatenate the original dataframe with the full aggregated dataframe
        result_df = pd.concat([_df, agg_full_df], ignore_index=False)

        return result_df

    # df = pd.concat([df, df.agg(['mean', 'std', 'min', 'max', 'sum'])], ignore_index=False)
    df = add_agg_to_df(df)

    workbook = writer.book
    # cell_format = workbook.add_format()
    cell_format = workbook.add_format({'num_format': '#,##0.00'})
    cell_format.set_font_size(16)

    columns_order = [c for c in columns_order if c != column_name_as_index]
    df.to_excel(writer, sheet_name=sheet_name, columns=columns_order, startrow=1, startcol=1, header=False, index=False)
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'font_size': 16,
        'valign': 'top',
        'border': 1})

    max_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#E6FFCC'})
    min_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#FFB3B3'})
    last_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#C0C0C0',
        'border': 1,
        'num_format': '#,##0.00'})

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 1)

    n = df.shape[0] - 5
    for col in np.arange(len(columns_order)) + 1:
        for i, measure in enumerate(['AVERAGE', 'STDEV', 'MIN', 'MAX', 'SUM'], start=1):
            col_name = xl_col_to_name(col)
            worksheet.write(f'{col_name}{n + i + 1}', f'{{={measure}({col_name}2:{col_name}{n + 1})}}')

    if f1_scores is not None:
        for col_name in f1_scores:
            f1_col_name = xl_col_to_name(columns_order.index(col_name) + 1)
            p_col_name = xl_col_to_name(columns_order.index(f1_scores[col_name][0]) + 1)
            r_col_name = xl_col_to_name(columns_order.index(f1_scores[col_name][1]) + 1)
            worksheet.write(f'{f1_col_name}{n + 2}', f'{{=HARMEAN({p_col_name}{n + 2}:{r_col_name}{n + 2})}}')
            for i in range(1, 5):
                worksheet.write(f'{f1_col_name}{n + 2 + i}', " ")

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4),
                                 {'type': 'formula',
                                  'criteria': '=B2=B$' + str(len(df.axes[0])),
                                  'format': max_format})

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4),
                                 {'type': 'formula',
                                  'criteria': '=B2=B$' + str(
                                      len(df.axes[0]) - 1),
                                  'format': min_format})

    for i in range(len(df.axes[0]) - 4, len(df.axes[0]) + 1):
        worksheet.set_row(i, None, last_format)

    for col_num, value in enumerate(columns_order):
        worksheet.write(0, col_num + 1, value, header_format)
    for row_num, value in enumerate(df.axes[0].astype(str)):
        worksheet.write(row_num + 1, 0, value, header_format)

    # Fix first column
    column_len = df.axes[0].astype(str).str.len().max() + df.axes[0].astype(str).str.len().max() * 0.5
    worksheet.set_column(0, 0, column_len, cell_format)

    # Fix all  the rest of the columns
    for i, col in enumerate(columns_order):
        # find length of column i
        column_len = df[col].astype(str).str.len().max()
        # Setting the length if the column header is larger
        # than the max column value length
        column_len = max(column_len, len(col))
        column_len += column_len * 0.5
        # set the column length
        worksheet.set_column(i + 1, i + 1, column_len, cell_format)


def calculate_runtime(t):
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'


def bbox2_3D(img):
    x = np.any(img, axis=(1, 2))
    y = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return xmin, xmax, ymin, ymax, zmin, zmax


def bbox2_2D(img):
    x = np.any(img, axis=1)
    y = np.any(img, axis=0)

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]

    return xmin, xmax, ymin, ymax


def get_liver_segments(liver_case: np.ndarray) -> np.ndarray:
    xmin, xmax, ymin, ymax, _, _ = bbox2_3D(liver_case)
    res = np.zeros_like(liver_case)
    res[(xmin + xmax) // 2:xmax, (ymin + ymax) // 2:ymax, :] = 1
    res[xmin:(xmin + xmax) // 2, (ymin + ymax) // 2:ymax, :] = 2
    res[:, ymin:(ymin + ymax) // 2, :] = 3
    res *= liver_case
    return res


def get_center_of_mass(img: np.ndarray) -> Tuple[int, int, int]:
    xmin, xmax, ymin, ymax, zmin, zmax = bbox2_3D(img)
    return (xmin + xmax) // 2, (ymin + ymax) // 2, (zmin + zmax) // 2


def print_full_df(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def approximate_sphere(relevant_points_in_real_space: np.ndarray, relevant_points_in_voxel_space: np.ndarray,
                       center_of_mass_in_voxel_space: Tuple[int, int, int], approximate_radius_in_real_space: float,
                       affine_matrix: np.ndarray):
    center_of_mass_in_real_space = affines.apply_affine(affine_matrix, center_of_mass_in_voxel_space)
    final_points_in_voxel_space = relevant_points_in_voxel_space[
        ((relevant_points_in_real_space - center_of_mass_in_real_space) ** 2).sum(
            axis=1) <= approximate_radius_in_real_space ** 2]
    sphere = np.zeros(relevant_points_in_voxel_space[-1] + 1)
    sphere[final_points_in_voxel_space[:, 0], final_points_in_voxel_space[:, 1], final_points_in_voxel_space[:, 2]] = 1
    return sphere


def is_a_scan(case: np.ndarray) -> bool:
    if np.unique(case).size <= 2:
        return False
    return True


def is_a_mask(case: np.ndarray) -> bool:
    if np.any((case != 0) & (case != 1)):
        return False
    return True


def is_a_labeled_mask(case: np.ndarray, relevant_labels: Collection) -> bool:
    if np.all(np.isin(case, relevant_labels)):
        return True
    return False


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


def scans_sort_key(name, full_path_is_given=False):
    if full_path_is_given:
        name = os.path.basename(name)
    split = name.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def pairs_sort_key(name, full_path_is_given=False):
    if full_path_is_given:
        name = os.path.basename(name)
    name = name.replace('BL_', '')
    bl_name, fu_name = name.split('_FU_')
    return (*scans_sort_key(bl_name), *scans_sort_key(fu_name))


def sort_dataframe_by_key(dataframe: pd.DataFrame, column: str, key: Callable) -> pd.DataFrame:
    """ Sort a dataframe from a column using the key """
    sort_ixs = sorted(np.arange(len(dataframe)), key=lambda i: key(dataframe.iloc[i][column]))
    return pd.DataFrame(columns=list(dataframe), data=dataframe.iloc[sort_ixs].values)


def get_minimum_distance_between_CCs(mask: np.ndarray, voxel_to_real_space_trans: Optional[np.ndarray] = None,
                                     max_points_per_CC: Optional[int] = None, seed: Optional[int] = None) -> float:
    """
    Get the minimum distance between every 2 connected components in a binary image
    """

    rand = np.random.RandomState(seed)

    def dist_between_PCs_squared(pc1, pc2):
        return np.min(((np.expand_dims(pc1, -1) - np.expand_dims(pc2.T, 0)) ** 2).sum(axis=1))

    def choose_n_random_points(pc: np.ndarray, n) -> np.ndarray:
        perm = rand.permutation(pc.shape[0])
        idx = perm[:n]
        return pc[idx, :]

    def filter_and_transfer_points(pc: np.ndarray):
        if (max_points_per_CC is not None) and (pc.shape[0] > max_points_per_CC):
            pc = choose_n_random_points(pc, max_points_per_CC)
        if voxel_to_real_space_trans is not None:
            pc = affines.apply_affine(voxel_to_real_space_trans, pc)
        return pc

    mask = get_connected_components(mask > 0)

    list_of_PCs = [filter_and_transfer_points(r.coords) for r in regionprops(mask)]

    n_CCs = len(list_of_PCs)

    if n_CCs >= 2:
        return np.sqrt(np.min([dist_between_PCs_squared(list_of_PCs[i], list_of_PCs[j])
                               for i in range(n_CCs) for j in range(i + 1, n_CCs)]))

    return np.inf


def get_tumors_intersections(gt: np.ndarray, pred: np.ndarray, unique_intersections_only: bool = False) -> Dict[
    int, List[int]]:
    """
    Get intersections of tumors between GT and PRED

    :param gt: GT tumors case
    :param pred: PRED tumors case
    :param unique_intersections_only: If considering only unique intersections

    :return: a dict containing for each relevant GT tumor (key) a list with the relevant intersections (value)
    """

    # extract intersection pairs of tumors
    pairs = np.hstack([gt.reshape([-1, 1]), pred.reshape([-1, 1])])
    pairs = np.unique(pairs[~np.any(pairs == 0, axis=1)], axis=0)

    if unique_intersections_only:
        # filter out unique connections
        unique_gt = np.stack(np.unique(pairs[:, 0], return_counts=True)).T
        unique_gt = unique_gt[unique_gt[:, 1] == 1][:, 0]
        unique_pred = np.stack(np.unique(pairs[:, 1], return_counts=True)).T
        unique_pred = unique_pred[unique_pred[:, 1] == 1][:, 0]
        pairs = pairs[np.isin(pairs[:, 0], unique_gt)]
        pairs = pairs[np.isin(pairs[:, 1], unique_pred)]

    intersections = []
    previous_gt = None
    for k, gt in enumerate(pairs[:, 0]):
        if previous_gt is not None and (gt == previous_gt[0]):
            previous_gt = (gt, previous_gt[1] + [int(pairs[k, 1])])
            intersections[-1] = previous_gt
        else:
            previous_gt = (gt, [int(pairs[k, 1])])
            intersections.append(previous_gt)

    return dict(intersections)


CC = Tuple[List[int], List[int]]


def get_CCs_of_tumors_intersections(gt: np.ndarray, pred: np.ndarray) -> List[CC]:
    """
    Get Connected Components (CC) intersections of tumors between GT and PRED.

    Parameters
    ----------
    gt : numpy.ndarray
        Labeled GT tumors case.
    pred : numpy.ndarray
        Labeled PRED tumors case.

    Returns
    -------
    CCs : list of CCs
        List of CCs. Each CC is a tuple of two list of ints, indicating the gt and pred tumors in the corresponding CC,
        respectively.
    """

    # extract intersection tumors
    intersection_tumors = get_tumors_intersections(gt, pred)

    # define nodes
    define_nodes = lambda ts, extension: [f'{int(t)}{extension}' for t in np.unique(ts) if t != 0]
    gt_nodes = define_nodes(gt, '_gt')
    pred_nodes = define_nodes(pred, '_pred')

    # define edges
    edges = []
    for gt_t in intersection_tumors:
        for pred_t in intersection_tumors[gt_t]:
            edges.append((f'{int(gt_t)}_gt', f'{int(pred_t)}_pred'))

    # build the graph
    G = nx.Graph()
    G.add_nodes_from(gt_nodes, bipartite='gt')
    G.add_nodes_from(pred_nodes, bipartite='pred')
    G.add_edges_from(edges)

    # extract CCs
    CCs = []
    for cc in nx.algorithms.components.connected_components(G):
        current_gts, current_preds = [], []
        for t in cc:
            _t = int(t.split('_')[0])
            if t.endswith('_gt'):
                current_gts.append(_t)
            else:
                current_preds.append(_t)
        CCs.append((current_gts, current_preds))

    return CCs


def get_CCs_of_longitudinal_tumors_intersection(tumors_masks: List[np.ndarray]):
    """
    Get Connected Components (CC) intersections of tumors between multiple tumors masks arrays.

    Parameters
    ----------
    tumors_masks : list of ndarray
        A list containing arrays containing labeled tumors masks. The list should contain at least two arrays.

    Returns
    -------
    CCs : list of CCs
        List of CCs. Each CC is a tuple of `n` list of ints, where `n` is the size of the given `tumors_mask` list,
        indicating the tumors in the corresponding CC, respectively.
    """

    assert len(tumors_masks) >= 2

    define_nodes = lambda ts, extension: [f'{int(t)}{extension}' for t in np.unique(ts) if t != 0]

    # build the graph
    G = nx.Graph()

    # add 1st array lesion nodes
    G.add_nodes_from(define_nodes(tumors_masks[0], '_0'), bipartite='0')

    for i in range(len(tumors_masks) - 1):
        # extract intersection tumors
        intersection_tumors = get_tumors_intersections(tumors_masks[i], tumors_masks[i + 1])

        # define paired array lesion nodes
        new_nodes = define_nodes(tumors_masks[i + 1], f'_{i + 1}')

        # define edges
        edges = []
        for gt_t in intersection_tumors:
            for pred_t in intersection_tumors[gt_t]:
                edges.append((f'{int(gt_t)}_{i}', f'{int(pred_t)}_{i + 1}'))

        # add new nodes and edges to the graph
        G.add_nodes_from(new_nodes, bipartite=f'{i + 1}')
        G.add_edges_from(edges)

    # extract CCs
    CCs = []
    for cc in nx.algorithms.components.connected_components(G):
        current_ccs = []
        for _ in range(len(tumors_masks)):
            current_ccs.append([])
        for t in cc:
            label, ind = [int(c) for c in t.split('_')]
            current_ccs[ind].append(label)
        current_ccs = tuple([sorted(ccs) for ccs in current_ccs])
        CCs.append(current_ccs)

    return CCs


def __get_mean_and_std(scan: str):
    try:
        s, _ = load_nifti_data(scan)
        return s.mean(), s.std()
    except Exception:
        return np.nan, np.nan


def get_duplicate_scans(scans: List[str], multiprocess: bool = False):
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


def create_approximated_spheres(labeled_tumors_mask: nibabel.Nifti1Image,
                                desired_tumors_labels: Optional[List[int]] = None) -> np.ndarray:
    """
    Creates approximated spheres for a given labeled tumors masks.

    Parameters
    ----------
    labeled_tumors_mask : nibabel.Nifti1Image
        The desired labeled tumors mask nifti file as a nibabel.Nifti1Image object.
    desired_tumors_labels : list of int, optional
        The desired tumor labels to consider. If None, all the tumors will be considered.

    Returns
    -------
    res : numpy.ndarray
        An ndarray sape shape as given mask with all the tumors replaced with approximated spheres.
    """

    tumors = labeled_tumors_mask.get_fdata(dtype=np.float32)
    pix_dims = labeled_tumors_mask.header.get_zooms()
    voxel_volume = pix_dims[0] * pix_dims[1] * pix_dims[2]
    affine_matrix = labeled_tumors_mask.affine

    nX, nY, nZ = tumors.shape
    min_p = affines.apply_affine(affine_matrix, (0, 0, 0))
    max_p = affines.apply_affine(affine_matrix, tumors.shape)
    relevant_points_in_real_space = np.vstack([np.repeat(np.arange(min_p[0], max_p[0], pix_dims[0]), nY * nZ),
                                               np.tile(np.repeat(np.arange(min_p[1], max_p[1], pix_dims[1]), nZ), nX),
                                               np.tile(np.arange(min_p[2], max_p[2], pix_dims[2]), nX * nY)]).T
    relevant_points_in_voxel_space = np.vstack([np.repeat(np.arange(0, nX), nY * nZ),
                                                np.tile(np.repeat(np.arange(0, nY), nZ), nX),
                                                np.tile(np.arange(0, nZ), nX * nY)]).T

    tumors_to_consider = np.unique(tumors) if desired_tumors_labels is None else desired_tumors_labels
    if desired_tumors_labels is None:
        tumors_to_consider = list(tumors_to_consider[tumors_to_consider != 0])

    res = np.zeros_like(tumors)
    for t in tumors_to_consider:
        current_t = (tumors == t).astype(tumors.dtype)
        current_t_vol = current_t.sum() * voxel_volume
        current_t_diameter = approximate_diameter(current_t_vol)
        current_t_centroid = centroid(current_t)
        current_sphere = approximate_sphere(relevant_points_in_real_space, relevant_points_in_voxel_space,
                                            current_t_centroid, current_t_diameter / 2, affine_matrix)
        res[current_sphere == 1] = 1

    return res


IndexExpression3D = Tuple[slice, slice, slice]


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
    """

    # ensure images have the same shape over x and y axes
    assert np.all(np.asarray(im1.shape[0:2]) == np.asarray(im2.shape[0:2]))

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

