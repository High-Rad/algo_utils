"""
Algorithm Utilities: This module contains various functions for measurements and metrics.
"""

from typing import Optional, Tuple

import numpy as np
from medpy.metric import hd, assd
from scipy.ndimage import _ni_support, generate_binary_structure, binary_erosion, distance_transform_edt
from skimage.measure import label

from .segmentation_features import crop_to_relevant_joint_bbox, get_tumors_intersections
from .algo_typing import VoxelSpacing


__all__ = ['min_distance', 'Hausdorff', 'ASSD', 'assd_and_hd', 'assd_hd_and_min_distance', 'dice', 'tp_dice',
           'approximate_diameter']


def _surface_distances(result: np.ndarray, reference: np.ndarray,
                       voxelspacing: Optional[VoxelSpacing] = None, connectivity: int = 1) -> np.ndarray:
    """
    Calculate the surface distances between the surface voxel of binary objects in result and their nearest partner
    surface voxel of a binary object in reference. The distance unit is the same as for the spacing of elements along
    each dimension, which is usually given in mm.

    Notes
    -----
    This function is copied from medpy.metric version 0.3.0.

    Parameters
    ----------
    result : np.ndarray
        Input data containing objects. Can be any type but will be converted into binary: background where 0, object
        everywhere else.
    reference : np.ndarray
        Input data containing objects. Can be any type but will be converted into binary: background where 0, object
    voxelspacing : Optional[VoxelSpacing]
        The voxelspacing in a distance unit i.e. spacing of elements along each dimension. If None, a grid spacing of
        unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface of the binary objects.


    Returns
    -------
    np.ndarray
        The distances between the surface voxel of binary objects in result and their nearest partner surface voxel of
        a binary object in reference.
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


def min_distance(result: np.ndarray, reference: np.ndarray, voxelspacing: Optional[VoxelSpacing] = None,
                 connectivity: int = 1, crop_to_relevant_scope: bool = True) -> float:
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


def Hausdorff(result: np.ndarray, reference: np.ndarray, voxelspacing: Optional[VoxelSpacing] = None,
              connectivity: int = 1, crop_to_relevant_scope: bool = True) -> float:
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


def ASSD(result: np.ndarray, reference: np.ndarray, voxelspacing: Optional[VoxelSpacing] = None,
         connectivity: int = 1, crop_to_relevant_scope: bool = True) -> float:
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


def assd_and_hd(result: np.ndarray, reference: np.ndarray, voxelspacing: Optional[VoxelSpacing] = None,
                connectivity: int = 1, crop_to_relevant_scope: bool = True) -> Tuple[float, float]:
    """
    The concept is taken from medpy.metric.assd and medpy.metric.hd

    Average symmetric surface distance and Hausdorff Distance.

    Computes the average symmetric surface distance (ASSD) and the (symmetric) Hausdorff Distance (HD) between the
    binary objects in two images.

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


def assd_hd_and_min_distance(result: np.ndarray, reference: np.ndarray, voxelspacing: Optional[VoxelSpacing] = None,
                             connectivity: int = 1, crop_to_relevant_scope: bool = True) -> Tuple[float, float, float]:
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


def dice(gt_seg: np.ndarray, prediction_seg: np.ndarray, val_if_both_are_empty: float = 1.0) -> float:
    """
    Calculates the Dice Coefficient between the two given segmentations. The Dice Coefficient is defined as
    :math:`2 * |A \cap B| / (|A| + |B|)` where :math:`A` is the ground truth segmentation and :math:`B` is the
    prediction segmentation.

    Notes
    -----
    The Dice Coefficient is symmetric and ranges from 0 to 1, where 1 indicates a perfect overlap and 0 indicates no
    overlap at all. The Dice Coefficient is a real metric. The binary images can therefore be supplied in any order.

    Parameters
    ----------
    gt_seg : np.ndarray
        The ground truth segmentation.
    prediction_seg : np.ndarray
        The prediction segmentation to compare with the ground truth segmentation.
    val_if_both_are_empty : float, default 1.0
        The value to return if both segmentations are empty. By default, it returns 1.0.

    Returns
    -------
    float
        The Dice Coefficient between the two segmentations.
    """
    seg1 = np.asarray(gt_seg).astype(np.bool_)
    seg2 = np.asarray(prediction_seg).astype(np.bool_)

    # Compute Dice coefficient
    intersection = np.logical_and(seg1, seg2)
    denominator_of_res = seg1.sum() + seg2.sum()
    if denominator_of_res == 0:
        return val_if_both_are_empty
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


def approximate_diameter(volume: float) -> float:
    """
    Approximate the diameter of a sphere from its volume.
    The formula is :math:`d = 2 * r` where :math:`r = ((3 * V) / (4 * \pi))^{1/3}`.

    Parameters
    ----------
    volume : float
        The volume of the sphere.

    Returns
    -------
    float
        The diameter of the sphere.
    """
    r = ((3 * volume) / (4 * np.pi)) ** (1 / 3)
    diameter = 2 * r
    return diameter
