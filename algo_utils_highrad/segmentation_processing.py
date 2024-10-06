"""
Algorithm Utilities: This module contains various functions for image processing.
"""

from typing import Union

import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_fill_holes
from skimage.measure import label
from skimage.morphology import disk, remove_small_objects

from .segmentation_features import distance_transform_edt_for_certain_label


__all__ = ['get_connected_components', 'getLargestCC', 'get_liver_border', 'expand_labels', 'expand_per_label',
           'pre_process_segmentation']


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


def pre_process_segmentation(seg, remove_small_obs=True, min_obj_size: int = 20):
    # fill holes over 2D slices
    res = binary_fill_holes(seg, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(np.bool_))

    # removing small objects
    if remove_small_obs:
        res = remove_small_objects(res, min_size=min_obj_size)

    return res.astype(seg.dtype)
