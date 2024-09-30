from typing import Tuple, Optional, Dict, List, Union

from skimage.measure import centroid
import networkx as nx
import nibabel
import numpy as np
from nibabel import affines
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, label

from algo_typing import CC, IndexExpression3D, VoxelSpacing



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


def get_minimum_distance_between_CCs(mask: np.ndarray, voxel_to_real_space_trans: Optional[np.ndarray] = None,
                                     max_points_per_CC: Optional[int] = None, seed: Optional[int] = None,
                                     connectivity: Optional[int] = None) -> float:
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

    mask = label(mask > 0, connectivity=connectivity)

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
        current_t_diameter = 2 * ((3 * current_t_vol) / (4 * np.pi)) ** (1 / 3) # diameter = 2 * (3V / 4pi)^(1/3)
        current_t_centroid = centroid(current_t)
        current_sphere = approximate_sphere(relevant_points_in_real_space, relevant_points_in_voxel_space,
                                            current_t_centroid, current_t_diameter / 2, affine_matrix)
        res[current_sphere == 1] = 1

    return res
